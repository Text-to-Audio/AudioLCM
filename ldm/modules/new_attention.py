from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from ldm.modules.diffusionmodules.util import checkpoint


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)



class Conv1dGEGLU(nn.Module):
    def __init__(self, dim_in, dim_out,kernel_size = 9):
        super().__init__()
        self.proj = nn.Conv1d(dim_in, dim_out * 2,kernel_size=kernel_size,padding=kernel_size//2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=1)
        return x * F.gelu(gate)

class Conv1dFeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.,kernel_size = 9):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Conv1d(dim, inner_dim,kernel_size=kernel_size,padding=kernel_size//2),
            nn.GELU()
        ) if not glu else Conv1dGEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Conv1d(inner_dim, dim_out,kernel_size=kernel_size,padding=kernel_size//2)
        )

    def forward(self, x): # x shape (B,C,T)
        return self.net(x)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.zero-initializing the final convolutional layer in each block prior to any residual connections can accelerate training. 
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):# 如果设置了context_dim就不是自注意力了
        super().__init__()
        inner_dim = dim_head * heads # inner_dim == SpatialTransformer.model_channels
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):# x:(b,T,C), context:(b,seq_len,context_dim)
        h = self.heads

        q = self.to_q(x)# q:(b,T,inner_dim)
        context = default(context, x)
        k = self.to_k(context)# (b,seq_len,inner_dim)
        v = self.to_v(context)# (b,seq_len,inner_dim)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))# n is seq_len for k and v

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale # (b*head,T,seq_len)

        if exists(mask):# false
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)# (b*head,T,inner_dim/head)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)# (b,T,inner_dim)
        return self.to_out(out)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True): # 1 self 1 cross or 2 self
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention,if context is none
        self.ff = Conv1dFeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # use as cross attention
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):# x shape:(B,T,C)
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x

        x = self.ff(self.norm3(x).permute(0,2,1)).permute(0,2,1) + x
        return x

class TemporalTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head 
        self.norm = Normalize(in_channels)
        
        self.proj_in = nn.Conv1d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv1d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))# initialize with zero

    def forward(self, x, context=None):# x shape (b,c,t)
        # note: if no context is given, cross-attention defaults to self-attention
        x_in = x
        x = self.norm(x)# group norm
        x = self.proj_in(x)# no shape change
        x = rearrange(x,'b c t -> b t c')
        for block in self.transformer_blocks:
            x = block(x, context=context)# context shape [b,seq_len=77,context_dim]
        x = rearrange(x,'b t c -> b c t')
        
        x = self.proj_out(x)
        return x + x_in

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens,  max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, 
            torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, x):
        x = x + self.P[:, :x.shape[1], :].to(x.device)
        return x
    
class PositionEmbedding(nn.Module):
    MODE_EXPAND = 'MODE_EXPAND'
    MODE_ADD = 'MODE_ADD'
    MODE_CONCAT = 'MODE_CONCAT'
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 mode=MODE_ADD):
        super(PositionEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        if self.mode == self.MODE_EXPAND:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings * 2 + 1, embedding_dim))
        else:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # use xavier_normal_ to initialize
        torch.nn.init.xavier_normal_(self.weight)
        # use sin cons to initialize
        # X = torch.arange(self.num_embeddings, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, 
        #     torch.arange(0, self.embedding_dim, 2, dtype=torch.float32) / self.embedding_dim)
        # init = torch.Tensor(self.num_embeddings,self.embedding_dim)
        # init[:, 0::2] = torch.sin(X)
        # init[:, 1::2] = torch.cos(X)   
        # self.weight.data.copy_(init)

    def forward(self, x):
        if self.mode == self.MODE_EXPAND:
            indices = torch.clamp(x, -self.num_embeddings, self.num_embeddings) + self.num_embeddings
            return F.embedding(indices.type(torch.LongTensor), self.weight)
        batch_size, seq_len = x.size()[:2]
        embeddings = self.weight[:seq_len, :].view(1, seq_len, self.embedding_dim)
        if self.mode == self.MODE_ADD:
            return x + embeddings
        if self.mode == self.MODE_CONCAT:
            return torch.cat((x, embeddings.repeat(batch_size, 1, 1)), dim=-1)
        raise NotImplementedError('Unknown mode: %s' % self.mode)

    def extra_repr(self):
        return 'num_embeddings={}, embedding_dim={}, mode={}'.format(
            self.num_embeddings, self.embedding_dim, self.mode,
        )

class TemporalTransformerSkip(TemporalTransformer):
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__(in_channels, n_heads, d_head,
                 depth, dropout, context_dim)
        self.skip_linear = nn.Linear(2 * in_channels, in_channels)

    def forward(self, x,skip, context=None):# x shape (b,c,t)
        # note: if no context is given, cross-attention defaults to self-attention
        x_in = x
        x = self.norm(x)# group norm
        x = self.proj_in(x)# no shape change
        x = rearrange(x,'b c t -> b t c')
        skip = rearrange(skip,'b c t -> b t c')
        x  = self.skip_linear(torch.cat([x,skip],dim=-1))
        for block in self.transformer_blocks:
            x = block(x, context=context)# context shape [b,seq_len=77,context_dim]
        x = rearrange(x,'b t c -> b c t')
        
        x = self.proj_out(x)
        return x + x_in
        