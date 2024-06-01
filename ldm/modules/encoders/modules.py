import torch
import torch.nn as nn
from functools import partial
from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test
from torch.utils.checkpoint import checkpoint
from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel, AutoTokenizer
from importlib_resources import files
from ldm.modules.encoders.CLAP.utils import read_config_as_args
from ldm.modules.encoders.CLAP.clap import TextEncoder
import copy
from ldm.util import default, count_params
import pytorch_lightning as pl

class AbstractEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]# (bsz,1)
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):# 这里不是用的pretrained bert,是用的transformers的BertTokenizer加自定义的TransformerWrapper
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class FrozenT5Embedder(AbstractEncoder):
    """Uses the T5 transformer encoder for text"""
    def __init__(self, version="google/t5-v1_1-large", device="cuda", max_length=77, freeze=True):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length   # TODO: typical value?
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)

class FrozenFLANEmbedder(AbstractEncoder):
    """Uses the T5 transformer encoder for text"""
    def __init__(self, version="google/flan-t5-large", device="cuda", max_length=77, freeze=True):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length   # TODO: typical value?
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)# tango的flanT5是不定长度的batch，这里做成定长的batch
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)

class FrozenCLAPEmbedder(AbstractEncoder):
    """Uses the CLAP transformer encoder for text from microsoft"""
    def __init__(self, weights_path, freeze=True, device="cuda", max_length=77):  # clip-vit-base-patch32
        super().__init__()

        model_state_dict = torch.load(weights_path, map_location=torch.device('cpu'))['model']
        match_params = dict()
        for key in list(model_state_dict.keys()):
            if 'caption_encoder' in key:
                match_params[key.replace('caption_encoder.', '')] = model_state_dict[key]

        config_as_str = files('ldm').joinpath('modules/encoders/CLAP/config.yml').read_text()
        args = read_config_as_args(config_as_str, is_config_str=True)

        # To device
        self.tokenizer = AutoTokenizer.from_pretrained(args.text_model) # args.text_model
        self.caption_encoder = TextEncoder(
            args.d_proj, args.text_model, args.transformer_embed_dim
        )

        self.max_length = max_length
        self.device = device
        if freeze: self.freeze()

        print(f"{self.caption_encoder.__class__.__name__} comes with {count_params(self.caption_encoder) * 1.e-6:.2f} M params.")

    def freeze(self):# only freeze
        self.caption_encoder.base = self.caption_encoder.base.eval()
        for param in self.caption_encoder.base.parameters():
            param.requires_grad = False


    def encode(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)

        outputs = self.caption_encoder.base(input_ids=tokens)
        z = self.caption_encoder.projection(outputs.last_hidden_state)
        return z

class FrozenLAIONCLAPEmbedder(AbstractEncoder):
    """Uses the CLAP transformer encoder for text from LAION-AI"""
    def __init__(self, weights_path, freeze=True,sentence=False, device="cuda", max_length=77):  # clip-vit-base-patch32
        super().__init__()
        # To device
        from transformers import RobertaTokenizer
        from ldm.modules.encoders.open_clap import create_model
        self.sentence = sentence

        model, model_cfg = create_model(
            'HTSAT-tiny',
            'roberta',
            weights_path,
            enable_fusion=True,
            fusion_type='aff_2d'
        )

        del model.audio_branch, model.audio_transform, model.audio_projection
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = model

        self.max_length = max_length
        self.device = device
        self.to(self.device)
        if freeze: self.freeze()

        param_num = sum(p.numel() for p in model.parameters())
        print(f'{self.model.__class__.__name__} comes with: {param_num / 1e6:.3f} M params.')

    def to(self,device):
        self.model.to(device=device)
        self.device=device

    def freeze(self):
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def encode(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt").to(self.device)
        if self.sentence:
            z = self.model.get_text_embedding(batch_encoding).unsqueeze(1)
        else:
            # text_branch is roberta
            outputs = self.model.text_branch(input_ids=batch_encoding["input_ids"].to(self.device), attention_mask=batch_encoding["attention_mask"].to(self.device))
            z = self.model.text_projection(outputs.last_hidden_state)
        
        return z
    
class FrozenLAIONCLAPSetenceEmbedder(AbstractEncoder):
    """Uses the CLAP transformer encoder for text from LAION-AI"""
    def __init__(self, weights_path, freeze=True, device="cuda", max_length=77):  # clip-vit-base-patch32
        super().__init__()
        # To device
        from transformers import RobertaTokenizer
        from ldm.modules.encoders.open_clap import create_model


        model, model_cfg = create_model(
            'HTSAT-tiny',
            'roberta',
            weights_path,
            enable_fusion=True,
            fusion_type='aff_2d'
        )

        del model.audio_branch, model.audio_transform, model.audio_projection
        self.tokenize = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = model

        self.max_length = max_length
        self.device = device
        if freeze: self.freeze()

        param_num = sum(p.numel() for p in model.parameters())
        print(f'{self.model.__class__.__name__} comes with: {param_num / 1e+6:.3f} M params.')

    def freeze(self):
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def tokenizer(self, text):
        result = self.tokenize(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        return result

    def encode(self, text):
        with torch.no_grad():
            # the 'fusion' truncate mode can be changed to 'rand_trunc' if run in unfusion mode
            text_data = self.tokenizer(text)# input_ids shape:(b,512)
            embed = self.model.get_text_embedding(text_data)
        embed = embed.unsqueeze(1)# (b,1,512)
        return embed

class FrozenCLAPOrderEmbedder2(AbstractEncoder):# 每个object后面都加上|
    """Uses the CLAP transformer encoder for text (from huggingface)"""
    def __init__(self, weights_path, freeze=True, device="cuda"):
        super().__init__()

        model_state_dict = torch.load(weights_path, map_location=torch.device('cpu'))['model']
        match_params = dict()
        for key in list(model_state_dict.keys()):
            if 'caption_encoder' in key:
                match_params[key.replace('caption_encoder.', '')] = model_state_dict[key]

        config_as_str = files('ldm').joinpath('modules/encoders/CLAP/config.yml').read_text()
        args = read_config_as_args(config_as_str, is_config_str=True)

        # To device
        self.tokenizer = AutoTokenizer.from_pretrained(args.text_model) # args.text_model
        self.caption_encoder = TextEncoder(
            args.d_proj, args.text_model, args.transformer_embed_dim
        ).to(device)
        self.max_objs = 10
        self.max_length = args.text_len
        self.device = device
        self.order_to_label = self.build_order_dict()
        if freeze: self.freeze()

        print(f"{self.caption_encoder.__class__.__name__} comes with {count_params(self.caption_encoder) * 1.e-6:.2f} M params.")

    def freeze(self):
        self.caption_encoder.base = self.caption_encoder.base.eval()
        for param in self.caption_encoder.base.parameters():
            param.requires_grad = False

    def build_order_dict(self):
        order2label = {}
        num_orders = 10
        time_stamps = ['start','mid','end']
        time_num = len(time_stamps)
        for i in range(num_orders):
            for j,time_stamp in enumerate(time_stamps):
                order2label[f'order {i} {time_stamp}'] = i * time_num + j
        order2label['all'] = num_orders*len(time_stamps)
        order2label['unknown'] = num_orders*len(time_stamps) + 1
        return order2label

    def encode(self, text):
        obj_list,orders_list = [],[]
        for raw in text:
            splits = raw.split('@') # raw example: '<man speaking& order 1 start>@<man speaking& order 2 mid>@<idle engine& all>'
            objs = []
            orders = []
            for split in splits:# <obj& order>
                split = split[1:-1]
                obj,order = split.split('&')
                objs.append(obj.strip())
                try:
                    orders.append(self.order_to_label[order.strip()])
                except:
                    print(order.strip(),raw)
            assert len(objs) == len(orders)
            obj_list.append(' | '.join(objs)+' |')# '|' after every word
            orders_list.append(orders)
        batch_encoding = self.tokenizer(obj_list, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"]

        outputs = self.caption_encoder.base(input_ids=tokens.to(self.device))
        z = self.caption_encoder.projection(outputs.last_hidden_state)
        return {'token_embedding':z,'token_ids':tokens,'orders':orders_list}
    
class FrozenCLAPOrderEmbedder3(AbstractEncoder):# 相比于FrozenCLAPOrderEmbedder2移除了projection,使用正确的max_len,去除了order仅保留时间。
    """Uses the CLAP transformer encoder for text (from huggingface)"""
    def __init__(self, weights_path, freeze=True, device="cuda"):  # clip-vit-base-patch32
        super().__init__()

        model_state_dict = torch.load(weights_path, map_location=torch.device('cpu'))['model']
        match_params = dict()
        for key in list(model_state_dict.keys()):
            if 'caption_encoder' in key:
                match_params[key.replace('caption_encoder.', '')] = model_state_dict[key]

        config_as_str = files('ldm').joinpath('modules/encoders/CLAP/config.yml').read_text()
        args = read_config_as_args(config_as_str, is_config_str=True)

        # To device
        self.tokenizer = AutoTokenizer.from_pretrained(args.text_model) # args.text_model
        self.caption_encoder = TextEncoder(
            args.d_proj, args.text_model, args.transformer_embed_dim
        ).to(device)
        self.max_objs = 10
        self.max_length = args.text_len
        self.device = device
        self.order_to_label = self.build_order_dict()
        if freeze: self.freeze()

        print(f"{self.caption_encoder.__class__.__name__} comes with {count_params(self.caption_encoder) * 1.e-6:.2f} M params.")

    def freeze(self):
        self.caption_encoder.base = self.caption_encoder.base.eval()
        for param in self.caption_encoder.base.parameters():
            param.requires_grad = False

    def build_order_dict(self):
        order2label = {}
        time_stamps = ['all','start','mid','end']
        for i,time_stamp in enumerate(time_stamps):
            order2label[time_stamp] = i
        return order2label

    def encode(self, text):
        obj_list,orders_list = [],[]
        for raw in text:
            splits = raw.split('@') # raw example: '<man speaking& order 1 start>@<man speaking& order 2 mid>@<idle engine& all>'
            objs = []
            orders = []
            for split in splits:# <obj& order>
                split = split[1:-1]
                obj,order = split.split('&')
                objs.append(obj.strip())
                try:
                    orders.append(self.order_to_label[order.strip()])
                except:
                    print(order.strip(),raw)
            assert len(objs) == len(orders)
            obj_list.append(' | '.join(objs)+' |')# '|' after every word
            orders_list.append(orders)
        batch_encoding = self.tokenizer(obj_list, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"]
        attn_mask = batch_encoding["attention_mask"]
        outputs = self.caption_encoder.base(input_ids=tokens.to(self.device))
        z = outputs.last_hidden_state
        return {'token_embedding':z,'token_ids':tokens,'orders':orders_list,'attn_mask':attn_mask}

class FrozenCLAPT5Embedder(AbstractEncoder):
    """Uses the CLAP transformer encoder for text from microsoft"""
    def __init__(self, weights_path,t5version="google/flan-t5-large", freeze=True, device="cuda", max_length=77):  # clip-vit-base-patch32
        super().__init__()

        model_state_dict = torch.load(weights_path, map_location=torch.device('cpu'))['model']
        match_params = dict()
        for key in list(model_state_dict.keys()):
            if 'caption_encoder' in key:
                match_params[key.replace('caption_encoder.', '')] = model_state_dict[key]

        config_as_str = files('ldm').joinpath('modules/encoders/CLAP/config.yml').read_text()
        args = read_config_as_args(config_as_str, is_config_str=True)

        self.clap_tokenizer = AutoTokenizer.from_pretrained(args.text_model) # args.text_model
        self.caption_encoder = TextEncoder(
            args.d_proj, args.text_model, args.transformer_embed_dim
        )
    
        self.t5_tokenizer = T5Tokenizer.from_pretrained(t5version)
        self.t5_transformer = T5EncoderModel.from_pretrained(t5version)

        self.max_length = max_length
        self.to(device=device)
        if freeze: self.freeze()

        print(f"{self.caption_encoder.__class__.__name__} comes with {count_params(self.caption_encoder) * 1.e-6:.2f} M params.")

    def freeze(self):
        self.caption_encoder = self.caption_encoder.eval()
        for param in self.caption_encoder.parameters():
            param.requires_grad = False

    def to(self,device):
        self.t5_transformer.to(device)
        self.caption_encoder.to(device)
        self.device = device

    def encode(self, text):
        ori_caption = text['ori_caption']
        struct_caption = text['struct_caption']
        # print(ori_caption,struct_caption)
        clap_batch_encoding = self.clap_tokenizer(ori_caption, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        ori_tokens = clap_batch_encoding["input_ids"].to(self.device)
        t5_batch_encoding = self.t5_tokenizer(struct_caption, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        struct_tokens = t5_batch_encoding["input_ids"].to(self.device)
        outputs = self.caption_encoder.base(input_ids=ori_tokens)
        z = self.caption_encoder.projection(outputs.last_hidden_state)
        z2 = self.t5_transformer(input_ids=struct_tokens).last_hidden_state
        return torch.concat([z,z2],dim=1)


class FrozenCLAPFLANEmbedder(AbstractEncoder):
    """Uses the CLAP transformer encoder for text from microsoft"""
    def __init__(self, weights_path,t5version="../ldm/modules/encoders/CLAP/t5-v1_1-large", freeze=True, device="cuda", max_length=77):  # clip-vit-base-patch32
        super().__init__()

        model_state_dict = torch.load(weights_path, map_location=torch.device('cpu'))['model']
        match_params = dict()
        for key in list(model_state_dict.keys()):
            if 'caption_encoder' in key:
                match_params[key.replace('caption_encoder.', '')] = model_state_dict[key]

        config_as_str = files('ldm').joinpath('modules/encoders/CLAP/config.yaml').read_text()
        args = read_config_as_args(config_as_str, is_config_str=True)

        self.clap_tokenizer = AutoTokenizer.from_pretrained(args.text_model) # args.text_model
        self.caption_encoder = TextEncoder(
            args.d_proj, args.text_model, args.transformer_embed_dim
        )
    
        self.t5_tokenizer = T5Tokenizer.from_pretrained(t5version)
        self.t5_transformer = T5EncoderModel.from_pretrained(t5version)

        self.max_length = max_length
        # self.to(device=device)
        if freeze: self.freeze()

        print(f"{self.caption_encoder.__class__.__name__} comes with {count_params(self.caption_encoder) * 1.e-6:.2f} M params.")

    def freeze(self):
        self.caption_encoder = self.caption_encoder.eval()
        for param in self.caption_encoder.parameters():
            param.requires_grad = False

    def to(self,device):
        self.t5_transformer.to(device)
        self.caption_encoder.to(device)
        self.device = device

    def encode(self, text):
        ori_caption = text['ori_caption']
        struct_caption = text['struct_caption']
        # print(ori_caption,struct_caption)
        clap_batch_encoding = self.clap_tokenizer(ori_caption, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        ori_tokens = clap_batch_encoding["input_ids"].to(self.device)
        t5_batch_encoding = self.t5_tokenizer(struct_caption, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        struct_tokens = t5_batch_encoding["input_ids"].to(self.device)
        # if self.caption_encoder.device != ori_tokens.device:
        # self.to(self.device)
        outputs = self.caption_encoder.base(input_ids=ori_tokens)
        z = self.caption_encoder.projection(outputs.last_hidden_state)
        z2 = self.t5_transformer(input_ids=struct_tokens).last_hidden_state
        return torch.concat([z,z2],dim=1)
