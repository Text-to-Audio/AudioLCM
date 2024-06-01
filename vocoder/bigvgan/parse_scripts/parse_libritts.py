# Copyright (c) 2022 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

import os, glob

def get_wav_and_text_filelist(data_root, data_type, subsample=1):
    wav_list = sorted([path.replace(data_root, "")[1:] for path in glob.glob(os.path.join(data_root, data_type, "**/**/*.wav"))])
    wav_list = wav_list[::subsample]
    txt_filelist = [path.replace('.wav', '.normalized.txt') for path in wav_list]

    txt_list = []
    for txt_file in txt_filelist:
        with open(os.path.join(data_root, txt_file), 'r') as f_txt:
            text = f_txt.readline().strip('\n')
            txt_list.append(text)
    wav_list = [path.replace('.wav', '') for path in wav_list]

    return wav_list, txt_list

def write_filelist(output_path, wav_list, txt_list):
    with open(output_path, 'w') as f:
        for i in range(len(wav_list)):
            filename = wav_list[i] + '|' + txt_list[i]
            f.write(filename + '\n')

if __name__ == "__main__":

    data_root = "LibriTTS"

    # dev and test sets. subsample each sets to get ~100 utterances
    data_type_list = ["dev-clean", "dev-other", "test-clean", "test-other"]
    subsample_list = [50, 50, 50, 50]
    for (data_type, subsample) in zip(data_type_list, subsample_list):
        print("processing {}".format(data_type))
        data_path = os.path.join(data_root, data_type)
        assert os.path.exists(data_path),\
            "path {} not found. make sure the path is accessible by creating the symbolic link using the following command: "\
            "ln -s /path/to/your/{} {}".format(data_path, data_path, data_path)
        wav_list, txt_list = get_wav_and_text_filelist(data_root, data_type, subsample)
        write_filelist(os.path.join(data_root, data_type+".txt"), wav_list, txt_list)

    # training and seen speaker validation datasets (libritts-full): train-clean-100 + train-clean-360 + train-other-500
    wav_list_train, txt_list_train = [], []
    for data_type in ["train-clean-100", "train-clean-360", "train-other-500"]:
        print("processing {}".format(data_type))
        data_path = os.path.join(data_root, data_type)
        assert os.path.exists(data_path),\
            "path {} not found. make sure the path is accessible by creating the symbolic link using the following command: "\
            "ln -s /path/to/your/{} {}".format(data_path, data_path, data_path)
        wav_list, txt_list = get_wav_and_text_filelist(data_root, data_type)
        wav_list_train.extend(wav_list)
        txt_list_train.extend(txt_list)

    # split the training set so that the seen speaker validation set contains ~100 utterances
    subsample_val = 3000
    wav_list_val, txt_list_val = wav_list_train[::subsample_val], txt_list_train[::subsample_val]
    del wav_list_train[::subsample_val]
    del txt_list_train[::subsample_val]
    write_filelist(os.path.join(data_root, "train-full.txt"), wav_list_train, txt_list_train)
    write_filelist(os.path.join(data_root, "val-full.txt"), wav_list_val, txt_list_val)

    print("done")