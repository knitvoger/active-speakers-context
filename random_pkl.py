import os
import csv
import sys
import torch

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from core.dataset import AudioVideoDatasetAuxLossesForwardPhase
from core.optimization import optimize_av_losses
from core.io import set_up_log_and_ws_out
from core.util import configure_backbone_forward_phase, load_train_video_set, load_val_video_set

import core.custom_transforms as ct
import core.config as exp_conf
import pickle
import string
import random

def random_string(length=10):
    chars = string.ascii_letters + string.digits  # a-z, A-Z, 0-9
    return ''.join(random.choice(chars) for _ in range(length))

#Written for simplicity, paralelize/shard as you wish
if __name__ == '__main__':
    clip_lenght = int(sys.argv[1])
    cuda_device_number = str(sys.argv[2])
    image_size = (144, 144) #Dont forget to assign this same size on ./core/custom_transforms

    model_weights = '/home/azureuser/gitm/active-speakers-context/model_output/ste_encoder/48.pth'
    target_directory = '/home/azureuser/gitm/active-speakers-context/avadata/randomglobalid/18_11/train'
    io_config = exp_conf.STE_inputs
    opt_config = exp_conf.STE_forward_params
    opt_config['batch_size'] = 1

    # cuda config
    backbone = configure_backbone_forward_phase(opt_config['backbone'], model_weights, clip_lenght)
    has_cuda = torch.cuda.is_available()
    device = torch.device('cuda:'+cuda_device_number if has_cuda else 'cpu')
    backbone = backbone.to(device)

    video_data_transforms = {
        'val': ct.video_val
    }

    video_val_path = os.path.join(io_config['video_dir'], 'train')
    audio_val_path = os.path.join(io_config['audio_dir'], 'train')

    #train_videos = load_train_video_set()
    with open("/home/azureuser/gitm/active-speakers-context/train_real_video_list.txt", "r") as f:
        val_videos = [line.strip() for line in f if line.strip()]

    i = 0
    for video_key in val_videos:
        print(f'forward video {video_key}')
        target_pkl = os.path.join(target_directory, video_key + ".pkl")
        if os.path.exists(target_pkl):
            print(f"skip {video_key}")
            continue

        # load original pkl from spell project
        with open(f"/home/azureuser/git/GraVi-T/data/features/RESNET18-TSM-AUG_my_18_11_no_color/train/{video_key}.pkl", "rb") as f:
            tracks = pickle.load(f)
        for ts, persons in tracks.items():
            for person in persons:
                person['global_id'] = i
                i += 1
        
        with open(target_pkl, 'wb') as f:
            pickle.dump(tracks, f)
