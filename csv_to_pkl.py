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

#Written for simplicity, paralelize/shard as you wish
if __name__ == '__main__':
    clip_lenght = int(sys.argv[1])
    cuda_device_number = str(sys.argv[2])
    video_val_path = sys.argv[3]
    audio_val_path = sys.argv[4]
    csv_path = sys.argv[5]
    video_name = sys.argv[6]

    image_size = (144, 144) #Dont forget to assign this same size on ./core/custom_transforms

    model_weights = '/home/azureuser/gitm/active-speakers-context/model_output/ste_encoder/48.pth'
    target_directory = '/home/azureuser/git/GraVi-T/data/features/RESNET18-TSM-AUG/val/'
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

    #train_videos = load_train_video_set()
    val_videos = [video_name]

    global_id = 0
    for video_key in val_videos:
        data = {}
        print(f'forward video {video_key}')
        target_pkl = os.path.join(target_directory, video_key + ".pkl")
        # if os.path.exists(target_pkl):
        #     print(f"skip {video_key}")
        #     continue

        d_val = AudioVideoDatasetAuxLossesForwardPhase(video_key, audio_val_path, video_val_path,
                                        csv_path, clip_lenght,
                                        image_size, video_data_transforms['val'],
                                        do_video_augment=False)

        dl_val = DataLoader(d_val, batch_size=opt_config['batch_size'],
                            shuffle=False, num_workers=opt_config['threads'])

        print(f'data size {len(dl_val)}')
        for idx, dl in enumerate(dl_val):
            audio_data, video_data, video_id, ts, entity_id, bbox, gt = dl
            # video_data = video_data.view(1*clip_lenght, 3, 144, 144)
            video_data = video_data.to(device)
            audio_data = audio_data.to(device)

            ts = f"{round(float(ts[0]), 2):g}"
            with torch.set_grad_enabled(False):
                preds, _, _, feats = backbone(audio_data, video_data)
                feats = feats.detach().cpu().numpy()[0]
                # vf_writer.writerow([video_id[0], ts[0], entity_id[0], float(gt[0]), float(preds[0][0]), float(preds[0][1]), list(feats)])
                entry = {
                            'person_box': f'{bbox[0][0]},{bbox[1][0]},{bbox[2][0]},{bbox[3][0]}',
                            'person_id': entity_id[0],
                            'global_id': global_id,
                            'feature': feats,
                            'label': 0
                        
                }

                global_id += 1
                if ts not in data.keys():
                    data[ts] = []

                data[ts].append(entry)

        with open(target_pkl, 'wb') as f:
            pickle.dump(data, f)