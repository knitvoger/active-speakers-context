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
    image_size = (144, 144) #Dont forget to assign this same size on ./core/custom_transforms

    model_weights = sys.argv[3]
    video_list_file = sys.argv[4]
    kind = sys.argv[5]
    target_directory = sys.argv[6]

    io_config = exp_conf.STE_inputs
    opt_config = exp_conf.STE_forward_params
    opt_config['batch_size'] = 1
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # cuda config
    backbone = configure_backbone_forward_phase(opt_config['backbone'], model_weights, clip_lenght)
    has_cuda = torch.cuda.is_available()
    device = torch.device('cuda:'+cuda_device_number if has_cuda else 'cpu')
    backbone = backbone.to(device)

    video_data_transforms = {
        'val': ct.video_val
    }

    video_val_path = os.path.join(io_config['video_dir'], kind)
    audio_val_path = os.path.join(io_config['audio_dir'], kind)

    with open(video_list_file, "r") as f:
        val_videos = [line.strip() for line in f if line.strip()]

    for video_key in val_videos:
        print(f'forward video {video_key}')
        target_pkl = os.path.join(target_directory, video_key + ".pkl")
        if os.path.exists(target_pkl):
            print(f"skip {video_key}")
            continue

        # load original pkl from spell project
        with open(f"/home/azureuser/git/GraVi-T/data/features/RESNET18-TSM-AUG_original/{kind}/{video_key}.pkl", "rb") as f:
            tracks = pickle.load(f)

        # with open(target_directory+video_key+'.csv', mode='w') as vf:
            # vf_writer = csv.writer(vf, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        d_val = AudioVideoDatasetAuxLossesForwardPhase(video_key, audio_val_path, video_val_path,
                                        io_config[f'csv_{kind}_full'], clip_lenght,
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

            with torch.set_grad_enabled(False):
                preds, _, _, feats = backbone(audio_data, video_data)
                feats = feats.detach().cpu().numpy()[0]
                # vf_writer.writerow([video_id[0], ts[0], entity_id[0], float(gt[0]), float(preds[0][0]), float(preds[0][1]), list(feats)])

                timestamp = ts[0].replace(".0", "") if ts[0].endswith(".0") else ts[0]
                if timestamp in tracks.keys():
                    found_person = False
                    for person in tracks[timestamp]:
                        same_person = True
                        person_box = [float(x) for x in person['person_box'].split(',')]
                        for i in range(4):
                            if round(float(bbox[i][0]), 4) != round(person_box[i], 4):
                                same_person = False
                                break
                        if same_person:
                            found_person = True
                            person['feature'] = feats
                            break

                    if not found_person:
                        raise RuntimeError("Person not found!")
                else:
                    raise RuntimeError("Unexpected timestamp!")
        
        with open(target_pkl, 'wb') as f:
            pickle.dump(tracks, f)
