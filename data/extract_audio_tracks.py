import subprocess
import glob
import os
import sys

def extractAudioFromVideo(video, targetAudio):
    if not os.path.exists(targetAudio):
        command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (video, targetAudio))
        subprocess.call(command, shell=True, stdout=None)

if __name__ == '__main__':
    if len(sys.argv) == 3:
        ava_video_dir = sys.argv[1]
        target_audios = sys.argv[2]
    else:
        ava_video_dir = '/home/azureuser/gitm/active-speakers-context/topshot/trainval'
        target_audios = '/home/azureuser/gitm/active-speakers-context/topshot/audio_tracks'
    if not os.path.exists(target_audios):
        os.makedirs(target_audios)

    all_videos = os.listdir(ava_video_dir)
    all_videos = [v.split('.')[0] for v in all_videos]

    for video_name in all_videos:
        print('process video ', video_name)
        actual_file_name = glob.glob(os.path.join(ava_video_dir, video_name+'*'))
        print(actual_file_name)
        extractAudioFromVideo(actual_file_name[0], os.path.join(target_audios,
                              video_name+'.wav'))
