import json
import os
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import math
import torch as th
import multiprocessing as mp

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = th.FloatTensor(mean).view(1, 3, 1, 1)
        self.std = th.FloatTensor(std).view(1, 3, 1, 1)

    def __call__(self, tensor):
        tensor = (tensor - self.mean) / (self.std + 1e-8)
        return tensor

### Borrow pre-processor from
class Preprocessing(object):
    def __init__(self, type):
        self.type = type
        if type == "2d":
            self.norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif type == "3d":
            self.norm = Normalize(mean=[110.6, 103.2, 96.3], std=[1.0, 1.0, 1.0])

    def _zero_pad(self, tensor, size):
        n = size - len(tensor) % size
        if n == size:
            return tensor
        else:
            z = th.zeros(n, tensor.shape[1], tensor.shape[2], tensor.shape[3])
            return th.cat((tensor, z), 0)

    def __call__(self, tensor):
        if self.type == "2d":
            tensor = tensor / 255.0
            tensor = self.norm(tensor)
        elif self.type == "3d":
            # tensor = self._zero_pad(tensor, 16)
            tensor = self.norm(tensor)
            # tensor = tensor.view(-1, 16, 3, 112, 112)
            # tensor = tensor.transpose(1, 2)
        return tensor

preprocess = Preprocessing("3d")
from modules.Timesformer.timesformer.models.vit import TimeSformer

model = TimeSformer(
    img_size=224,
    num_classes=400,
    num_frames=32,
    attention_type="divided_space_time",
    pretrained_model="/4TBssd/NSVA/Timesformer/models/TimeSformer_divST_32x32_224_HowTo100M.pyth",
).cuda()


model = model.cuda()
torch.backends.cudnn.benchmark = False
model.eval()


def main(file_path, out_path, mp4_path):
    global model


    if not os.path.exists(out_path):
        os.mkdir(out_path)

    mp4 = file_path

    """Some videos are too short or too-long (taking too much memory)"""
    # if (
    # mp4
    # == "2018-12-12-0021800408-517-78f6a73c-284a-b255-7db2-3a419e9e9cc7_1280x720.mp4"
    # == "2019-03-16-0021801041-61-73628268-9ffa-73a0-de37-015cd4a7c012_1280x720.npy"
    # ):
    # continue

    if os.path.exists(os.path.join(out_path, mp4.replace(".mp4", ".npy"))):
        return

    print("Processing {}".format(mp4.replace(".mp4", ".npy")))

    """mp4 io to torch tensor """
    vid_f, vid_a, vid_meta = torchvision.io.read_video(
        os.path.join(mp4_path, mp4), pts_unit="sec"
    )

    """do not ignore """
    # if vid_f.shape[0] < 64:
    # print("{} is too short".format(mp4))
    # continue

    clip = F.interpolate(vid_f.permute(0, 3, 1, 2), size=(256, 384))
    clip = clip.unsqueeze(0).permute(0, 2, 1, 3, 4).float()

    input_temporal = 9
    num_frame = clip.shape[2]
    try:
        num_sec = math.ceil(num_frame * 1.0 / int(vid_meta["video_fps"]))
    except:
        print("Error while processing {}".format(mp4))
        return
    # downsample_frames = math.ceil(
    #     num_frame / int(vid_meta["video_fps"]) * input_temporal
    # )

    downsample_frames = 30
    """Down-Sample to 8 frames per second, rather than 60 frames """
    num_idx = np.round(np.linspace(0, num_frame - 1, downsample_frames)).astype(int)

    """Comment out if not do any downsampling """
    # clip = clip[:, :, num_idx]

    with torch.no_grad():
        """Output is [1, fea_dim, T/8, 1, 1]"""
        frame_interval = 32
        clip = preprocess(clip.squeeze().permute(1, 0, 2, 3))
        num_iter = math.ceil(clip.shape[0] / frame_interval)
        rst_list = []
        for iter in range(num_iter):
            min_ind = iter * frame_interval
            max_ind = (iter + 1) * frame_interval

            if iter == num_iter - 1:
                rst_list.append(
                    model.model.forward_features(
                        clip[-frame_interval:]
                        .unsqueeze(0)
                        .permute(0, 2, 1, 3, 4)
                        .cuda()
                    ).cpu()
                )
            else:

                rst_list.append(
                    model.model.forward_features(
                        # model.model.(
                        clip[min_ind:max_ind]
                        .unsqueeze(0)
                        .permute(0, 2, 1, 3, 4)
                        .cuda()
                    ).cpu()
                )
        vid_fea = torch.cat(rst_list, 0)

    with open(os.path.join(out_path, mp4.replace(".mp4", ".npy")), "+wb") as f:
        np.save(f, np.squeeze(vid_fea))
        print("Saved {}".format(mp4.replace(".mp4", ".npy")))
    return


def transform(snippet):
    """ stack & noralization """
    snippet = np.concatenate(snippet, axis=-1)
    snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
    snippet = snippet.mul_(2.0).sub_(255).div(255)

    return snippet.view(1, -1, 3, snippet.size(1), snippet.size(2)).permute(
        0, 2, 1, 3, 4
    )


def process_videos_in_parallel(tasks, num_processes=11):
    with mp.get_context("spawn").Pool(num_processes) as pool:
        pool.starmap(main, tasks)
        
def find_mp4_files(root_dir):
    """Recursively find all .mp4 files in the directory."""
    mp4_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".mp4"):
                mp4_files.append(os.path.join(root, file))
    return mp4_files

# if __name__ == "__main__":
#     vid_folders = os.listdir("/cold2TB/datasets/NSVA/tools/pbp_videos")
#     print(vid_folders)
#     store_dir = os.listdir("/cold2TB/datasets/NSVA/features/unlabeled_TSF")
    
#     file_name = "/cold2TB/datasets/NSVA/processed_videos.json"
#     with open(file_name, 'r') as file:
#         video_dict = json.load(file)

#     for f in vid_folders:
#         try:
#             v = video_dict[f.split('.')[0]]
#         except:
#             print(f"Error processing {f}")
#             continue
#         if f not in store_dir:
#             main(f, "/cold2TB/datasets/NSVA/features/unlabeled_TSF", "/cold2TB/datasets/NSVA/tools/pbp_videos")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    root_video_dir = "/4TBssd/kinetics-dataset/k700-2020/train"
    output_dir = "/4TBssd/NSVA/features/kinetics_train"

    all_mp4_files = find_mp4_files(root_video_dir)
    tasks = []

    for mp4_file in all_mp4_files:
        file_name = os.path.basename(mp4_file)
        try:
            tasks.append((file_name, output_dir, mp4_file.replace(file_name, "")))
        except KeyError:
            print(f"File not in files dict {file_name}")

    tasks = list(set(tasks))
    print(f"Processing {len(tasks)} videos")
    process_videos_in_parallel(tasks)

