import os
import numpy as np
import cv2
import torch
import torchvision
import torch.nn.functional as F
import math
import gc
import torch as th

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
            # From official repo:
            # These are the mean and std values used for data normalization: 
            # _C.DATA.MEAN = [0.45, 0.45, 0.45], _C.DATA.STD = [0.225, 0.225, 0.225].
            self.norm = Normalize(mean=[114.75, 114.75, 114.75], std=[57.375, 57.375, 57.375])

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
    num_frames=8,
    attention_type="divided_space_time",
    pretrained_model="/4TBSSD_permanent/NSVA/tools/Timesformer/models/TimeSformer_divST_32x32_224_HowTo100M.pyth",
).cpu()


model = model.cpu()
torch.backends.cudnn.benchmark = False
model.eval()


def main(file_path, w_size=8, downsize_only=True):
    global model

    """Read mp4 files"""
    mp4_path = "/4TBSSD_permanent/NSVA/raw_data"
    # mp4_path = "/home/karolwojtulewicz/code/pytorch-i3d-feature-extraction/raw_thumos/validation"
    # out_path = os.path.join("/home/karolwojtulewicz/code/TimeSformer/dataset/UCF101_features", file_path)
    out_path = "/4TBSSD_permanent/NSVA/downscaled_frames"
    # out_path = "/home/karolwojtulewicz/code/TimeSformer/dataset/thumos_val_feat"

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    
    actionFormer_path = "/4TBSSD_permanent/NSVA/features_NSVA"

    all_mp4 = os.listdir(mp4_path)
    actionFormer_all = os.listdir(actionFormer_path)

    for mp4 in all_mp4:

        """Some videos are too short or too-long (taking too much memory)"""
        
        # if(mp4.split(".mp4")[0] == "video_test_0000793" 
        #    or mp4.split(".mp4")[0] == "video_validation_0000369"
        #    or mp4.split(".mp4")[0] == "video_validation_0000314"
        #    or mp4.split(".mp4")[0] == "video_test_0001207"
        #    or mp4.split(".mp4")[0] == "video_validation_0000666"
        #    or mp4.split(".mp4")[0] == "video_test_0000950"
        #    ):
        #     print("Skipping, Memory issues {}".format(mp4.replace(".mp4", ".npy")))
        #     continue

        if os.path.exists(os.path.join(out_path, mp4.replace(".mp4", ".npy"))):
            print("Skipping, already computed {}".format(mp4.replace(".mp4", ".npy")))
            continue
        # if(mp4.replace(".mp4", ".npy") not in actionFormer_all):
        #     print("Skipping, not in the DB {}".format(mp4.replace(".mp4", ".npy")))
        #     continue

        print("Processing {}".format(mp4.replace(".mp4", ".npy")))

        """mp4 io to torch tensor """
        vid_f, vid_a, vid_meta = torchvision.io.read_video(
            os.path.join(mp4_path, mp4), pts_unit="sec"
        )
        
        del vid_a
        gc.collect()

        """do not ignore """
        # if vid_f.shape[0] < 64:
        # print("{} is too short".format(mp4))
        # continue

        clip = F.interpolate(vid_f.permute(0, 3, 1, 2), size=(224, 224)).float()
        # clip = clip.permute(1, 0, 2, 3).float()

        # input_temporal = 4
        # num_frame = clip.shape[2]
        # num_sec = math.ceil(num_frame * 1.0 / int(vid_meta["video_fps"]))
        # downsample_frames = math.ceil(
        #     num_frame / int(vid_meta["video_fps"]) * input_temporal
        # )

        # downsample_frames = 2
        # downsample_frames = math.ceil(downsample_frames*num_frame/vid_meta["video_fps"])
        # downsample_frames = math.ceil(num_frame/downsample_frames)
        """Down-Sample to 8 frames per second, rather than 60 frames """
        # num_idx = np.round(np.linspace(0, num_frame - 1, downsample_frames)).astype(int)

        """Comment out if not do any downsampling """
        # clip = clip[:, :, num_idx]

        with torch.no_grad():
            """Output is [1, fea_dim, T/8, 1, 1]"""
            window_size = w_size
            # clip = preprocess(clip.squeeze().permute(1, 0, 2, 3))
            clip = preprocess(clip)
            num_iter = math.ceil(clip.shape[0] / window_size)
            rst_list = []
            for iter in range(num_iter):
                min_ind = iter * window_size
                max_ind = (iter + 1) * window_size

                if iter == num_iter - 1:
                    rst_list.append(
                        model.model.forward_features(
                            clip[-window_size:]
                            .unsqueeze(0)
                            .permute(0, 2, 1, 3, 4)
                            .cuda()
                            
                        ).detach().cpu()
                        if not downsize_only
                        else clip[-window_size:]
                            .unsqueeze(0)
                            .permute(0, 2, 1, 3, 4)
                    )
                else:

                    rst_list.append(
                        model.model.forward_features(
                            # model.model.(
                            clip[min_ind:max_ind]
                            .unsqueeze(0)
                            .permute(0, 2, 1, 3, 4)
                            .cuda()
                        ).detach().cpu()
                        if not downsize_only
                        else clip[-window_size:]
                            .unsqueeze(0)
                            .permute(0, 2, 1, 3, 4)
                    )
            try:
                vid_fea = torch.cat(rst_list[2:-2], 0)
            except:
                print("Skipping, Memory issues {}".format(mp4.replace(".mp4", ".npy")))
                continue

        with open(os.path.join(out_path, mp4.replace(".mp4", ".npy")), "+wb") as f:
            np.save(f, np.squeeze(vid_fea))


def transform(snippet):
    """ stack & noralization """
    snippet = np.concatenate(snippet, axis=-1)
    snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
    snippet = snippet.mul_(2.0).sub_(255).div(255)

    return snippet.view(1, -1, 3, snippet.size(1), snippet.size(2)).permute(
        0, 2, 1, 3, 4
    )


if __name__ == "__main__":
    # vid_folders = os.listdir("/home/karolwojtulewicz/code/pytorch-i3d-feature-extraction/raw_thumos/validation") 
    vid_folders = os.listdir("/4TBSSD_permanent/NSVA/raw_data") 
    window_size = 32
    print(vid_folders)
    # store_dir = os.listdir("/home/karolwojtulewicz/code/TimeSformer/dataset/thumos_val_feat")
    out_path = "/4TBSSD_permanent/NSVA/features_NSVA/w_size_{}".format(window_size)
    # out_path = "/home/karolwojtulewicz/code/TimeSformer/dataset/thumos_val_feat"

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    store_dir = os.listdir(out_path)

    for f in vid_folders:
        if f not in store_dir:
            main(f, w_size=window_size,downsize_only=True)

