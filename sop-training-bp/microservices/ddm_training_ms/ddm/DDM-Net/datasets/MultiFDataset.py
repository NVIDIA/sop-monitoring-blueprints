import os
import os.path as osp
import random
import sys
import json
import math
import pickle
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm
try:
    from torchcodec.decoders import VideoDecoder
    TORCHCODEC_AVAILABLE = True
except ImportError:
    TORCHCODEC_AVAILABLE = False

try:
    import av
    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False
import torchvision.transforms as T
from torchvision.transforms import v2

try:
    from datasets.augmentation import Scale, ToTensor, Normalize
except:
    from augmentation import Scale, ToTensor, Normalize


def pil_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


multi_frames_transform = transforms.Compose(
    [Scale(size=(224, 224)), ToTensor(), Normalize()]
)


class KineticsGEBDMulFrames(Dataset):
    def __init__(
        self,
        mode="train",
        dataroot="PATH_TO/Kinetics_GEBD_frame",
        frames_per_side=5,
        tmpl="img_{:05d}.png",
        transform=None,
        seed=666,
        args=None,
    ):
        assert mode.lower() in ["train", "val", "valnew", "test"], "Wrong mode for k400"
        self.mode = mode
        self.split_folder = mode + "_" + "split"
        self.train = self.mode.lower() == "train"
        self.dataroot = dataroot
        if self.mode == "train":
            self.dataroot = "../../data/kinetics_GEBD_RGB/train"
        if self.mode == "val":
            self.dataroot = "../../data/kinetics_GEBD_RGB/val"
        if self.mode == "test":
            self.dataroot = "../../data/kinetics_GEBD_RGB/test"
        self.frame_per_side = frames_per_side
        self.tmpl = tmpl
        # assert negtive_step > 0, f'negtive_step = {negtive_step} is illegal!'
        # self.negtive_step = negtive_step
        self.seed = seed
        self.train_file = "multi-frames-GEBD-train-{}.pkl".format(frames_per_side)
        self.val_file = "multi-frames-GEBD-{}-{}.pkl".format(mode, frames_per_side)
        self.load_file = self.train_file if self.mode == "train" else self.val_file
        self.load_file_path = os.path.join("./DataAssets", self.load_file)

        if not (
            os.path.exists(self.load_file_path) and os.path.isfile(self.load_file_path)
        ):
            if (args is not None and args.rank == 0) or args is None:
                print("Preparing pickle file ...")
                self._prepare_pickle(
                    anno_path="../../data/k400_mr345_{}_min_change_duration0.3.pkl".format(
                        mode
                    ),
                    downsample=3,
                    min_change_dur=0.3,
                    keep_rate=1,
                    load_file_path=self.load_file_path, # save name after prepare pickle file
                )
        if transform is not None:
            self.transform = transform
        else:
            self.transform = multi_frames_transform

        self.seqs = pickle.load(open(self.load_file_path, "rb"), encoding="lartin1")
        self.seqs = np.array(self.seqs, dtype=object)

        self.labels_set = list(np.arange(args.num_classes))
        if self.mode == "train":
            self.train_labels = torch.LongTensor([dta["label"] for dta in self.seqs])
            self.label_to_indices = {
                label: np.where(self.train_labels.numpy() == label)[0]
                for label in self.labels_set
            }
            self.ratios = [
                len(self.label_to_indices[0]) / len(self.label_to_indices[1]),
                1,
            ]
        elif self.mode == "val":
            self.val_labels = torch.LongTensor([dta["label"] for dta in self.seqs])
            self.label_to_indices = {
                label: np.where(self.val_labels.numpy() == label)[0]
                for label in self.labels_set
            }
        elif self.mode == "test":
            self.test_labels = torch.LongTensor([dta["label"] for dta in self.seqs])
            self.label_to_indices = {
                label: np.where(self.test_labels.numpy() == label)[0]
                for label in self.labels_set
            }

        self.img = None

    def _get_training_samples(self, index):
        indices = []
        for class_ in self.labels_set:
            real_index = self.label_to_indices[class_][int(index * self.ratios[class_])]
            indices.append(real_index)
        return indices

    def _read_data(self, index):
        item = self.seqs[index]
        block_idx = item["block_idx"]
        folder = item["folder"]
        current_idx = item["current_idx"]

        img = self.transform(
            [pil_loader(os.path.join(folder, self.tmpl.format(i))) for i in block_idx]
        )
        img = torch.stack(img, dim=0)

        return img, item["label"], os.path.join(folder, self.tmpl.format(current_idx))

    def __getitem__(self, index):
        if self.train:
            indices = self._get_training_samples(index)
            img_list = []
            label_list = []
            path_list = []
            for real_index in indices:
                img, label, img_path = self._read_data(real_index)
                img_list.append(img)
                label_list.append(label)
                path_list.append(img_path)
        else:
            img, label, img_path = self._read_data(index)
            img_list = [
                img,
            ]
            label_list = [
                label,
            ]
            path_list = [
                img_path,
            ]

        return {
            "inp": torch.stack(img_list, dim=0),
            "label": torch.LongTensor(label_list),
            "path": path_list,
        }

    def shuffle(self):
        np.random.seed(self.seed)
        for class_ in self.labels_set:
            np.random.shuffle(self.label_to_indices[class_])

    def __len__(self):
        if self.mode == "train":
            return len(self.label_to_indices[1])
        # from functools import reduce
        # return reduce(sum, [len(v) for v in self.label_to_indices.values()])
        return sum([len(v) for v in self.label_to_indices.values()])

    def _prepare_pickle(
        self,
        anno_path="/PATH_TO/k400_mr345_train_min_change_duration0.3.pkl",
        downsample=3,
        min_change_dur=0.3,
        keep_rate=0.8,
        load_file_path="./data/multi-frames-train.pkl",
    ):
        # prepare file for multi-frames-GEBD
        # dict_train_ann
        with open(anno_path, "rb") as f:
            dict_train_ann = pickle.load(f, encoding="lartin1")

        # downsample factor: sample one every `ds` frames
        ds = downsample

        SEQ = []
        neg = 0
        pos = 0

        for vname in dict_train_ann.keys():
            if not osp.exists(osp.join(self.dataroot, vname)):
                continue

            vdict = dict_train_ann[vname]
            vlen = vdict["num_frames"]
            vlen = min(vlen, len(os.listdir(osp.join(self.dataroot, vname))))
            fps = vdict["fps"]
            f1_consis = vdict["f1_consis"]
            path_frame = vdict["path_frame"]

            cls, frame_folder = path_frame.split("/")[:2]

            # select the annotation with highest f1 score
            highest = np.argmax(f1_consis)
            change_idices = vdict["substages_myframeidx"][highest]

            # (float)num of frames with min_change_dur/2
            half_dur_2_nframes = min_change_dur * fps / 2.0
            # (int)num of frames with min_change_dur/2
            ceil_half_dur_2_nframes = int(np.ceil(half_dur_2_nframes))

            start_offset = np.random.choice(ds) + 1
            selected_indices = np.arange(start_offset, vlen, ds)

            # idx chosen after from downsampling falls in the time range [change-dur/2, change+dur/2]
            # should be tagged as positive(bdy), otherwise negative(bkg)
            GT = []
            for i in selected_indices:
                GT.append(0)
                for change in change_idices:
                    if (
                        i >= change - half_dur_2_nframes
                        and i <= change + half_dur_2_nframes
                    ):
                        GT.pop()  # pop '0'
                        GT.append(1)
                        break
            # assert(len(selected_indices)==len(GT),'length frame indices is not equal to length GT.')
            assert len(selected_indices) == len(
                GT
            ), "length frame indices is not equal to length GT."

            for idx, (current_idx, lbl) in enumerate(zip(selected_indices, GT)):
                # for multi-frames input
                if self.train and random.random() > keep_rate:
                    continue

                record = dict()
                shift = np.arange(-self.frame_per_side, self.frame_per_side)
                shift[shift >= 0] += 1
                shift = shift * ds
                block_idx = shift + current_idx
                block_idx[block_idx < 1] = 1
                block_idx[block_idx > vlen] = vlen
                block_idx = block_idx.tolist()

                record["folder"] = f"{cls}/{frame_folder}"
                record["current_idx"] = current_idx
                record["block_idx"] = block_idx
                record["label"] = lbl
                SEQ.append(record)

                if lbl == 0:
                    neg += 1
                else:
                    pos += 1
        print(f" #bdy-{pos}\n #bkg-{neg}\n #total-{pos+neg}.")
        folder = "/".join(load_file_path.split("/")[:-1])
        if not os.path.exists(folder):
            os.makedirs(folder)
        pickle.dump(SEQ, open(load_file_path, "wb"))
        print(len(SEQ))


class TaposGEBDMulFrames(Dataset):
    def __init__(
        self,
        mode="train",
        dataroot="PATH_TO/TAPOS_GEBD_frame",
        frames_per_side=5,
        tmpl="img_{:05d}.png",
        transform=None,
        seed=666,
        args=None,
    ):
        assert mode.lower() in ["train", "val"], "Wrong mode for TAPOS"
        self.mode = mode
        self.split_folder = mode + "_" + "split"
        self.train = self.mode.lower() == "train"
        self.dataroot = dataroot
        self.frame_per_side = frames_per_side
        self.tmpl = tmpl
        # assert negtive_step > 0, f'negtive_step = {negtive_step} is illegal!'
        # self.negtive_step = negtive_step
        self.seed = seed
        self.train_file = "multi-frames-TAPOS-GEBD-train-{}.pkl".format(frames_per_side)
        self.val_file = "multi-frames-TAPOS-GEBD-{}-{}.pkl".format(mode, frames_per_side)
        self.load_file = self.train_file if self.mode == "train" else self.val_file
        self.load_file_path = os.path.join("./DataAssets", self.load_file)

        if not (
            os.path.exists(self.load_file_path) and os.path.isfile(self.load_file_path)
        ):
            if (args is not None and args.rank == 0) or args is None:
                print("Preparing pickle file ...")
                self._prepare_pickle(
                    anno_path="PATH_TO/TAPOS_{}_anno.pkl".format(
                        mode
                    ),
                    downsample=3,
                    min_change_dur=0.3,
                    keep_rate=1,
                    load_file_path=self.load_file_path,
                )
        if transform is not None:
            self.transform = transform
        else:
            self.transform = multi_frames_transform

        self.seqs = pickle.load(open(self.load_file_path, "rb"), encoding="lartin1")
        self.seqs = np.array(self.seqs, dtype=object)

        self.labels_set = list(np.arange(args.num_classes))
        if self.mode == "train":
            self.train_labels = torch.LongTensor([dta["label"] for dta in self.seqs])
            self.label_to_indices = {
                label: np.where(self.train_labels.numpy() == label)[0]
                for label in self.labels_set
            }
            self.ratios = [
                len(self.label_to_indices[0]) / len(self.label_to_indices[1]),
                1,
            ]
        elif self.mode == "val":
            self.val_labels = torch.LongTensor([dta["label"] for dta in self.seqs])
            self.label_to_indices = {
                label: np.where(self.val_labels.numpy() == label)[0]
                for label in self.labels_set
            }
        elif self.mode == "test":
            self.test_labels = torch.LongTensor([dta["label"] for dta in self.seqs])
            self.label_to_indices = {
                label: np.where(self.test_labels.numpy() == label)[0]
                for label in self.labels_set
            }

        self.img = None

    def _get_training_samples(self, index):
        indices = []
        for class_ in self.labels_set:
            real_index = self.label_to_indices[class_][int(index * self.ratios[class_])]
            indices.append(real_index)
        return indices

    def _read_data(self, index):
        item = self.seqs[index]
        block_idx = item['block_idx']
        folder = item['folder']
        current_idx = item['current_idx']
        # '''

        img = self.transform([pil_loader(
            os.path.join(folder, self.tmpl.format(i))
        ) for i in block_idx])
        img = torch.stack(img, dim=0)
        # '''
        # print('img = ', img.shape)
        # if self.img is None:
        #     img = self.transform([pil_loader(
        #         os.path.join(folder, self.tmpl.format(i))
        #     ) for i in block_idx])
        #     img = torch.stack(img, dim=0)
        #     self.img = img
        # else:
        #     img = self.img
        return img, item['label'], os.path.join(folder, self.tmpl.format(current_idx))

    def __getitem__(self, index):
        if self.train:
            indices = self._get_training_samples(index)
            img_list = []
            label_list = []
            path_list = []
            for real_index in indices:
                img, label, img_path = self._read_data(real_index)
                img_list.append(img)
                label_list.append(label)
                path_list.append(img_path)
        else:
            img, label, img_path = self._read_data(index)
            img_list = [img, ]
            label_list = [label, ]
            path_list = [img_path, ]
        # print('img2 = ', torch.stack(img_list, dim=0).shape)
        return {
            'inp': torch.stack(img_list, dim=0),
            'label': torch.LongTensor(label_list),
            'path': path_list
        }

    def shuffle(self):
        np.random.seed(self.seed)
        for class_ in self.labels_set:
            np.random.shuffle(self.label_to_indices[class_])

    def __len__(self):
        if self.mode == "train":
            return len(self.label_to_indices[1])
        # from functools import reduce
        # return reduce(sum, [len(v) for v in self.label_to_indices.values()])
        return sum([len(v) for v in self.label_to_indices.values()])

    def _prepare_pickle(
        self,
        anno_path="/PATH_TO/TAPOS/save_output/TAPOS_train_anno.pkl",
        downsample=3,
        min_change_dur=0.3,
        keep_rate=0.8,
        load_file_path="./data/multi-frames-train.pkl",
    ):
        # prepare file for multi-frames-GEBD
        # dict_train_ann
        with open(anno_path, "rb") as f:
            dict_train_ann = pickle.load(f, encoding="lartin1")

        # Some fields in anno for reference
        # {'raw': {'action': 11, 'substages': [0, 79, 195], 'total_frames': 195, 'shot_timestamps': [43.36, 53.48], 'subset': 'train'},
        # 'path': 'yMK2zxDDs2A/s00004_0_100_7_931',
        # 'myfps': 25.0,
        # 'my_num_frames': 197,
        # 'my_duration': 7.88,
        # 'my_substages_frameidx': [79]
        # }

        # downsample factor: sample one every `ds` frames
        ds = downsample

        SEQ = []
        neg = 0
        pos = 0

        for vname in dict_train_ann.keys():
            if not osp.exists(osp.join(self.dataroot, vname)):
                continue

            vdict = dict_train_ann[vname]
            vlen = vdict["my_num_frames"]
            vlen = min(vlen, len(os.listdir(osp.join(self.dataroot, vname))))
            fps = vdict["myfps"]
            path_frame = vdict["path"]

            change_idices = vdict["my_substages_frameidx"]

            # (float)num of frames with min_change_dur/2
            half_dur_2_nframes = min_change_dur * fps / 2.0
            # (int)num of frames with min_change_dur/2
            ceil_half_dur_2_nframes = int(np.ceil(half_dur_2_nframes))

            start_offset = np.random.choice(ds) + 1
            selected_indices = np.arange(start_offset, vlen, ds)

            # idx chosen after from downsampling falls in the time range [change-dur/2, change+dur/2]
            # should be tagged as positive(bdy), otherwise negative(bkg)
            GT = []
            for i in selected_indices:
                GT.append(0)
                for change in change_idices:
                    if (
                        i >= change - half_dur_2_nframes
                        and i <= change + half_dur_2_nframes
                    ):
                        GT.pop()  # pop '0'
                        GT.append(1)
                        break
            # assert(len(selected_indices)==len(GT),'length frame indices is not equal to length GT.')
            assert len(selected_indices) == len(
                GT
            ), "length frame indices is not equal to length GT."

            for idx, (current_idx, lbl) in enumerate(zip(selected_indices, GT)):
                # for multi-frames input
                if self.train and random.random() > keep_rate:
                    continue

                record = dict()
                shift = np.arange(-self.frame_per_side, self.frame_per_side)
                shift[shift >= 0] += 1
                shift = shift * ds
                block_idx = shift + current_idx
                block_idx[block_idx < 1] = 1
                block_idx[block_idx > vlen] = vlen
                block_idx = block_idx.tolist()

                record["folder"] = path_frame
                record["current_idx"] = current_idx
                record["block_idx"] = block_idx
                record["label"] = lbl
                SEQ.append(record)

                if lbl == 0:
                    neg += 1
                else:
                    pos += 1
        print(f" #bdy-{pos}\n #bkg-{neg}\n #total-{pos+neg}.")
        folder = "/".join(load_file_path.split("/")[:-1])
        if not os.path.exists(folder):
            os.makedirs(folder)
        pickle.dump(SEQ, open(load_file_path, "wb"))
        print(len(SEQ))


###################################################################################
# Dummy dataset for debugging
###################################################################################
class MultiFDummyDataSet(Dataset):
    def __init__(self, mode="train", transform=None, args=None):
        assert mode.lower() in ["train", "val", "test", "valnew"], "Wrong split"
        self.mode = mode
        self.train = self.mode.lower() == "train"
        self.args = args

        if transform is not None:
            self.transform = transform

        self.train_labels = torch.LongTensor(np.random.choice([0, 1], 1000000))
        self.val_labels = torch.LongTensor(np.random.choice([0, 1], 1000000))
        self.load_file = self.train_labels if self.mode == "train" else self.val_labels
        self.load_file = self.train_labels if self.mode == "train" else self.val_labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, label) where target is class_index of the target class.
        """
        label = self.load_file[index]
        inp = torch.randn(10, 3, 224, 224)

        return {"inp": inp, "label": label}

    def __len__(self):
        return len(self.load_file)


class PyAVVideoDecoder:
    """Wrapper class for PyAV to match torchcodec's VideoDecoder interface"""
    def __init__(self, video_path):
        self.video_path = video_path
        self.container = av.open(video_path)
        self.stream = self.container.streams.video[0]

        # Cache metadata
        self.metadata = type('obj', (object,), {
            'average_fps': float(self.stream.average_rate),
            'duration_seconds': float(self.stream.duration * self.stream.time_base) if self.stream.duration else self.container.duration / av.time_base
        })

        # Cache frames for efficient random access
        self._frames_cache = None
        self._num_frames = None

    def __len__(self):
        if self._num_frames is None:
            # Count frames - this might be slow for large videos
            self._num_frames = self.stream.frames
            if self._num_frames == 0:
                # If frame count is not available in metadata, count manually
                self._num_frames = sum(1 for _ in self.container.decode(video=0))
                self.container.seek(0)  # Reset to beginning
        return self._num_frames

    def get_frames_at(self, indices):
        """Get frames at specific indices

        Args:
            indices: numpy array or list of frame indices

        Returns:
            Object with .data attribute containing frames as torch tensor (N, 3, H, W) in uint8
        """
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()

        frames = []

        # Reset container to beginning
        self.container.seek(0)

        # Create a set for faster lookup
        indices_set = set(indices)
        frame_idx = 0

        # Decode frames
        for frame in self.container.decode(video=0):
            if frame_idx in indices_set:
                # Convert frame to numpy array and then to torch tensor
                img = frame.to_ndarray(format='rgb24')  # (H, W, 3)
                img = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
                frames.append(img)

                # Remove from set to track which indices we've found
                indices_set.remove(frame_idx)

                # Break early if we've found all frames
                if not indices_set:
                    break

            frame_idx += 1

        # Handle case where some indices were out of bounds
        # Fill missing frames with the last frame (or zeros)
        if indices_set and frames:
            last_frame = frames[-1]
            for _ in indices_set:
                frames.append(last_frame.clone())

        # Stack frames and ensure correct order
        frames_dict = {}
        frame_idx = 0
        for i, idx in enumerate(indices):
            if idx in frames_dict:
                continue
            if frame_idx < len(frames):
                frames_dict[idx] = frames[frame_idx]
                frame_idx += 1

        # Reorder frames according to original indices
        ordered_frames = [frames_dict.get(idx, frames[-1] if frames else torch.zeros(3, 224, 224, dtype=torch.uint8)) for idx in indices]

        # Stack into single tensor (N, 3, H, W)
        frames_tensor = torch.stack(ordered_frames, dim=0)

        # Return object with .data attribute to match torchcodec interface
        result = type('obj', (object,), {'data': frames_tensor})
        return result

    def close(self):
        """Close the video container"""
        if hasattr(self, 'container'):
            self.container.close()

    def __del__(self):
        """Ensure container is closed when object is destroyed"""
        self.close()


class SOPMulFrames(Dataset):
    def __init__(
        self,
        mode="train",
        anno_path="PATH_TO/SOP_train_anno.json",
        dataroot="PATH_TO/Kinetics_GEBD_frame",
        frames_per_side=5,
        transform=None,
        seed=666,
        args=None,
        video_backend="pyav",  # "torchcodec" or "pyav"
    ):
        assert mode.lower() in ["train", "val", "test"], "Wrong mode for SOP"
        assert video_backend in ["torchcodec", "pyav"], f"video_backend must be 'torchcodec' or 'pyav', got {video_backend}"

        # Check if the selected backend is available
        if video_backend == "torchcodec" and not TORCHCODEC_AVAILABLE:
            raise ImportError("torchcodec is not available. Please install it or use 'pyav' backend.")
        elif video_backend == "pyav" and not PYAV_AVAILABLE:
            raise ImportError("pyav is not available. Please install it or use 'torchcodec' backend.")

        self.mode = mode
        self.train = self.mode.lower() == "train"
        self.dataroot = dataroot
        self.frame_per_side = frames_per_side
        self.seed = seed
        self.video_backend = video_backend
        self.downsample = getattr(args, "downsample", 1) if args is not None else 1
        self.min_change_dur = getattr(args, "min_change_dur", 0.3) if args is not None else 0.3

        self.seqs = [] # list of dicts: video_id, label, current_idx, block_idx
        self.video_paths = {} # dict of video_id: video_features
        self.video_info = {} # dict of video_id: fps
        self.process_data(anno_path) # process sequence and video features

        if transform is not None:
            self.transform = transform
        else:
            self.transform = v2.Compose([
                v2.Resize((224, 224)),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.labels_set = list(np.arange(args.num_classes))
        if self.mode == "train":
            self.train_labels = torch.LongTensor([dta["label"] for dta in self.seqs])
            self.label_to_indices = {
                label: np.where(self.train_labels.numpy() == label)[0]
                for label in self.labels_set
            }
            self.ratios = [
                len(self.label_to_indices[0]) / len(self.label_to_indices[1]),
                1,
            ]
        elif self.mode == "val":
            self.val_labels = torch.LongTensor([dta["label"] for dta in self.seqs])
            self.label_to_indices = {
                label: np.where(self.val_labels.numpy() == label)[0]
                for label in self.labels_set
            }
        elif self.mode == "test":
            self.test_labels = torch.LongTensor([dta["label"] for dta in self.seqs])
            self.label_to_indices = {
                label: np.where(self.test_labels.numpy() == label)[0]
                for label in self.labels_set
            }


    def process_data(self, anno_path):

        for k, content in (json.load(open(anno_path, "r"))).items():
            video_path = os.path.join(self.dataroot, k + ".mp4")
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file {video_path} not found")

            self.video_paths[k] = video_path # Store the path

            # You can still open the video temporarily to get metadata
            try:
                if self.video_backend == "torchcodec":
                    vd = VideoDecoder(video_path)
                else: # pyav
                    vd = PyAVVideoDecoder(video_path)

                fps = vd.metadata.average_fps
                duration = vd.metadata.duration_seconds
                vlen = len(vd)
                del vd # IMPORTANT: Close the file handle immediately
            except Exception as e:
                print(f"Could not read metadata for {video_path}: {e}")
                continue

            self.video_info[k] = {"fps": fps, "duration": duration}
            boundary_list = []
            # content.pop() # Exclude Final Segment
            content = [ct for ct in content if ct["description"] != "Final Segment"]
            for s_sample, e_sample in zip(content[:-1], content[1:]):
                s_time = s_sample["end_timestamp"]
                e_time = e_sample["start_timestamp"]
                boundary_list.append(math.floor((s_time + e_time) / 2 * fps))

            labels = np.zeros(vlen)
            half_dur_2_nframes = self.min_change_dur * fps / 2
            for boundary in boundary_list:
                start_idx = math.ceil(max(0, boundary - half_dur_2_nframes))
                end_idx = math.floor(min(vlen, round(boundary + half_dur_2_nframes) + 1))
                labels[start_idx: end_idx] = 1

            for selected_idx in range(0, vlen, self.downsample):
                if selected_idx == 0 or selected_idx == vlen - 1:
                    continue
                block_idx = selected_idx + np.arange(-self.downsample * self.frame_per_side, self.downsample * (self.frame_per_side + 1), self.downsample)
                block_idx = np.clip(block_idx, 0, vlen - 1)
                sample = {
                    "video_id": k,
                    "label": labels[selected_idx],
                    "current_idx": block_idx[len(block_idx) // 2],
                    "block_idx": block_idx
                }
                self.seqs.append(sample)
        self.seqs = np.array(self.seqs, dtype=object)


    def _get_training_samples(self, index):
        indices = []
        for class_ in self.labels_set:
            real_index = self.label_to_indices[class_][int(index * self.ratios[class_])]
            indices.append(real_index)
        return indices

    def _read_data(self, index):
        item = self.seqs[index]
        video_id = item["video_id"]
        block_idx = item["block_idx"]
        current_idx = block_idx[len(block_idx) // 2]

        video_path = self.video_paths[video_id]

        try:
            # "Lazy load" the video here
            if self.video_backend == "torchcodec":
                decoder = VideoDecoder(video_path)
            else: # pyav
                decoder = PyAVVideoDecoder(video_path)

            # Use the decoder to get frames
            img = self.transform(decoder.get_frames_at(block_idx).data / 255.0)
            del decoder # Clean up the decoder and its file handle

        except Exception as e:
            print(f"Error reading video {video_id} at index {index}: {e}")
            # Return a dummy tensor or raise the error
            # For simplicity, we can raise it
            raise e

        return img, item["label"], video_id, current_idx

    def __getitem__(self, index):

        if self.train:
            indices = self._get_training_samples(index)
            img_list = []
            label_list = []
            path_list = []
            current_idx_list = []
            for real_index in indices:
                img, label, img_path, current_idx = self._read_data(real_index)
                img_list.append(img)
                label_list.append(label)
                path_list.append(img_path)
                current_idx_list.append(current_idx)
        else:
            img, label, img_path, current_idx = self._read_data(index)
            img_list = [img,]
            label_list = [label,]
            path_list = [img_path,]
            current_idx_list = [current_idx,]

        return {
            "inp": torch.stack(img_list, dim=0),
            "label": torch.LongTensor(label_list),
            "path": path_list,
            "current_ids": current_idx_list,
        }

    def shuffle(self):
        np.random.seed(self.seed)
        for class_ in self.labels_set:
            np.random.shuffle(self.label_to_indices[class_])

    def __len__(self):
        if self.mode == "train":
            return len(self.label_to_indices[1])
        # from functools import reduce
        # return reduce(sum, [len(v) for v in self.label_to_indices.values()])
        return sum([len(v) for v in self.label_to_indices.values()])

    def collate_fn(self, batch):
        raise NotImplementedError("Not implemented")


if __name__ == "__main__":
    # KineticsGEBDMulFrames
    dataset = KineticsGEBDMulFrames(mode="val")
    print(dataset[24511])
