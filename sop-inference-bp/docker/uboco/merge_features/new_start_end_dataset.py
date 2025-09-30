import torch
from torch.utils.data import Dataset
import numpy as np
import logging
from os.path import join, exists
from utils.basic_utils import load_jsonl, l2_normalize_np_array
from utils.tensor_utils import pad_sequences_1d
from moviepy.editor import VideoFileClip
import os
import subprocess
import json

logger = logging.getLogger(__name__)


class NewStartEndDataset(Dataset):
    # Q_FEAT_TYPES = ["pooler_output", "last_hidden_state"]
    """One line in data loaded from data_path."
    {
      "qid": 7803,
      "query": "Man in gray top walks from outside to inside.",
      "duration": 150,
      "vid": "RoripwjYFp8_360.0_510.0",
    }
    """

    def __init__(self, data_path, eval_path, v_feat_dirs,
                 max_v_l=150, data_ratio=1.0, ctx_mode="video",
                 normalize_v=True, normalize_t=True, load_labels=True,
                 clip_len=2, max_windows=5, span_loss_type="l1", txt_drop_ratio=0):
        self.data_path = data_path
        self.eval_path = eval_path
        self.data_ratio = data_ratio
        self.v_feat_dirs = v_feat_dirs \
            if isinstance(v_feat_dirs, list) else [v_feat_dirs]
        self.max_v_l = max_v_l
        self.ctx_mode = ctx_mode
        self.use_tef = "tef" in ctx_mode
        self.use_video = "video" in ctx_mode
        self.normalize_v = normalize_v
        self.clip_len = clip_len
        # self.max_windows = max_windows  # maximum number of windows to use as labels
        # self.span_loss_type = span_loss_type
        # self.txt_drop_ratio = txt_drop_ratio
        # if "val" in data_path or "test" in data_path:
            # assert txt_drop_ratio == 0

        # data
        self.data = self.load_data()
        rewrite = False
        for v in self.data:
            path = os.path.join(self.data_path, v["vid"] + ".mp4")
            if "duration" not in v:
                probe = subprocess.run(
                    ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
                    path], stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT)
                v['duration'] = float(probe.stdout)
                rewrite=True
            if "fps" not in v:
                probe = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=avg_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1',
                path], stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT)
                framerate_str = probe.stdout.decode().strip()
                
                # Convert the framerate string to a float
                num, denom = framerate_str.split('/')
                v['fps'] = float(num) / float(denom)
                rewrite=True


        if rewrite:
            with open(self.eval_path, "w") as f:
                for v in self.data:
                    f.write(json.dumps(v) + "\n")

        


    def load_data(self):
        datalist = load_jsonl(self.eval_path)
        if self.data_ratio != 1:
            n_examples = int(len(datalist) * self.data_ratio)
            datalist = datalist[:n_examples]
            logger.info("Using {}% of the data: {} examples"
                        .format(self.data_ratio * 100, n_examples))
        return datalist

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        meta = self.data[index]

        # print(f"meta: {meta}")
        model_inputs = dict()

        # model_inputs["query_feat"] = self._get_query_feat_by_qid(meta["qid"])  # (Dq, ) or (Lq, Dq)
        if self.use_video:
            model_inputs["video_feat"] = self._get_video_feat_by_vid(meta["vid"])  # (Lv, Dv)
            ctx_l = len(model_inputs["video_feat"])
        else:
            ctx_l = self.max_v_l
        # print(ctx_l)
        # print(model_inputs.keys())
        if self.use_tef:
            # print('use_tef true')
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
            if self.use_video:
                model_inputs["video_feat"] = torch.cat(
                    [model_inputs["video_feat"], tef], dim=1)  # (Lv, Dv+2)
            else:
                model_inputs["video_feat"] = tef

        # if self.load_labels:
        #     model_inputs["span_labels"] = self.get_span_labels(meta["relevant_windows"], ctx_l)  # (#windows, 2)
        #     # print(model_inputs["span_labels"])
        #     if "subs_train" not in self.data_path:
        #         model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"] = \
        #             self.get_saliency_labels_w_annot(meta["relevant_clip_ids"], meta["saliency_scores"], ctx_l)
        #     else:
        #         model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"] = \
        #             self.get_saliency_labels_sub_as_query(meta["relevant_windows"][0], ctx_l)  # only one gt

        return dict(meta=meta, model_inputs=model_inputs)

    def _get_video_feat_by_vid(self, vid):
        v_feat_list = []
        for _feat_dir in self.v_feat_dirs:
            _feat_path = join(_feat_dir, f"{vid}.npz")
            try:
                _feat = np.load(_feat_path)["features"][:self.max_v_l].astype(np.float32)
            except Exception as e:
                print("feat path", self.v_feat_dirs, _feat_path, flush=True)
                raise e
            if self.normalize_v:
                _feat = l2_normalize_np_array(_feat)
            v_feat_list.append(_feat)
        # some features are slightly longer than the others
        min_len = min([len(e) for e in v_feat_list])
        v_feat_list = [e[:min_len] for e in v_feat_list]
        v_feat = np.concatenate(v_feat_list, axis=1)
        return torch.from_numpy(v_feat)  # (Lv, D)

def start_end_collate(batch):
    batch_meta = [e["meta"] for e in batch]  # seems no need to collate ?

    model_inputs_keys = batch[0]["model_inputs"].keys()
    batched_data = dict()
    for k in model_inputs_keys:
        if k == "span_labels":
            batched_data[k] = [dict(spans=e["model_inputs"]["span_labels"]) for e in batch]
            continue
        if k in ["saliency_pos_labels", "saliency_neg_labels"]:
            batched_data[k] = torch.LongTensor([e["model_inputs"][k] for e in batch])
            continue
        else:
            batched_data[k] = pad_sequences_1d(
                [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None)
    
    # assert 1==2, (batch[0]["model_inputs"]["video_feat"].size(), batched_data['video_feat'][0].size()) # (torch.Size([24, 3074]), torch.Size([21, 134, 3074]))
    return batch_meta, batched_data