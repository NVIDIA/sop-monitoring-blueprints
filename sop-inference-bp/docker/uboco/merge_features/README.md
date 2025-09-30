# Overview
This code generates event boundaries for a video using slowfast and clip features extracted with the `HERO_Video_Feature_Extractor` folder
Based off of `https://github.com/jinhyunj/EaTR/blob/main/models/model.py`
## Usage for getting event boundaries from video features

First prepare a list of video files whose features you've extracted, and then convert them to a json of the same form as `benchmark_test.jsonl` (only "vid" is necessary)

Next, run 
`python pseudo_event_boundaries_inference.py --v_feat_dirs {list of folders where .npz feature files are stored for all the videos} --result_path {text file storing results} --eval_path {a file where each line is a dictionary with a key "vid" with the video id} --data_path {folder where the videos are stored, they must all be .mp4}`

For example:
`python pseudo_event_boundaries_inference.py --v_feat_dirs /home/alhu/nv-label-studio-video/HERO_Video_Feature_Extractor/clip/output/clip-intern_cliplen1_features /home/alhu/nv-label-studio-video/HERO_Video_Feature_Extractor/slowfast/output/slowfast_features_cliplen1 --result_path results/benchmark_test_intern_complete.txt --eval_path benchmark_test.jsonl --data_path /home/alhu/nv-label-studio-video/data/benchmark`

## Usage for getting event boundaries
The `ensemble_boundaries.py` script assumes you have 10 features in `results/benchmark_test_intern_cliplen1_start0{i}` for `i` in `range(10)`. Each folder represents the features extracted from the videos starting at time `0.{i}`.

manually edit the code to either do:
- Algorithm 1: Take the boundaries from extracting features starting at 0, 0.1, 0.2, and combine the boundaries. For each boundary from left to right, see if there's at least one other boundary with 0.3 of it, and if so, add this boundary to the final list of boundaries. 
- Algorithm 2: Combine boundaries from 0, 0.1, 0.2, ... 0.9 offsets, and run K-means with number of clusters set to the mean number of boundaries across offsets. Round the first and last boundary to 0 and the video duration.

Algorithm 1 performs better. Its results can be found in `../dense_captioning_eval/merge_times_intern_merge_panda.json`
