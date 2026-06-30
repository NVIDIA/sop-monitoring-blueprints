# multi-modal-autolabel-augmentation-pipeline
This repo contains multiple generation pipeline for BCQ, GQA and MCQ with prior knowledge. Please be noted that this is developed based on SOP monitoring use case.

The output of this pipeline would be a json file in Llava 1.5 format.


## Implemented Features
1. `gqa_to_gqas`: General QA pairs generation based on given reference QA
2. `golden_gqa_to_gqa`: Transform predefined golden QA into Llava 1.5 format
3. `config_to_bcq`: Binary QA (Yes/No) pairs generation based on given question and choices
4. `config_to_sequential_mcq`: Sequential Multiple Choices QA pairs generation based on given question and choices


## Installation
1. Clone this repo

2. Build dev docker image:
```
docker build --no-cache --progress=plain -t <docker-image-name> . 2>&1 | tee build.log
```

3. Run dev docker container:
```
docker run --gpus all \
           --ipc=host \
           --ulimit memlock=-1 \
           --ulimit stack=67108864 \
           --rm \
           -it \
           -v ./multi-modal-autolabel-augmentation-pipeline:/workspace/multi-modal-autolabel-augmentation-pipeline \
           --name <container-name> <image-name>
```

4. cd to project directory (below command is run inside dev docker container)
```
cd multi-modal-autolabel-augmentation-pipeline
```

## Prerequisite
* Input video root folder structure

The `--video-root` argument for augmentation piplines (gqa_to_gqas, golden_gqa_to_gqa, config_to_bcq, config_to_sequential_mcq) needs to be pointed to the `video_root` folder level.

The video chunk name needs to follow the format of [step]\_[video_name]\_[cnt]_\[order].mp4

For instance "02_action_1_2.mp4" means the 2nd (02) step showing in 2nd order, and "02_action_2_3.mp4" means the 2nd video of the 2nd (02) step showing in 3rd order. (Sometimes one action can be performed for multiple times.)
```
video_root
    |
    |---video-1
    |     |---01_action_1_1.mp4
    |     |---02_action_1_2.mp4
    |     |---03_action_1_3.mp4
    |     |---04_action_1_4.mp4
    |---video-2
    |     |---01_action_1_1.mp4
    |     |---02_action_1_2.mp4
    |     |---02_action_2_3.mp4
    |     |---04_action_1_4.mp4
```


## Usage
### gqa_to_gqas
`Input`: predefined golden QA

`Output`: Json file in llava 1.5 format

`Reference Command`:
```
bash scripts/run_gqa_to_gqas.sh --llm meta/llama-3.1-70b-instruct \
                                --api-key <NVIDIA NIM API KEY> \
                                --sample-qa-root ./data/gqa_to_gqas \
                                --action-json ./actions.json \
                                --num-qa-llm 5 \
                                --video-root ./sample_data/input_video \
                                --video-ext mp4 \
                                --human-suffix "<video>\n" \
                                --num-qa-per-chunk 3 \
                                --output-root ./sample_data/gqa_test \
                                --output-name gqa_test \
                                --replace false \
                                --min_frames 4 \
                                --max_frames 16 \
                                --frames_upperbound 16 \
                                --dynamic_sample true
```

`Arguments Definition`:
* `llm`: Nvidia NIM LLM model (For NV internal user, please use internal endpoint, ex: nvdev/meta/llama-3.1-70b-instruct)
* `api-key`: Nvidia NIM api-key
* `sample-qa-root`: sample golden qa files for llm to generate more qa pairs
* `action-json`: actions json file that is used by labelling tool (only be effective when )
* `num-qa-llm`: number of qa pairs for LLM to generate
* `video-root`: input video root (please noted that the video root needs to follow the folder structure mentioned above)
* `video-ext`: video file extension
* `human-suffix`: suffix for human value in llava 1.5 data format (default is \<video\>\n)
* `num-qa-per-chunk`: number of qa pairs to sample for an action chunk
* `output-root`: root path to store output json file
* `output-name`: name of the output json file
* `replace`: whether to do replacement during random sample
* `min_frames`: metadata of minimum frames for dynamic sampling during SFT
* `max_frames`: metadata of maximum frames for dynamic sampling during SFT
* `frames_upperbound`: metadata of maximum allowed frames for dynamic sampling during SFT
* `dynamic_sample`: metadata of whether to enable dynamic sampling during SFT

### golden_gqa_to_gqa
`Input`: predefined golden QA

`Output`: Golden QA in json file in llava 1.5 format

`Reference Command`:
```
bash scripts/run_golden_gqa_to_gqa.sh --golden-qa-root ./data/gqa_to_gqas \
                                      --action-json ./actions.json \
                                      --video-root ./sample_data/input_video \
                                      --video-ext mp4 \
                                      --human-suffix "<video>\n" \
                                      --output-root ./sample_data/golden_gqa_test \
                                      --output-name golden_gqa_test \
                                      --min_frames 4 \
                                      --max_frames 16 \
                                      --frames_upperbound 16 \
                                      --dynamic_sample true
```

`Arguments Definition`:
* `golden-qa-root`: golden qa files to be transform into llava 1.5 json format
* `action-json`: actions json file that is used by labelling tool (only be effective when )
* `video-root`: input video root (please noted that the video root needs to follow the folder structure mentioned above)
* `video-ext`: video file extension
* `human-suffix`: suffix for human value in llava 1.5 data format (default is \<video\>\n)
* `output-root`: root path to store output json file
* `output-name`: name of the output json file
* `min_frames`: metadata of minimum frames for dynamic sampling during SFT
* `max_frames`: metadata of maximum frames for dynamic sampling during SFT
* `frames_upperbound`: metadata of maximum allowed frames for dynamic sampling during SFT
* `dynamic_sample`: metadata of whether to enable dynamic sampling during SFT

### config_to_bcq
`Input`: predefined question and choices

`Output`: Binary QA pairs in json file in llava 1.5 format

`Reference Command`:
```
bash scripts/run_config_to_bcq.sh --config-root ./data/config_to_bcq \
                                  --action-json ./actions.json \
                                  --subject operator \
                                  --video-root ./sample_data/input_video \
                                  --video-ext mp4 \
                                  --negative-ratio 2.0 \
                                  --output-root ./sample_data/bcq \
                                  --output-name bcq_test \
                                  --min_frames 4 \
                                  --max_frames 16 \
                                  --frames_upperbound 16 \
                                  --dynamic_sample true
```

`Arguments Definition`:
* `config-root`: root path for storing question and choices file
* `action-json`: actions json file that is used by labelling tool (only be effective when )
* `video-root`: input video root (please noted that the video root needs to follow the folder structure mentioned above)
* `video-ext`: video file extension
* `output-root`: root path to store output json file
* `output-name`: name of the output json file
* `min_frames`: metadata of minimum frames for dynamic sampling during SFT
* `max_frames`: metadata of maximum frames for dynamic sampling during SFT
* `frames_upperbound`: metadata of maximum allowed frames for dynamic sampling during SFT
* `dynamic_sample`: metadata of whether to enable dynamic sampling during SFT

### config_to_sequential_mcq
`Input`: predefined question and choices files

`Output`: MCQ pairs in json file in llava 1.5 format

`Reference Command`:
```
bash scripts/run_config_to_mcq.sh --config-root ./data/config_to_sequential_mcq \
                                  --action-json ./actions.json \
                                  --video-root ./sample_data/input_video \
                                  --exclude-action 7_8 \
                                  --video-ext mp4 \
                                  --max-chunk-len 3 \
                                  --output-root ./sample_data/mcq_test \
                                  --output-name mcq_test \
                                  --min_frames 4 \
                                  --max_frames 16 \
                                  --frames_upperbound 16 \
                                  --dynamic_sample true
```

`Arguments Definition`:
* `config-root`: root path for storing question and choices file
* `action-json`: actions json file that is used by labelling tool (only be effective when )
* `video-root`: input video root (please noted that the video root needs to follow the folder structure mentioned above)
* `exclude-action`: exclude specified actions from sequential actions chunks generation, the action should be seperated by underline "_". (Please noted that the excluded actions are still generate single MCQ chunk)
* `video-ext`: video file extension
* `max-chunk-len`: maximum number of actions within a chunk
* `output-root`: root path to store output json file
* `output-name`: name of the output json file
* `min_frames`: metadata of minimum frames for dynamic sampling during SFT
* `max_frames`: metadata of maximum frames for dynamic sampling during SFT
* `frames_upperbound`: metadata of maximum allowed frames for dynamic sampling during SFT
* `dynamic_sample`: metadata of whether to enable dynamic sampling during SFT

## License
This project is dual-licensed under the `CC-BY-4.0 AND Apache-2.0` terms in the top-level [`LICENSE`](../../../../../../LICENSE) file: source code under Apache-2.0, documentation under CC-BY-4.0. Bundled third-party software is listed in [`THIRD_PARTY_NOTICES.md`](../../../../../../THIRD_PARTY_NOTICES.md).
