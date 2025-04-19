# VR SFA

## Feature Loading
1. download TemporalBench dataset
2. Run feature_extraction.py and minimally run with:
``` python feature_extraction.py --root_dir=<PATH_TO_HIGHEST_VIDEO_DIRECTORY> --feature_dir=<PATH_TO_SAVE_FEATURES_TO> --token=<HUGGINGFACE_ACCESS_TOKEN> ```
This will run it with an extraction batch size of 16 and for VideoMAEv2-Large on the temporal bench dataset. You can also set `--overwrite` to re-extract features

This can be used to both extract features (if features don't exist it will save them out) and load features. The features will be of size (t, feature_dim).

*NOTE: DOWNSAMPLING
To trigger downsampling with default features, use `--downsample` in the command. I only recommend potentially changing `--downsample_type` from uniform to mean as an ablation/test case. 

## Benchmarking (Vivian + Macy)
1. download TemporalBench dataset
2. initialize model and write inference code in `eval/sfa_inference.py` (see `eval/llava-onevision.py` for example)
   check repo https://github.com/mu-cai/TemporalBench/tree/main for more reference.
3. run like
    ```bash
    CUDA_VISIBLE_DEVICES=0 python eval/llava-onevision.py --data_json temporalbench_short_qa.json
    CUDA_VISIBLE_DEVICES=1 python eval/llava-onevision.py --data_json temporalbench_long_qa.json
    CUDA_VISIBLE_DEVICES=2 python eval/llava-onevision.py --data_json temporalbench_short_caption.json
    ```
4. Get scores like
    ```bash
    # for QA
    python get_qa_acc.py --data_json temporalbench_short_qa.json
    python get_qa_acc.py --data_json temporalbench_long_qa.json
    # for captioning
    python get_captioning_score.py 
    ```


    https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2/single_modality