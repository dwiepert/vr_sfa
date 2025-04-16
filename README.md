# VR SFA



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