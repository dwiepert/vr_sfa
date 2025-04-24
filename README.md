# VR SFA

## Feature Loading
1. download pre-computed features from [VideoMAEv2/TAD.md](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/TAD.md)
2. Load in features with [VideoDataset](https://github.com/dwiepert/vr_sfa/features/feature_extraction.py).
    * can Downsample with `downsample`=True and `downsample_method` = ['uniform','mean']
    * if you need outputs as tensors, use `to_tensor`=True

## Training SFA
Install `emaae` package:
To install, use

```
$ git clone https://github.com/dwiepert/emaae.git
$ cd emaae
$ pip install . 
```

Minimally run training with:
```
python sfa_model.py --feat_dir=<PATH_TO_FEATURE_DIR> --out_dir=<PATH_TO_SAVE> --input_dim=1408 --inner_size=<1408_OR_2048> \
 --n_encoder=<2_OR_3> --n_decoder=<2_OR_3> <--exclude_all_norm OR --batchnorm_first> \
 --batch_sz=<INT> --epochs=<INT> --lr=<FLOAT> --alpha=0.001 --early_stop --encoding_loss=tvl2 --skip_eval
```

Run evaluation for a different feature set with:
```
python sfa_model.py --feat_dir=<PATH_TO_FEATURE_DIR> \
 --out_dir=<PATH_TO_SAVE> --model_config=<PATH_TO_model_config.json> \
 --checkpoint=<PATH_TO_.pth_FILE> --encode --eval_only 
```



## Benchmarking (Vivian + Macy)
1. clone ActionFormer repo and download data
```
git clone https://github.com/happyharrycn/actionformer_release.git
wget https://huggingface.co/datasets/OpenGVLab/VideoMAEv2-TAL-Features/resolve/main/th14_mae_g_16_4.tar.gz
gdown --id 1zt2eoldshf99vJMDuu8jqxda55dCyhZP
tar -xf th14_mae_g_16_4.tar.gz
tar -xf thumos.tar.gz
```
you can also download the THUMOS features from [here](https://drive.google.com/file/d/1zt2eoldshf99vJMDuu8jqxda55dCyhZP/view?usp=sharing)


2. add and modify some files
```
mv thumos_mae.yaml actionformer_release/configs
rm actionformer_release/libs/utils/metrics.py
mv metrics.py actionformer_release/libs/utils
rm actionformer_release/eval.py
mv eval.py actionformer_release/
```

3. ActionFormer setup
```
cd actionformer_release/libs/utils
python setup.py install --user
```

4. run train/eval
```
cd actionformer_release
python train.py configs/thumos_mae.yaml --output maefeatures
python eval.py ./configs/thumos_mae.yaml ./ckpt/thumos_mae_maefeatures
```

reference: 
- https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/TAD.md
- https://github.com/happyharrycn/actionformer_release/tree/main?tab=readme-ov-file#to-reproduce-our-results-on-thumos14

    https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2/single_modality