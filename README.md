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
1. clone ActionFormer repo and download THUMOS data
```
git clone https://github.com/happyharrycn/actionformer_release.git
gdown --id 1zt2eoldshf99vJMDuu8jqxda55dCyhZP
tar -xf thumos.tar.gz
```
if you don't have gdown, you can also download the THUMOS features from [here](https://drive.google.com/file/d/1zt2eoldshf99vJMDuu8jqxda55dCyhZP/view?usp=sharing)

2. download and unzip experimental data

- 1) [Baseline 1408 features](https://huggingface.co/datasets/OpenGVLab/VideoMAEv2-TAL-Features/resolve/main/th14_mae_g_16_4.tar.gz)
- 3) [Baseline PCA-50 features](https://utexas.app.box.com/s/tisrjpglsqublqh9yaghltqskze8ex9e/file/1844650508672)
- 5) [PCA-50 SFA features](https://utexas.app.box.com/s/tisrjpglsqublqh9yaghltqskze8ex9e/file/1844967056835)
- 7) [SFA No PCA features](https://utexas.app.box.com/s/9e2sx6xoji97z6p8uody8bjcum8lu66d)

TO DO: 
- 2) [Baseline 1408 features downsampled](https://utexas.app.box.com/s/tisrjpglsqublqh9yaghltqskze8ex9e/file/1845742706066)
- 4) [Baseline PCA-50 features downsampled](https://utexas.app.box.com/s/tisrjpglsqublqh9yaghltqskze8ex9e/file/1845732681557)
- 6) [PCA-SFA features downsampled](https://utexas.app.box.com/s/tisrjpglsqublqh9yaghltqskze8ex9e/file/1845734330208)

2. add and modify some files
```
mv actionformer_files/thumos_mae.yaml actionformer_release/configs
rm actionformer_release/libs/utils/metrics.py
mv actionformer_files/metrics.py actionformer_release/libs/utils
rm actionformer_release/eval.py
mv actionformer_files/eval.py actionformer_release/
rm actionformer_release/libs/datasets/thumos14.py
mv thumos14.py actionformer_release/libs/datasets/
```

3. ActionFormer setup
```
cd actionformer_release/libs/utils
python setup.py install --user
```

4. adjust config file: dataset.json_file and dataset.feat_folder. dataset.input_dim and dataset.file_ext may also be of interest to you. see sample config files in [actionformer_files](actionformer_files) folder.

5. run train/eval
```
cd actionformer_release
python train.py configs/thumos_mae.yaml --output test
python eval.py ./configs/thumos_mae.yaml ./ckpt/thumos_mae_test
```

6. add results to [plaintext_results.txt](plaintext_results.txt)

reference: 
- https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/TAD.md
- https://github.com/happyharrycn/actionformer_release/tree/main?tab=readme-ov-file#to-reproduce-our-results-on-thumos14

    https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2/single_modality