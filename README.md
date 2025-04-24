# VR SFA

## Feature Loading
1. download TemporalBench dataset
2. Run feature_extraction.py and minimally run with:
``` python feature_extraction.py --root_dir=<PATH_TO_HIGHEST_VIDEO_DIRECTORY> --feature_dir=<PATH_TO_SAVE_FEATURES_TO> --token=<HUGGINGFACE_ACCESS_TOKEN> ```
This will run it with an extraction batch size of 16 and for VideoMAEv2-Large on the temporal bench dataset. You can also set `--overwrite` to re-extract features. One final thing to consider is `--use_dataset` for when you're using the dataset to load features. 

This can be used to both extract features (if features don't exist it will save them out) and load features. The features will be of size (t, feature_dim).

*NOTE: DOWNSAMPLING
To trigger downsampling with default features, use `--downsample` in the command. I only recommend potentially changing `--downsample_type` from uniform to mean as an ablation/test case. 

## Training SFA
Install `emaae` package:
To install, use

```
$ git clone https://github.com/dwiepert/emaae.git
$ cd emaae
$ pip install . 
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