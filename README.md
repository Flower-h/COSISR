# COSISR
Jiaxu Zhang, Yaowen Lv

>Changchun university of science and technology

---

## Introduction

This study presents a general framework based on deep learning to address the catadioptric omnidirectional images super-resolution problem.

---

## Environment
- [PyTorch >= 1.7](https://pytorch.org/)
- [BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/INSTALL.md) 
#### Installation
```
pip install -r requirements.txt
```

## Data Preparation
> We propose a cylindrical projection model for transforming the data structure of catadioptric omnidirectional images.Based on this model, we generated a projection image dataset COCP-SR and a simulation dataset SIM-SR.
1. COCP-SR is generated from the publicly available dataset [Omnicam](https://www.cvlibs.net/projects/omnicam).
```bash
mkdir datasets
ln -s YOUR_LAU_DATASET_PATH COSISR/datasets/data
```
2. Image size correction.
```bash
python Data_preparation/img_reset.py
```
3. SIM-SR is generated from the publicly available dataset [DF2K](https://opendatalab.org.cn/df2k_ost).
```bash
ln -s YOUR_DIV2K_TRAINSET_PATH COISR/datasets/DIV2K_train_HR
ln -s YOUR_DIV2K_TRAINSET_PATH COISR/datasets/DIV2K_valid_HR
ln -s YOUR_FLICKR2K_TRAINSET_PATH COISR/datasets/Flickr2K_HR
```
4. Cylindrical projection
```bash
python Data_preparation/create_data_lists.py
python Data_preparation/cylindrical_projection.py
```
5. Crop training patches （COCP-SR and SIM-SR）
```bash
python Data_preparation/sub_image.py
```

## Training
1. Standard Training
We provide standard training options of COSISR-full and COSISR-light on X2 and X4 SR.

Training COSISR and other baseline models:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py -opt ./options/train/*.yml --launcher pytorch
```

2. Correction  Training
We provide correction training options of COSISR-sim on X2 and X4 SR.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py -opt ./options/train/*_sim.yml --launcher pytorch
```

## Testing 
Pretrained models can be downloaded [here](https://drive.google.com/drive/folders/1xd-xbRPwDeLVOq7qunlgIOxIJeJByfoi?usp=sharing).

1. Testing of Standard Training. The results are displayed in the file fig/X2 and fig/X4
```bash
ln -s YOUR_PRETRAIN_MODEL_PATH pretrained_models
CUDA_VISIBLE_DEVICES=0 python test.py -opt ./options/test/*.yml
```

2. Testing of Correction Training. The results are displayed in the file fig/COSISR_X2_sim and fig/COSISR_X4_sim
```bash
ln -s YOUR_PRETRAIN_MODEL_PATH pretrained_models
CUDA_VISIBLE_DEVICES=0 python test.py -opt ./options/test/*_sim.yml
```

## License and Acknowledgement
This project is released under the MIT license. The codes are heavily based on [BasicSR](https://github.com/XPixelGroup/BasicSR). Please also follow their licenses. Thanks for their awesome works.
