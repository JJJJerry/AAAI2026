## Installation：

Use python version 3.10.0:
```bash
conda create -n MCoT-MVS python=3.10.0
```
Then install the requirements:
```bash
pip install -r requirements.txt
```

## Segment:
If you want to get masks and seg feature by yourself,please follow the steps:

1.download some checkpoints:
ram++: https://huggingface.co/xinyu1205/recognize-anything-plus-model/blob/main/ram_plus_swin_large_14m.pth
put it to segment/ram_models directory

sam2: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
put it to segment/sam2/checkpoints directory

2.get masks
```bash
cd segment
python get_mask.py
```

3.get seg feature
```bash
python extract_seg_feature_cirr.py
python extract_seg_feature_fiq.py
```

or you can directly download the seg feature we uploaded:



## Train 
### cirr
```bash
cd src
CUDA_VISIBLE_DEVICES=x python train.py --dataset cirr --batch_size 16 --loss_weight 100
```

### fiq
```bash
cd src
CUDA_VISIBLE_DEVICES=x python train.py --dataset dress --batch_size 16 --loss_weight 10 --lr 1e-5
```

## inference
### cirr
```bash
cd src
CUDA_VISIBLE_DEVICES=x python cirr_test_submission.py --model_path xxx --submission_name test
```

### fiq
```bash
cd src
CUDA_VISIBLE_DEVICES=x python fiq_validate.py --model_path xxx --dress_type dress
```
