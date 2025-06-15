# RDVC-Unet
This repo holds code for "Enhanced 3D Medical Image Segmentation via Residual Deformable Volumetric Convolutional Networks"
## Usage
### 1. Prepare data
* Download the MICCAI Challenge 2021 FLARE dataset from the official [website](https://flare.grand-challenge.org/).
* Download the MICCAI Challenge 2022 AMOS dataset from the official [website](https://amos22.grand-challenge.org/).
* Download the MSD Hepatic Vessel dataset from the official [website](http://medicaldecathlon.com/ ).

We initially divide different datasets in the following structure:
```
path to all data directory/
├── FLARE
├── AMOS
├── MSD
├── ...
```

We further sub-divide the samples into training, validation and testing as follow:

```
root_dir/
├── imagesTr
├── labelsTr
├── imagesVal
├── labelsVal
├── imagesTs
```

### 2. Environment
Please prepare an environment with python>=3.9, and then use the command "pip install -r requirements.txt" for the dependencies.
### 3. Train/Test/Eval
* Train
Command to start model training:
```
RDVC-UNet:
python train.py \
  --root /path/to/train_dataset \
  --output ./output_dir/train \
  --dataset flare \
  --batch_size 1 \
  --crop_sample 2 \
  --lr 0.0002 \
  --max_iter 60000 \
  --eval_step 500 \
  --gpu 0
```
* Eval
Evaluating Model Performance：
```
RDVC-UNet:
python eval.py \
  --root /path/to/val_dataset \
  --output ./output_dir/eval \
  --dataset flare \
  --batch_size 1 \
  --trained_weights ./output_dir/train/RDVCUNET_best_model_flare_dice.pth \
  --gpu 0
```

* Test
Visualization of model segmentation performance：
```
RDVC-UNet:
python test_seg.py \
  --root /path/to/test_dataset \
  --output ./output_dir/test_result \
  --dataset flare \
  --trained_weights ./output_dir/train/RDVCUNET_best_model_flare_dice.pth \
  --sw_batch_size 2 \
  --overlap 0.25 \
  --gpu 0
```
## Reference

## Citations
