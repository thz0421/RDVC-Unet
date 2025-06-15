import numpy as np
import nibabel as nib
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch
import torch
from load_datasets_transforms import data_loader, data_transforms, infer_post_transforms
import os
import argparse
from network.network_RDVC import RDVCUNET


parser = argparse.ArgumentParser(description='Enhanced 3D Medical Image Segmentation via Residual Deformable Volumetric Convolutional Networks')
## Input data hyperparameters
parser.add_argument('--root', type=str, default='/root/myenv/3DUXNET/train/amos/imagesTest', required=False, help='Root folder of all your images and labels')
parser.add_argument('--output', type=str, default='./RDVC_flare_result', required=False, help='Output folder for both tensorboard and the best model')
parser.add_argument('--dataset', type=str, default='amos', required=False, help='Datasets: {flare}, Fyi: You can add your dataset here')

## Input model & training hyperparameters
parser.add_argument('--network', type=str, default='RDVCUNET', required=False, help='Network models: {RDVCUNET}')
parser.add_argument('--trained_weights', default='/root/myenv/3DUXNET/train/amos best model/D3D_best_model_dice.pth', required=False, help='Path of trained weights')
parser.add_argument('--mode', type=str, default='test', help='Training or testing mode')
parser.add_argument('--sw_batch_size', type=int, default=4, help='Sliding window batch size for inference')
parser.add_argument('--overlap', type=float, default=0.25, help='Sub-volume overlapped percentage')

## Efficiency hyperparameters
parser.add_argument('--gpu', type=str, default='0', help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=0.1, help='Cache rate to cache your dataset into GPUs')
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

test_samples, out_classes = data_loader(args)

test_files = [
    {"image": image_name} for image_name in zip(test_samples['images'])
]

set_determinism(seed=0)

test_transforms = data_transforms(args)
post_transforms = infer_post_transforms(args, test_transforms, out_classes)

## Inference Pytorch Data Loader and Caching
test_ds = CacheDataset(
    data=test_files, transform=test_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers)

## Load Networks
device = torch.device("cuda:0")

if args.network == 'RDVCUNET':
    model = RDVCUNET(
        in_chans=1,
        out_chans=out_classes,
        depths=[1, 1, 2, 1],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        spatial_dims=3,
    ).to(device)
model.load_state_dict(torch.load(args.trained_weights), strict=False)
model.eval()

os.makedirs(args.output, exist_ok=True)
print(f"Number of samples in test dataset: {len(test_ds)}")
print(f"Number of batches in test_loader: {len(test_loader)}")
with torch.no_grad():
    for i, test_data in enumerate(test_loader):
        images = test_data["image"].to(device)

        predictions = sliding_window_inference(
            images, roi_size=(96, 96, 96), sw_batch_size=args.sw_batch_size, predictor=model, overlap=args.overlap
        )

        pred_dict = {"pred": predictions}

        # 如果有元信息，直接放入pred_dict，保持对应batch维度
        if "meta_dict" in test_data:
            pred_dict["pred_meta_dict"] = test_data["meta_dict"]

        # decollate 后得到列表，逐个样本处理，同时传入对应的元信息
        preds_list = decollate_batch(pred_dict["pred"])
        if "pred_meta_dict" in pred_dict:
            meta_list = pred_dict["pred_meta_dict"]
        else:
            meta_list = [None] * len(preds_list)

        predictions_post = []
        for pred_tensor, meta in zip(preds_list, meta_list):
            data_dict = {"pred": pred_tensor}
            if meta is not None:
                data_dict["pred_meta_dict"] = meta
            prediction_post = post_transforms(data_dict)
            predictions_post.append(prediction_post)

        # 保存结果，逐个样本处理
        for j, prediction in enumerate(predictions_post):
            image_name = test_data["image_meta_dict"]["filename_or_obj"][j]
            save_path = os.path.join(args.output, os.path.basename(image_name).replace('.nii', '_segmentation.nii.gz'))

            pred_data = prediction.squeeze().cpu().numpy()
            pred_data = (pred_data > 0.5).astype(np.uint8)

            if "image_meta_dict" in test_data and "affine" in test_data["image_meta_dict"]:
                affine = test_data["image_meta_dict"]["affine"][j].cpu().numpy()
            else:
                affine = np.eye(4)

            pred_nii = nib.Nifti1Image(pred_data, affine)
            nib.save(pred_nii, save_path)
            print(f"Saved: {save_path}")