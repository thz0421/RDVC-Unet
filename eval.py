import csv

from monai.utils import set_determinism
from monai.data import CacheDataset, DataLoader, decollate_batch
import torch
from load_datasets_transforms import data_loader, data_transforms
import os
import argparse
from network.network_RDVC import RDVCUNET

parser = argparse.ArgumentParser(description='Enhanced 3D Medical Image Segmentation via Residual Deformable Volumetric Convolutional Networks')
## Input data hyperparameters
parser.add_argument('--root', type=str, default='/root/myenv/3DUXNET/train/flare', required=False, help='Root folder of all your images and labels')
parser.add_argument('--output', type=str, default='./flare_result', required=False, help='Output folder for both tensorboard and the best model')
parser.add_argument('--dataset', type=str, default='flare', required=False, help='Datasets: {feta, flare, amos, subtask1}, Fyi: You can add your dataset here')

## Input model & training hyperparameters
parser.add_argument('--network', type=str, default='RDVCUNET', help='Network models: {RDVCUNET}')
parser.add_argument('--mode', type=str, default='eval', help='Training or testing mode')
parser.add_argument('--batch_size', type=int, default='1', help='Batch size for subject input')
parser.add_argument('--crop_sample', type=int, default='2', help='Number of cropped sub-volumes for each subject')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for training')
parser.add_argument('--optim', type=str, default='AdamW', help='Optimizer types: Adam / AdamW')
parser.add_argument('--max_iter', type=int, default=40000, help='Maximum iteration steps for training')
parser.add_argument('--eval_step', type=int, default=500, help='Per steps to perform validation')

## Efficiency hyperparameters
parser.add_argument('--gpu', type=str, default='0', help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=0.1, help='Cache rate to cache your dataset into GPUs')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
parser.add_argument('--trained_weights', default='./RDVC_best_model_dice_flare_dice.pth', required=False, help='Path of trained weights')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print('Used GPU: {}'.format(args.gpu))

valid_samples, out_classes = data_loader(args)
val_transforms = data_transforms(args)

val_files = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(valid_samples['images'], valid_samples['labels'])
]


val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.num_workers)


set_determinism(seed=0)

device = torch.device("cuda:0")
print(f"验证集样本数: {len(val_ds)}")
print(f"验证批次数: {len(val_loader)}")

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

print('Chosen Network Architecture: {}'.format(args.network))


def validate_model(model, val_loader, args, out_classes, global_step=0, eval_num=1, epoch_loss=0.0, csv_path='/root/myenv/3DUXNET/RDVC-Unet/val_metrics.csv'):
    import logging, os, csv

    import torch
    from monai.metrics import DiceMetric, SurfaceDistanceMetric
    from monai.inferers import sliding_window_inference
    from monai.transforms import AsDiscrete
    from monai.data import decollate_batch
    from tqdm import tqdm

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        if hasattr(args, 'trained_weights') and args.trained_weights:
            if os.path.exists(args.trained_weights):
                logger.info(f"Loading model weights from {args.trained_weights}")
                state_dict = torch.load(args.trained_weights, map_location='cpu')
                model.load_state_dict(state_dict, strict=False)
            else:
                logger.warning(f"Weight file {args.trained_weights} not found, using current model weights")

        model.eval()

        asd_metric = SurfaceDistanceMetric(include_background=False, reduction="mean", symmetric=True)
        dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

        post_label = AsDiscrete(to_onehot=out_classes)
        post_pred = AsDiscrete(argmax=True, to_onehot=out_classes)

        dice_per_class = {i: -float('inf') for i in range(out_classes)}
        asd_per_class = {i: float('inf') for i in range(out_classes)}

        epoch_iterator_val = tqdm(val_loader, desc="Validating", dynamic_ncols=True)

        with torch.no_grad():
            for batch in epoch_iterator_val:
                val_inputs = batch["image"].cuda()
                val_labels = batch["label"].cuda()

                roi_size = getattr(args, 'roi_size', (96, 96, 96))
                sw_batch_size = getattr(args, 'sw_batch_size', 2)

                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model, overlap=0.25, mode="gaussian")

                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [post_label(lbl) for lbl in val_labels_list]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [post_pred(pred) for pred in val_outputs_list]

                for i in range(out_classes):
                    preds = [out[i:i+1] for out in val_output_convert]
                    gts = [lbl[i:i+1] for lbl in val_labels_convert]

                    valid_pairs = [(p, g) for p, g in zip(preds, gts) if torch.sum(g) > 0 or torch.sum(p) > 0]
                    if not valid_pairs:
                        continue

                    valid_preds, valid_gts = zip(*valid_pairs)
                    dice_metric(y_pred=list(valid_preds), y=list(valid_gts))
                    dice = dice_metric.aggregate().item()
                    dice_metric.reset()

                    if dice > dice_per_class[i]:
                        dice_per_class[i] = dice

                    if any(torch.sum(g) > 0 and torch.sum(p) > 0 for p, g in valid_pairs):
                        asd_metric(y_pred=list(valid_preds), y=list(valid_gts))
                        asd = asd_metric.aggregate().item()
                        asd_metric.reset()

                        if asd < asd_per_class[i]:
                            asd_per_class[i] = asd

        # 计算全局最大Dice和最小ASD
        dice_val_overall = max(dice_per_class.values()) if dice_per_class else -float('inf')
        asd_val_overall = min([v for v in asd_per_class.values() if v != float('inf')], default=float('inf'))

        # 保存到 CSV 文件
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        write_header = not os.path.exists(csv_path)
        with open(csv_path, mode='a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Step", "Train Loss"] +
                                [f"Dice_{i}" for i in range(out_classes)] +
                                ["Dice_Overall"] +
                                [f"ASD_{i}" for i in range(out_classes)] +
                                ["ASD_Overall"])
            writer.writerow([global_step // eval_num, f"{epoch_loss:.4f}"] +
                            [f"{dice_per_class[i]:.4f}" if dice_per_class[i] != -float('inf') else "NA" for i in range(out_classes)] +
                            [f"{dice_val_overall:.4f}" if dice_val_overall != -float('inf') else "NA"] +
                            [f"{asd_per_class[i]:.4f}" if asd_per_class[i] != float('inf') else "NA" for i in range(out_classes)] +
                            [f"{asd_val_overall:.4f}" if asd_val_overall != float('inf') else "NA"])

        results = {
            'dice_per_class': dice_per_class,
            'asd_per_class': asd_per_class,
            'dice_val_overall': dice_val_overall,
            'asd_val_overall': asd_val_overall,
        }
        return results

    except Exception as e:
        logger.error(f"评估过程出错: {str(e)}")
        return None
if __name__ == "__main__":
    validate_model(model, val_loader, args, out_classes)




