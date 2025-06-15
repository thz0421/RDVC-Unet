import os
from network.network_RDVC import RDVCUNET
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from monai.utils import set_determinism
from monai.transforms import AsDiscrete, Resize
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.metrics import SurfaceDistanceMetric
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from load_datasets_transforms import data_loader, data_transforms
from tqdm import tqdm
import argparse
import datetime
import csv

parser = argparse.ArgumentParser(description='Enhanced 3D Medical Image Segmentation via Residual Deformable Volumetric Convolutional Networks')
## Input data hyperparameters
parser.add_argument('--root', type=str, default='/root/myenv/3DUXNET/train/flare/', required=False, help='Root folder of all your images and labels')
parser.add_argument('--output', type=str, default='./flare best model', required=False, help='Output folder for both tensorboard and the best model')
parser.add_argument('--dataset', type=str, default='flare', required=False, help='Datasets: {feta, flare, amos, subtask1}, Fyi: You can add your dataset here')

## Input model & training hyperparameters
parser.add_argument('--network', type=str, default='RDVCUNET', help='Network models: {RDVCUNET}')
parser.add_argument('--mode', type=str, default='train', help='Training or testing mode')
parser.add_argument('--batch_size', type=int, default='1', help='Batch size for subject input')
parser.add_argument('--crop_sample', type=int, default='1', help='Number of cropped sub-volumes for each subject')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate for training')
parser.add_argument('--optim', type=str, default='AdamW', help='Optimizer types: Adam / AdamW')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
parser.add_argument('--max_iter', type=int, default=60000, help='Maximum iteration steps for training')
parser.add_argument('--eval_step', type=int, default=500, help='Per steps to perform validation')

## Efficiency hyperparameters
parser.add_argument('--gpu', type=str, default='0', help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=0.1, help='Cache rate to cache your dataset into GPUs')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print('Used GPU: {}'.format(args.gpu))

train_samples, valid_samples, out_classes = data_loader(args)

train_files = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_samples['images'], train_samples['labels'])
]

val_files = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(valid_samples['images'], valid_samples['labels'])
]

set_determinism(seed=0)

train_transforms, val_transforms = data_transforms(args)
dice_file = "dice{}.csv".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

## Train Pytorch Data Loader and Caching
print('Start caching datasets!')
train_ds = CacheDataset(
    data=train_files, transform=train_transforms,
    cache_rate=args.cache_rate, num_workers=args.num_workers)


train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
for batch in train_loader:

    break
## Valid Pytorch Data Loader and Caching
val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.num_workers)

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

print('Chosen Network Architecture: {}'.format(args.network))



loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
print('Loss for training: {}'.format('DiceCELoss'))
if args.optim == 'AdamW':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optim == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_iter, eta_min=1e-6)
print('Optimizer for training: {}, learning rate: {}'.format(args.optim, args.lr))

root_dir = os.path.join(args.output)
if os.path.exists(root_dir) == False:
    os.makedirs(root_dir)

t_dir = os.path.join(root_dir, 'tensorboard', '{}'.format(args.network))
if os.path.exists(t_dir) == False:
    os.makedirs(t_dir)
writer = SummaryWriter(log_dir=t_dir)


def validation(epoch_iterator_val,  model, device, out_classes):
    model.eval()

    dice_vals = {i: [] for i in range(out_classes)}
    dice_per_class = {i: -float('inf') for i in range(out_classes)}
    asd_per_class = {i: float('inf') for i in range(out_classes)}
    dice_val_overall = -float('inf')
    asd_val_overall = float('inf')

    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = batch["image"].to(device), batch["label"].to(device)
            original_shape = tuple(val_labels.shape[2:])
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 2, model)
            val_outputs = val_outputs.to(device)
            val_outputs_resized = F.interpolate(val_outputs, size=original_shape, mode="trilinear", align_corners=True)

            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(lbl) for lbl in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs_resized)
            val_output_convert = [post_pred(out) for out in val_outputs_list]

            for i in range(out_classes):
                dice_metric(y_pred=[torch.tensor(out[i:i + 1]).to(device) for out in val_output_convert],
                            y=[lbl[i:i + 1].to(device) for lbl in val_labels_convert])
                dice = dice_metric.aggregate().item()
                dice_vals[i].append(dice)
                dice_metric.reset()

                if dice > dice_per_class[i]:
                    dice_per_class[i] = dice
                if dice > dice_val_overall:
                    dice_val_overall = dice

                asd_metric(y_pred=[torch.tensor(out[i:i + 1]).to(device) for out in val_output_convert],
                           y=[lbl[i:i + 1].to(device) for lbl in val_labels_convert])
                asd = asd_metric.aggregate().item()
                asd_metric.reset()

                if asd < asd_per_class[i]:
                    asd_per_class[i] = asd
                if asd < asd_val_overall:
                    asd_val_overall = asd

            epoch_iterator_val.set_description(f"Validate Step: {step + 1} / {len(epoch_iterator_val)}")

    return dice_per_class, dice_val_overall, asd_per_class, asd_val_overall


def train(global_step, train_loader, dice_val_best, global_step_best, asd_val_best):
    model.train()
    epoch_loss = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)

    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = batch["image"].cuda(), batch["label"].cuda()
        logit_map = model(x)
        if isinstance(logit_map, list):
            logit_map = logit_map[0]
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss))

        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps)", dynamic_ncols=True)
            dice_per_class, dice_val_overall, asd_per_class, asd_val_overall = validation(
                epoch_iterator_val, global_step, model, val_loader, device, out_classes, args)

            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)

            print(f"epoch/validation:{global_step // eval_num}")
            print(f"train_loss: {epoch_loss:.4f}")
            print(f"dice_per_class: {dice_per_class}")
            print(f"dice_val_overall: {dice_val_overall}")
            print(f"asd_per_class: {asd_per_class}")
            print(f"asd_val_overall: {asd_val_overall}")


            if dice_val_overall > dice_val_best:
                dice_val_best = dice_val_overall
                global_step_best = global_step
                torch.save(model.state_dict(), os.path.join(root_dir, f"{args.network}_best_model_{args.dataset}_dice.pth"))
                print(f"Model Was Saved! Current Best Max Dice: {dice_val_best}")
            else:
                print(f"Model Was Not Saved! Current Best Max Dice: {dice_val_best}")

            if asd_val_overall < asd_val_best:
                asd_val_best = asd_val_overall
                torch.save(model.state_dict(), os.path.join(root_dir, f"{args.network}_best_model_{args.dataset}_asd.pth"))
                print(f"Model Was Saved! Current Best Min ASD: {asd_val_best}")
            else:
                print(f"Model Was Not Saved! Current Best Min ASD: {asd_val_best}")


            with open(dice_file, 'a', encoding='utf-8') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([
                    f"epoch/validation:{global_step // eval_num}",
                    f"train_loss: {epoch_loss:.4f}",
                    f"dice_per_class: {dice_per_class}",
                    f"dice_val_overall: {dice_val_overall}",
                    f"asd_per_class: {asd_per_class}",
                    f"asd_val_overall: {asd_val_overall}"
                ])

        writer.add_scalar(f'{args.network}_Training_Segmentation_Loss', loss.data, global_step)
        global_step += 1

    return global_step, dice_val_best, global_step_best, asd_val_best

max_iterations = args.max_iter
eval_num = args.eval_step
post_label = AsDiscrete(to_onehot=out_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=out_classes)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
asd_metric = SurfaceDistanceMetric(include_background=False, reduction="mean")
global_step = 0
dice_val_best = 0.0
asd_val_best = float('inf')
global_step_best = 0
epoch_loss_values = []
metric_values = []

while global_step < max_iterations:
    global_step, dice_val_best, global_step_best, asd_val_best = train(global_step, train_loader, dice_val_best, global_step_best, asd_val_best)