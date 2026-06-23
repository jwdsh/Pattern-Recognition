import argparse
import csv
import random
import sys
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.dataset import get_cifar10_ssl_datasets
from wideresnet import WideResNet


def set_seed(seed):
    # 固定随机种子，方便复现实验划分和训练过程。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_device(device_arg):
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_fixmatch_loss(
    model,
    images_l,
    targets_l,
    weak_u,
    strong_u,
    threshold,
    lambda_u,
):
    """计算 FixMatch 的有监督损失、无监督损失和总损失。"""
    logits_l = model(images_l)
    loss_x = F.cross_entropy(logits_l, targets_l)

    # 1. 用无标注样本的弱增强版本生成伪标签。
    with torch.no_grad():
        logits_weak = model(weak_u)
        probs = torch.softmax(logits_weak, dim=1)
        max_probs, pseudo_labels = torch.max(probs, dim=1)

        # 2. 只保留置信度大于阈值的伪标签，低置信度样本不参与无监督损失。
        mask = max_probs.ge(threshold).float()

    # 3. 用伪标签监督同一批无标注样本的强增强版本。
    logits_strong = model(strong_u)
    loss_u_all = F.cross_entropy(logits_strong, pseudo_labels, reduction="none")
    loss_u = (loss_u_all * mask).mean()

    loss = loss_x + lambda_u * loss_u

    with torch.no_grad():
        labeled_acc = (logits_l.argmax(dim=1) == targets_l).float().mean().item()
        mask_ratio = mask.mean().item()
        avg_confidence = max_probs.mean().item()

    return loss, loss_x, loss_u, labeled_acc, mask_ratio, avg_confidence


def train_one_epoch(
    model,
    labeled_loader,
    unlabeled_loader,
    optimizer,
    device,
    threshold,
    lambda_u,
    steps_per_epoch,
):
    model.train()
    labeled_iter = cycle(labeled_loader)

    total_loss = 0.0
    total_loss_x = 0.0
    total_loss_u = 0.0
    total_labeled_acc = 0.0
    total_mask_ratio = 0.0
    total_confidence = 0.0
    total_steps = 0

    for step, (weak_u, strong_u) in enumerate(unlabeled_loader, start=1):
        if steps_per_epoch is not None and step > steps_per_epoch:
            break

        images_l, targets_l = next(labeled_iter)
        images_l = images_l.to(device)
        targets_l = targets_l.to(device)
        weak_u = weak_u.to(device)
        strong_u = strong_u.to(device)

        loss, loss_x, loss_u, labeled_acc, mask_ratio, confidence = compute_fixmatch_loss(
            model=model,
            images_l=images_l,
            targets_l=targets_l,
            weak_u=weak_u,
            strong_u=strong_u,
            threshold=threshold,
            lambda_u=lambda_u,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_loss_x += loss_x.item()
        total_loss_u += loss_u.item()
        total_labeled_acc += labeled_acc
        total_mask_ratio += mask_ratio
        total_confidence += confidence
        total_steps += 1

    return {
        "loss": total_loss / total_steps,
        "loss_x": total_loss_x / total_steps,
        "loss_u": total_loss_u / total_steps,
        "labeled_acc": total_labeled_acc / total_steps,
        "mask_ratio": total_mask_ratio / total_steps,
        "confidence": total_confidence / total_steps,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = F.cross_entropy(logits, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def save_checkpoint(path, model, optimizer, epoch, accuracy, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "accuracy": accuracy,
            "args": vars(args),
        },
        path,
    )


def init_csv_logger(path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()


def append_csv_log(path, fieldnames, row):
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description="FixMatch on CIFAR-10")
    parser.add_argument("--data-root", default="./datasets")
    parser.add_argument("--num-labels", type=int, default=40, choices=[40, 250, 4000])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mu", type=int, default=7, help="unlabeled batch multiplier")
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--lambda-u", type=float, default=1.0)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = build_device(args.device)

    labeled_dataset, unlabeled_dataset, test_dataset, _ = get_cifar10_ssl_datasets(
        root=args.data_root,
        num_labels=args.num_labels,
        seed=args.seed,
        download=args.download,
    )

    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=args.batch_size * args.mu,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = WideResNet(depth=28, widen_factor=2, num_classes=10).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    best_accuracy = 0.0
    checkpoint_path = Path(args.checkpoint_dir) / f"fixmatch_{args.num_labels}_labels_best.pt"
    log_path = (
        Path(args.log_dir)
        / f"fixmatch_labels{args.num_labels}_epochs{args.epochs}_bs{args.batch_size}_mu{args.mu}_seed{args.seed}.csv"
    )
    log_fields = [
        "epoch",
        "loss",
        "loss_x",
        "loss_u",
        "labeled_acc",
        "mask_ratio",
        "confidence",
        "test_loss",
        "test_acc",
        "best_acc",
    ]
    init_csv_logger(log_path, log_fields)

    print(f"Device: {device}")
    print(f"Labeled samples: {len(labeled_dataset)}")
    print(f"Unlabeled samples: {len(unlabeled_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Threshold: {args.threshold}")
    print(f"Lambda_u: {args.lambda_u}")
    print(f"Unlabeled batch size: {args.batch_size * args.mu}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader,
            optimizer=optimizer,
            device=device,
            threshold=args.threshold,
            lambda_u=args.lambda_u,
            steps_per_epoch=args.steps_per_epoch,
        )
        test_loss, test_acc = evaluate(model, test_loader, device)

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            save_checkpoint(checkpoint_path, model, optimizer, epoch, test_acc, args)

        append_csv_log(
            log_path,
            log_fields,
            {
                "epoch": epoch,
                "loss": train_metrics["loss"],
                "loss_x": train_metrics["loss_x"],
                "loss_u": train_metrics["loss_u"],
                "labeled_acc": train_metrics["labeled_acc"],
                "mask_ratio": train_metrics["mask_ratio"],
                "confidence": train_metrics["confidence"],
                "test_loss": test_loss,
                "test_acc": test_acc,
                "best_acc": best_accuracy,
            },
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"loss {train_metrics['loss']:.4f} | "
            f"Lx {train_metrics['loss_x']:.4f} | "
            f"Lu {train_metrics['loss_u']:.4f} | "
            f"labeled acc {train_metrics['labeled_acc']:.4f} | "
            f"mask {train_metrics['mask_ratio']:.4f} | "
            f"conf {train_metrics['confidence']:.4f} | "
            f"test loss {test_loss:.4f} | "
            f"test acc {test_acc:.4f} | "
            f"best {best_accuracy:.4f}"
        )

    print(f"Best checkpoint: {checkpoint_path}")
    print(f"CSV log: {log_path}")


if __name__ == "__main__":
    main()
