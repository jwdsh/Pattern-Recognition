import argparse
import csv
import random
import sys
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
    # 固定随机种子，方便同一组实验重复运行时结果尽量一致。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_device(device_arg):
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, device):
    # 测试阶段关闭梯度，并切换到 eval 模式，避免 BatchNorm/Dropout 影响评估。
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

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


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
    parser = argparse.ArgumentParser(description="Supervised CIFAR-10 baseline")
    parser.add_argument("--data-root", default="./datasets")
    parser.add_argument("--num-labels", type=int, default=40, choices=[40, 250, 4000])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
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

    labeled_dataset, _, test_dataset, _ = get_cifar10_ssl_datasets(
        root=args.data_root,
        num_labels=args.num_labels,
        seed=args.seed,
        download=args.download,
    )

    # 纯监督 baseline 只使用 labeled_dataset，不使用无标注数据。
    train_loader = DataLoader(
        labeled_dataset,
        batch_size=args.batch_size,
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
    checkpoint_path = (
        Path(args.checkpoint_dir) / f"supervised_{args.num_labels}_labels_best.pt"
    )
    log_path = (
        Path(args.log_dir)
        / f"supervised_labels{args.num_labels}_epochs{args.epochs}_bs{args.batch_size}_seed{args.seed}.csv"
    )
    log_fields = [
        "epoch",
        "train_loss",
        "train_acc",
        "test_loss",
        "test_acc",
        "best_acc",
    ]
    init_csv_logger(log_path, log_fields)

    print(f"Device: {device}")
    print(f"Labeled samples: {len(labeled_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            save_checkpoint(checkpoint_path, model, optimizer, epoch, test_acc, args)

        append_csv_log(
            log_path,
            log_fields,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "best_acc": best_accuracy,
            },
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train loss {train_loss:.4f} | train acc {train_acc:.4f} | "
            f"test loss {test_loss:.4f} | test acc {test_acc:.4f} | "
            f"best {best_accuracy:.4f}"
        )

    print(f"Best checkpoint: {checkpoint_path}")
    print(f"CSV log: {log_path}")


if __name__ == "__main__":
    main()
