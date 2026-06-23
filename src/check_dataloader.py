import sys
from pathlib import Path

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.dataset import get_cifar10_ssl_datasets


def main():
    # 只用于检查数据加载流程：标注 batch、无标注弱/强增强 batch 是否形状正确。
    labeled_dataset, unlabeled_dataset, test_dataset, _ = get_cifar10_ssl_datasets(
        root="./datasets",
        num_labels=40,
        seed=0,
        download=False,
    )

    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    images_l, targets_l = next(iter(labeled_loader))
    weak_u, strong_u = next(iter(unlabeled_loader))

    # 期望输出：[B, 3, 32, 32]、[B]，以及无标注数据的两种增强版本。
    print(images_l.shape, targets_l.shape)
    print(weak_u.shape, strong_u.shape)
    print(targets_l)
    print(f"Test samples: {len(test_dataset)}")


if __name__ == "__main__":
    main()
