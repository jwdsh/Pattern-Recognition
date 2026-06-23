import argparse
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)


def get_train_transform():
    # 标注数据的普通训练增强：随机翻转 + 随机裁剪，是 CIFAR-10 常用设置。
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def get_test_transform():
    # 测试阶段不能使用随机增强，只做张量转换和归一化，保证评估结果稳定。
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def get_weak_transform():
    # FixMatch 中弱增强用于产生伪标签，通常保持和普通训练增强接近。
    return get_train_transform()


def get_strong_transform():
    # 强增强用于训练无标注样本，让模型在更强扰动下仍预测成同一个伪标签。
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandAugment(num_ops=2, magnitude=10),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def split_labeled_indices(targets, num_labels, num_classes=10, seed=0):
    if num_labels % num_classes != 0:
        raise ValueError("num_labels must be divisible by num_classes.")

    labels_per_class = num_labels // num_classes
    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)

    # 半监督实验要求每个类别均匀抽取少量标注样本，例如 40 张就是每类 4 张。
    labeled_indices = []
    for class_id in range(num_classes):
        class_indices = np.where(targets == class_id)[0]
        selected = rng.choice(class_indices, labels_per_class, replace=False)
        labeled_indices.extend(selected.tolist())

    labeled_indices = np.asarray(labeled_indices)
    rng.shuffle(labeled_indices)
    return labeled_indices.tolist()


class CIFAR10Labeled(Dataset):
    """只暴露被抽中的少量标注样本，返回 image 和真实 label。"""

    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, target = self.base_dataset[self.indices[index]]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class CIFAR10Unlabeled(Dataset):
    """无标注数据返回同一张图片的弱增强和强增强版本，供 FixMatch 使用。"""

    def __init__(self, base_dataset, transform_weak, transform_strong):
        self.base_dataset = base_dataset
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, _ = self.base_dataset[index]
        # 弱增强图像用于产生伪标签，强增强图像用这个伪标签计算无监督损失。
        weak_image = self.transform_weak(image)
        strong_image = self.transform_strong(image)
        return weak_image, strong_image


def get_cifar10_ssl_datasets(root="./datasets", num_labels=40, seed=0, download=True):
    # base_train 不带 transform，方便 labeled/unlabeled 对同一原图应用不同增强。
    base_train = datasets.CIFAR10(root=root, train=True, download=download)
    test_dataset = datasets.CIFAR10(
        root=root,
        train=False,
        transform=get_test_transform(),
        download=download,
    )

    labeled_indices = split_labeled_indices(base_train.targets, num_labels, seed=seed)

    # labeled_dataset 只包含少量有标签样本；unlabeled_dataset 默认使用全部训练图片。
    labeled_dataset = CIFAR10Labeled(
        base_dataset=base_train,
        indices=labeled_indices,
        transform=get_train_transform(),
    )
    unlabeled_dataset = CIFAR10Unlabeled(
        base_dataset=base_train,
        transform_weak=get_weak_transform(),
        transform_strong=get_strong_transform(),
    )

    return labeled_dataset, unlabeled_dataset, test_dataset, labeled_indices


def count_labeled_classes(base_dataset, indices):
    # 用于检查划分是否均衡，比如 40 张时应得到每类 4 张。
    labels = [base_dataset.targets[index] for index in indices]
    counts = Counter(labels)
    return [counts[class_id] for class_id in range(10)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./datasets")
    parser.add_argument("--num-labels", type=int, default=40, choices=[40, 250, 4000])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    labeled, unlabeled, test, labeled_indices = get_cifar10_ssl_datasets(
        root=args.root,
        num_labels=args.num_labels,
        seed=args.seed,
        download=not args.no_download,
    )

    class_counts = count_labeled_classes(labeled.base_dataset, labeled_indices)
    weak_image, strong_image = unlabeled[0]

    print(f"Labeled samples: {len(labeled)}")
    print(f"Unlabeled samples: {len(unlabeled)}")
    print(f"Test samples: {len(test)}")
    print(f"Class counts: {class_counts}")
    print(f"Weak image tensor shape: {tuple(weak_image.shape)}")
    print(f"Strong image tensor shape: {tuple(strong_image.shape)}")

    assert len(labeled) == args.num_labels
    assert class_counts == [args.num_labels // 10] * 10
    assert isinstance(weak_image, torch.Tensor)
    assert isinstance(strong_image, torch.Tensor)


if __name__ == "__main__":
    main()
