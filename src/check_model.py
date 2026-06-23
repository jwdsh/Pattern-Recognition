import torch

from wideresnet import WideResNet


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main():
    # 用随机 CIFAR-10 尺寸输入检查模型是否能正常前向传播到 10 类 logits。
    model = WideResNet(depth=28, widen_factor=2, num_classes=10)
    images = torch.randn(4, 3, 32, 32)
    logits = model(images)

    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    assert logits.shape == (4, 10)


if __name__ == "__main__":
    main()
