# USB FixMatch 对比实验

本目录用于记录 USB（Semi-supervised Learning，原 TorchSSL 升级维护版本）中的 FixMatch 对比实验。

## 1. 安装 USB

官方仓库：

https://github.com/microsoft/Semi-supervised-learning

推荐直接 clone 官方仓库到本目录外或本目录内：

```powershell
cd "D:\school\G3\Pattern Recognition\hw3\usb_experiment"
git clone https://github.com/microsoft/Semi-supervised-learning.git USB
cd USB
pip install -r requirements.txt
pip install -e .
```

如果网络允许，也可以尝试：

```powershell
pip install semilearn
```

## 2. 先跑官方 CIFAR-10 FixMatch 配置

USB 官方配置通常位于：

```text
USB/config/classic_cv/fixmatch/
```

先找 CIFAR-10、250 labels 的配置文件，名称可能类似：

```text
fixmatch_cifar10_250_0.yaml
```

运行方式通常是：

```powershell
python train.py --c config/classic_cv/fixmatch/fixmatch_cifar10_250_0.yaml
```

如果命令参数有变化，以 USB 仓库 README 为准。

## 3. 与自己实现保持公平

自己实现的 250 labels 设置：

```text
epochs = 50
batch_size = 32
mu = 7
unlabeled batch size = 32 * 7 = 224
threshold = 0.95
backbone = WideResNet-28-2
```

由于 CIFAR-10 训练集有 50000 张图，每个 epoch 约：

```text
50000 / 224 ≈ 223 steps
```

所以 50 epochs 约：

```text
223 * 50 ≈ 11150 train iterations
```

如果 USB 配置使用 iteration，而不是 epoch，请把训练步数设为约：

```text
num_train_iter: 11200
```

评估间隔可设为：

```text
num_eval_iter: 223
```

或为了少评估一些：

```text
num_eval_iter: 1000
```

## 4. 需要重点检查的配置项

尽量让 USB 配置和自实现一致：

```text
algorithm: fixmatch
dataset: cifar10
num_labels: 250
net: wrn_28_2 或 WideResNet-28-2
batch_size: 32
uratio / mu: 7
p_cutoff / threshold: 0.95
num_train_iter: 11200
num_classes: 10
```

不同 USB 版本配置字段名可能不同，例如：

```text
uratio
p_cutoff
num_train_iter
num_eval_iter
net
```

以实际 yaml 文件中的字段为准。

## 5. 记录结果

完成 USB 训练后，记录最佳测试准确率，并和自实现结果对比：

| 方法 | labels | 训练步数/epoch | batch size | mu | threshold | best acc |
|---|---:|---:|---:|---:|---:|---:|
| My FixMatch | 250 | 50 epochs, about 11150 steps | 32 | 7 | 0.95 | 72.00% |
| USB FixMatch | 250 | about 11200 steps | 32 | 7 | 0.95 | 待填 |

## 6. 报告中可以这样说明

为保证公平性，本文将自实现 FixMatch 与 USB FixMatch 设置为相近训练步数和主要超参数。由于 USB 内部可能包含更完整的数据增强、学习率调度、EMA 或工程优化，因此其结果可能优于自实现版本。本文主要比较两者在相同数据划分与相近训练预算下的性能差异。
