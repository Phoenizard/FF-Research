阶段1:复现FF已有算法

### 准确率（%）

| | CIFAR10 | CIFAR100 | TinyIN |
|---|---|---|---|
| BiCovG | 90.35 | 66.09 | XX |
| BiCovG (Random Readout) | 88.05 | 54.36 | 40.85 |
| ASGE | 87.97 | 57.64 | 43.63 |
| DTG-FF | 89.80 | 60.13 | XX |
| CwC | 83.89 | 53.07 | XX |
| DeeperForward | 86.85 | 57.32 | 39.54 |
| HCL-FF | 85.93 | 60.52 | XX |
| Global BP | 90.41 | 63.34 | XX |

### 训练总时长（分钟，不含评估）

| | CIFAR10 | CIFAR100 | TinyIN |
|---|---|---|---|
| BiCovG | 95.81 | 71.87 | XX |
| BiCovG (Random Readout) | 69.78 | 70.88 | 869.93 |
| ASGE | 54.90 | 87.49 | 671.14 |
| DTG-FF | 55.95 | 54.70 | XX |
| CwC | 48.16 | 48.44 | XX |
| DeeperForward | 45.67 | 47.62 | 562.06 |
| HCL-FF | 71.01 | 76.76 | XX |
| Global BP | 77.28 | 78.20 | XX |
