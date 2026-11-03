# AI 工作流

## work in autodl

本地写代码，AutoDL 只跑实验。SSH 操作只有三类：更新代码、提交实验、获取结果。

### 分工

| 位置 | 做什么 | 不做什么 |
|---|---|---|
| 本地（Claude Code） | 写代码、提交推送、发 ssh 命令、拉回结果、汇报 | 不跑正式实验；用户未决定不上传 W&B |
| GitHub | 代码中转与备份 | — |
| AutoDL | 拉取代码、在 `gpu` 环境跑实验、写 W&B 离线记录 | 不改代码，不联网上传 W&B |

### 一次性配置

用户配置 SSH 密钥，并在每次任务时提供实例地址，密码不出现在对话里。下文命令中的 `autodl` 代表该地址。AutoDL 仓库位于 `autodl-tmp/FF-Research`。

### ① 更新代码

本地推送到 GitHub：

```bash
git push origin <分支>
```

AutoDL 拉取：

```bash
ssh autodl 'cd autodl-tmp/FF-Research && source /etc/network_turbo && git fetch && git checkout <分支> && git pull && unset http_proxy https_proxy && git status -s && git log -1 --oneline'
```

`git status -s` 必须无输出；记录 commit 号，与本次实验对应。

### ② 提交实验

1. 确认没有其他实验在运行：

```bash
ssh autodl 'screen -ls'
```

2. 启动：

```bash
ssh autodl 'cd autodl-tmp/FF-Research && mkdir -p results/logs && screen -dmS <实验名> bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate gpu && WANDB_MODE=offline python train.py --datasets <数据集> --method <方法> --seeds <种子> --batch-size <batch_size> --wandb-mode offline > results/logs/<实验名>.log 2>&1"'
```

BiCovG 随机 readout 版加 `--bicovg-readout random --output results/bicovg-random`，避免与可学习版的输出目录冲突。

### ③ 获取结果

查看进度：

```bash
ssh autodl 'tail -n 2 autodl-tmp/FF-Research/results/logs/<实验名>.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'
```

实验结束后拉回本地，排除 checkpoint，保留 W&B 离线记录：

```bash
rsync -a --exclude '*.pt' autodl:autodl-tmp/FF-Research/results/ results/
```

向用户汇报 Fusion、Best Pred、训练时长、峰值显存。

### ④ 上传 W&B

仅在本地执行，用户审核后明确同意才上传：

```bash
wandb sync results/<数据集>/<方法>/seed<种子>/wandb/offline-run-*
```

不上传的实验保留在本地 `results/`，是否删除由用户决定。

### 规则

- AutoDL 上的 Python 命令一律在 `conda activate gpu` 后执行；安装依赖也在该环境内用 `pip install` 或 `conda install`。
- 加速只覆盖 GitHub 和 HuggingFace：AutoDL 访问 GitHub 的 git 操作前 `source /etc/network_turbo`，结束后立即 `unset http_proxy https_proxy`；实验命令不开加速。
- 实验运行期间 AutoDL 不能关机。
