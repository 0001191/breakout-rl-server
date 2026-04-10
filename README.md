# Breakout RL

这个目录现在专门放打砖块强化学习项目。
重点是训练过程本身，不是做一个花哨游戏壳子。

## 包含内容

- `train_breakout.py`
  训练 `ALE/Breakout-v5` 上的 DQN，并持续写出实时训练状态
- `evaluate_breakout.py`
  评估模型，也可以导出 AI 自己玩的录像
- `dashboard/`
  训练看板前端，显示训练曲线、预览帧和最近日志
- `launch_dashboard.py`
  一键启动本地看板服务
- `server_app.py`
  服务器版看板服务，对外提供页面、状态 JSON、预览图和日志
- `start_training_server.py`
  在服务器上后台启动训练

## 安装

```powershell
cd D:\打砖块
pip install -r requirements.txt
```

如果 ROM 还没装好：

```powershell
AutoROM --accept-license
```

## 训练

```powershell
python train_breakout.py --root-dir .
```

常用参数示例：

```powershell
python train_breakout.py --total-timesteps 500000 --n-envs 4 --buffer-size 50000 --preview-freq 10000 --device cuda
```

## 看训练过程

先开看板服务：

```powershell
python launch_dashboard.py
```

然后再启动训练。训练脚本会持续更新这些文件：

- `artifacts/live/status.json`
- `artifacts/live/history.json`
- `artifacts/live/preview.png`

看板里能看到：

- 当前训练步数
- 训练进度
- 最近 100 局平均奖励
- epsilon
- loss
- 当前策略预览帧
- 最近训练日志

## 部署到服务器

把整个 `D:\打砖块` 目录传到服务器后，进入项目目录：

```bash
pip install -r requirements.txt
python server_app.py
```

然后在另一个终端启动训练：

```bash
python start_training_server.py --total-timesteps 2000000 --device cuda
```

浏览器打开：

```text
http://<服务器IP>:8000/dashboard/
```

如果只是想快速初始化 Linux 服务器，也可以参考：

```bash
bash deploy_server.sh /home/yourname/breakout_rl_server
```

## TensorBoard

```powershell
tensorboard --logdir artifacts/logs
```

## 评估

```powershell
python evaluate_breakout.py --model-path .\artifacts\models\<run_name>\final_model.zip --deterministic
```

## 录像

```powershell
python evaluate_breakout.py --model-path .\artifacts\models\<run_name>\final_model.zip --deterministic --record-video
```

## 说明

- 训练看板显示的是训练中的当前策略预览，不是成品回放。
- 你之前截图里“看不到下面的板”，更可能是播放器底部控件把挡板盖住了，不是环境没画出来。
- 现在看板里的 `preview.png` 是原始环境帧，不会被播放器控件挡住。
- DeepMind 那套 Atari DQN 通常是按帧数和训练步数统计，不是简单只看“打了几局”。
