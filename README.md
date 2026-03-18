# leo_handover_demo — 基于深度学习的低轨卫星切换策略

本项目实现了一种基于**深度强化学习（Double DQN）**的低轨卫星（LEO）切换策略，
并与多种传统切换算法进行了系统对比分析。

---

## 项目背景

低轨卫星通信系统（如 Starlink、OneWeb 等）的卫星以约 7.6 km/s 的速度高速运动，
单颗卫星每次仅可见约 10 分钟，用户终端必须频繁切换接入卫星。不合理的切换策略会导致：

- **乒乓效应**：频繁切换增加信令开销与延迟
- **过晚切换**：信号质量持续下降
- **切换失败**：短暂通信中断

本项目提出基于 DQN 的智能切换策略，通过强化学习自动学习最优切换决策。

---

## 系统模型

| 参数 | 值 |
|------|-----|
| 轨道高度 | 550 km（类 Starlink Shell-1） |
| 载波频率 | 28 GHz（Ka 频段） |
| 星座规模 | 20 颗卫星，最多同时可见 6 颗 |
| 最低仰角 | 10° |
| 仿真步长 | 10 秒 |
| 回合长度 | 2 个轨道周期（约 191 分钟） |

**状态空间**（每步观测）：可见卫星的 `[RSRP, 仰角, 剩余可见时间, 负载]`，共 24 维，归一化到 `[-1, 1]`。

**动作空间**：选择连接哪颗可见卫星（0 … max\_visible − 1）。

**奖励函数**：
```
reward = rsrp_质量奖励 − 切换惩罚 × I(切换发生) − 中断惩罚 × I(无卫星可见) − 0.1 × 负载
```

---

## DQN 算法

| 组件 | 实现 |
|------|------|
| 网络结构 | 全连接 24 → 256 → 256 → 6（ReLU 激活） |
| 目标网络 | 硬更新（每 10 次梯度步后同步） |
| 经验回放 | 容量 100 000，随机均匀采样 |
| Double DQN | 策略网络选动作，目标网络评估 Q 值 |
| 探索策略 | ε-贪婪，指数衰减（1.0 → 0.05） |
| 梯度裁剪 | max-norm = 10 |
| 优化器 | Adam，lr = 1e-4 |

---

## 对比基线

| 算法 | 描述 |
|------|------|
| **DQN**（本方法） | 深度强化学习智能切换 |
| Best RSRP | 始终选择信号最强的卫星 |
| RSRP+Hysteresis(3dB) | 带 3 dB 滞后的信号强度切换（类 3GPP A3 事件） |
| Max Elevation | 选择仰角最高的卫星 |
| Longest Remaining Time | 选择剩余可见时间最长的卫星 |

---

## 安装

```bash
pip install -r requirements.txt
```

Python ≥ 3.10，主要依赖：`torch`, `gymnasium`, `numpy`, `matplotlib`。

---

## 使用方法

### 训练 DQN 智能体

```bash
python train.py --n_episodes 500 --save_dir checkpoints
```

可选参数（`python train.py --help` 查看全部）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n_episodes` | 500 | 训练回合数 |
| `--lr` | 1e-4 | Adam 学习率 |
| `--handover_penalty` | 0.5 | 切换惩罚系数 |
| `--save_dir` | `checkpoints/` | 模型保存目录 |

### 评估与对比

```bash
python evaluate.py --model_dir checkpoints --output_dir results
```

输出：控制台汇总表 + `results/comparison.png` + `results/training_curves.png`。

---

## 项目结构

```
leo_handover_demo/
├── src/
│   ├── environment.py          # 低轨卫星切换 gymnasium 环境
│   ├── dqn_agent.py            # Double DQN 智能体（网络 + 回放缓冲区）
│   ├── traditional_handover.py # 传统切换基线算法
│   └── utils.py                # 可视化与统计工具
├── tests/
│   ├── test_environment.py     # 环境单元测试（34 个测试）
│   └── test_agent.py           # 智能体单元测试
├── train.py                    # 训练脚本
├── evaluate.py                 # 评估脚本
├── requirements.txt
└── pytest.ini
```

---

## 运行测试

```bash
python -m pytest tests/ -v
```
