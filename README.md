# leo_handover_demo
低轨卫星切换策略研究

本仓库现包含一个最小可运行的 QMIX 低轨卫星切换示例：

- `src/environment.py`：离散时隙下的多用户-多卫星动态通信环境，包含可见卫星集合、链路速率、资源占用风险、切换代价与断连损失建模。
- `src/qmix.py`：集中训练/分布执行（CTDE）的 QMIX 实现，含局部 Q 网络、超网络驱动的混合网络与目标网络更新。
- `train.py`：简化训练入口，用于联调环境与 QMIX 策略。
- `tests/`：针对环境关键逻辑与 QMIX 训练/决策流程的单元测试。

## 运行测试
```bash
python -m unittest discover -s tests -q
```

## 运行示例训练
```bash
python train.py
```
