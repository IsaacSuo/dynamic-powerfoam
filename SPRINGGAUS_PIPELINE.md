# Spring-Gaus 真实数据流水线（Power Foam + DEM）

目标：用 Spring-Gaus 的真实抛掷数据（bun / burger / dog / pig / potato），先把物体重建成 Power Foam，再把 foam 格子当作力学积木（基本是 1 个格子 = 1 块积木），从 3 个相机的掩码视频中学习材料参数。

一键运行：

```bash
PY=/path/to/python ITERS=40 experiments/run_springgaus_pipeline.sh burger      # 从第 1 步开始
PY=/path/to/python experiments/run_springgaus_pipeline.sh burger 8             # 从第 8 步开始（前面的产物已带上）
```

已经存在的产物会自动跳过。想重跑某一步，删掉它的产物即可。日志都在 `output/springgaus_logs/<obj>/`。

## 环境（本机实测的版本）

Python 3.10，torch 2.9.1+cu128，warp-lang 1.10.0，numpy 2.2，scipy 1.15，opencv 4.12，pycolmap 3.12，numba 0.61。其余依赖见 `requirements.txt`（configargparse、fpsample、lpips、tensorboard、plyfile 等）。Warp 内核第一次运行时会现场编译，大约要等 1–2 分钟。

## 九个步骤

| 步 | 脚本 | 做什么 | 产物 |
|---|---|---|---|
| 1 | `springgaus_to_powerfoam.py` | 把静态照片转成 RGBA，分辨率降到 1/4。用视觉外壳自动找出**相机位姿错误的照片并丢掉**（burger 有 3 张：DSC09936/09990/09991），再把物体缩放到单位立方体内 | `data/springgaus_powerfoam/<obj>/` |
| 2 | `springgaus_dynamic.py prepare` | 3 个动态相机的掩码和图像去畸变，写出 `cameras.json`（单位为米，重力 +z，地面高度按 Spring-Gaus 的设定） | `data/springgaus_powerfoam/<obj>_dynamic/` |
| 3 | `train.py`（`configs/springgaus.yaml`） | 静态 Power Foam 重建：8k 个格子，**加间距约束**（权重 1e-2，κ=0.5） | `output/springgaus_<obj>_spacing8k/` |
| 4 | `springgaus_dynamic.py hull` / `register` | 严格视觉外壳采样，然后在第 0 帧做轮廓对齐（初值用 Spring-Gaus 配置里的 R/T/S） | `register_calib/frame00.json` |
| 5 | `springgaus_register_rgb.py --upright` | **颜色对齐**：汉堡几乎是圆的，光看轮廓分不出朝向。从 200 个随机朝向和"正立"朝向族（棋盘法向对齐重力，扫描绕竖轴的转角）出发，再做局部优化 | `register_calib/frame00_rgb.json` |
| 6 | `springgaus_initial_state.py --upright --no-spin` | **在落地前的帧上联合拟合**：姿态、尺度、初速度（角速度固定为 0，真实视频里不转），以及每台相机的**亚帧时间偏移**（物体每帧下落约 2.5 cm，三台相机只按整帧同步） | `register_calib/initial_state_upright_nospin.json` |
| 7 | `pacnerf_foam_geometry.py --min-fraction 0.1` | 建米制单位的 DEM。只把极小的碎片格子并到相邻格子里，99% 的积木都是单个格子 | `output/springgaus_dem/<obj>_spacing8k_mf0.1/` |
| 8 | `springgaus_train.py material` | 学 7 个共享的材料倍数。用 Warp rollout，blend 渲染，轮廓 L2 损失 | `.../material_shared/report.json` |
| 9 | `springgaus_train.py material --per-body` | 在第 8 步结果上，给每块积木学一个偏差，加平滑项 1e-2 | `.../material_perbody/report.json` |

**评估划分**（和 Spring-Gaus 一致）：C0733 留出不参与训练，用 C0787 和 C0801 训练。帧的划分是：1–14 帧训练，15–19 帧作为未来预测。`report.json` 里报四种 IoU：训练视角/留出视角 × 训练帧/未来帧。

注意：第 4–6 步估计初始状态时用了全部 3 个视角，包括留出的 C0733，但只用落地前的帧。这和 Spring-Gaus 用全部视角做对齐是一个思路。落地后的材料训练只用 2 个训练视角。

## 已经带上的产物（burger）

打包里带了 burger 第 1–7 步的全部产物，所以服务器上可以直接从第 8 步开始：
- `data/springgaus_powerfoam/burger`、`burger_dynamic`
- `output/springgaus_burger_spacing8k`（静态重建，留出视角物体区域 PSNR 32.6）
- `output/springgaus_dem/burger_spacing8k_mf0.1`（6,287 个格子 → 6,212 块积木，每帧 1,146 个子步）

例外：带上的 `initial_state_upright_nospin.json` 是用 3 万格子的旧重建（v2）拟合的，第 7 步的 DEM 也是基于它建的。如果想让对齐也用 8k 重建来做，删掉 `register_calib/frame00_rgb.json` 和 `initial_state_upright_nospin.json`，以及 DEM 目录，然后从第 5 步开始跑。

其他物体需要原始数据：把 `real_capture.zip` 解压到 `data/springgaus/real_capture/`，然后从第 1 步开始。

## 显存（这次换服务器的原因）

- 一次评估（不求梯度，19 帧 × 1,146 子步）在 12 GB 卡上大约 1 分钟。
- 训练要对 14 帧 × 1,146 ≈ 1.6 万个子步反向传播，Warp 要把每一步的中间状态都存下来。12 GB 不够（爆了几 GB，开始借用内存）。只训练 1 帧时一次迭代 3.3 秒，梯度这条路没问题。
- 粗略估计，完整训练需要的显存约为单帧的 14 倍，**建议 ≥ 24 GB**。不够的话：`--train-frames` 设小一些（只训到落地后几帧，比如 10），或者 `DOWNSAMPLE=3`。后者只减少渲染部分的显存。
- 可以做的优化（还没做）：第 1–6 帧是自由落体，和材料无关。可以先不求梯度地模拟到落地前，再从那一刻开始带梯度。这样大约能省下 40% 的模拟步数和显存。

## 本机已有的结果和观察（burger）

- 初始材料 E=2e4 Pa、ν=0.3（按密度 1000 计算；只有 E/ρ 有意义）、未训练时：训练视角 IoU，训练帧 0.77 / 未来帧 0.51；留出视角 0.82 / 0.41。未来帧（落地后）差距很大，说明默认材料不对，需要训练。
- 真实视频里汉堡落地时会被压成扁饼（第 8 帧），再弹回成竖长形（第 12 帧）。**非常软、变形非常大**，E 应该会往下走（子步数会随之减少，训练也会变快）。
- 汉堡底部在静态拍摄时没拍到，形状是靠视觉外壳估的。底部偏毛糙，而落地时最先接触地面的就是底部，可能影响接触的准确性。
