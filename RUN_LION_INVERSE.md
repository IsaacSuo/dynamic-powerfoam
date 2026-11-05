# 狮子材料反演（纯弹性，旧 DEM）：在服务器上跑

分支 `lion-inverse`（从 `worktree-dem-contact` 分出）。需要的数据都在仓库里，即 `inverse_data/lion/` 下的真值掩码、相机、坐标偏移和泡沫，不需要其它工作区，也不需要 Power Foam 渲染器。

## 一条命令跑完（无人值守）

```bash
cd /root/dynamic-powerfoam && git fetch origin && git checkout lion-inverse
conda activate lion
nohup bash run_lion_inverse.sh > lion_inverse.log 2>&1 &
tail -f lion_inverse.log          # 看进度，Ctrl-C 只停止查看
```

脚本按顺序执行以下各段：
- `env`：缺包时装上 `requirements-inverse.txt`，不动 torch。
- `check`：第 0 帧 IoU 和第 5–8 帧质心，和本机的数字逐项对比，对不上就停下并说明原因。
- `synth`：合成数据，并判定 PASS 或 FAIL。FAIL 不会中止，后面照常跑。
- `real`：真值视频。
- `eval`：拟合出的材料和真值材料各跑一次全 24 帧。
- `summary`：写 `out_inverse/summary.txt`，并打包 `lion_inverse_results.tgz`。最后只需要把这个包拷回来。

**中断后续跑**：重跑同一条命令即可。已经完成的段会自动跳过，LM 从最后一次被接受的点接着跑。

**11-06 之前跑过旧版的**：先删掉旧的合成数据输出再跑，`rm -rf out_inverse/synth out_inverse/synth.log`。旧版用 ν 作参数，不能续跑。check 段的结果还有效，会自动跳过。

**也可以单独跑某一段**：`bash run_lion_inverse.sh check`，其余段同理。

**可选的环境变量**：
- `PY`：python 路径，默认 `python`；
- `JOBS`：同时运行的进程数，默认按空闲显存计算，每个约 4.5 GB，最多 7 个；
- `OUT`：输出目录，默认 `out_inverse`。

下面各节说明脚本里每一步做的是什么。

## 跑的是什么

**正向模型**是旧 DEM，和 10-31 本机的 `dem_old1` 那次完全相同：
- 拟合皮，单向球冠，赫兹地面无摩擦；
- **没有自接触**，因为 Vid2Sim 的真值没有考虑接触；
- 平滑起点加信赖域跨步沿用，这两项只提速，不改变解；
- dt = 1/1200 s，容差为重量的 1e-3。

**场景**全部取自数据集，不参与反演：
- 泡沫：`fourier_lion_8k_round_clean`，7166 个格子；
- 初速度 (0, 0, −0.245) m/s，即 g·dt/2，dt 取 Vid2Sim 的 0.05 s；
- 重力 −z；
- 地面在泡沫坐标 z = −0.2，即世界坐标 −0.7。

**未知量**是 E、ν、β，求解时用 (ln μ, ln K/μ, ln β)：
- μ 是剪切模量，K/μ 是体积模量和剪切模量之比，β 是 Kelvin–Voigt 黏性时间。
- 为什么不直接用 ν：11-06 第一次合成运行时，第一步从 ν = 0.30 直接冲到了上限 0.49，而真值是 0.391。剪影对 ν 的响应非常非线性：K/μ 在 ν = 0.30 时是 2.2，0.391 时是 4.3，0.49 时是 50。
- 在起点线性化，看到的斜率很小，于是要求一大步。换成 ln K/μ 后响应接近线性。
- 每步每个参数最多变 0.3，第一步也一样。
- ν 限制在 [0, 0.49]；β 下限 0.05，这是狮子单向球冠需要的。
- 结果仍报告 E 和 ν。

**观测**：每个视角每一帧（20 fps，第 k 帧对应 t = k/20 s）先渲染剪影，再取剪影的重心、二阶宽度和 √面积，单位为像素。
- 我们的剪影是各格子球的并集。球心在 x，半径为 r0·det(F_B)^(1/3)。
- 真值是 Vid2Sim 的 alpha 掩码。
- 两边都减去各自第 0 帧的特征，以去掉静态形状偏差：第 0 帧我们的剪影比真值胖 1–4.5 px，IoU 为 0.92–0.98。

**协议**和 10-18 的 Vid2Sim 基准相同：
- 拟合第 1–15 帧；视角 5 留出，用其余 11 个视角；
- 报告留出视角，以及第 16–23 帧这些未来帧的 IoU；
- 报告 log10 E 误差和 ν 误差。真值为 E = 10648.6、ν = 0.391。

**求解**用 Levenberg–Marquardt。
- 雅可比用整段运行的中心差分，步长在 (ln μ, ln K/μ, ln β) 上分别是 0.01、0.05、0.01。每次迭代 6 次差分运行，各开一个进程同时跑，加上 1 次试探运行。
- 当预测变化低于正向本身的误差时停止。正向误差由容差 1e-3 和 2e-3 两次运行的差估计。
- 某次运行不收敛时，那一步按拒绝处理。

## 环境

已经按 `RUN_LION_SERVER.md` 装好 `lion` 环境的（Python 3.10、torch 2.9.1、warp 1.10），只要补三个包：

```bash
cd /root/dynamic-powerfoam
git fetch origin && git checkout lion-inverse
conda activate lion
pip install numba==0.61.2 pillow==12.0.0 PyYAML==6.0.3
export PYTHONPATH=$PWD
```

新装环境：

```bash
git clone -b lion-inverse https://github.com/IsaacSuo/dynamic-powerfoam.git && cd dynamic-powerfoam
conda create -n lion python=3.10 -y && conda activate lion
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128   # 按驱动选 CUDA 版本
pip install -r requirements-inverse.txt
export PYTHONPATH=$PWD
```

每次运行约占 4 GB 显存。`--jobs` 是同时运行的进程数，一次迭代最多需要 6 个。

## 先自检（几分钟）

```bash
python -u free_surface/lion_inverse.py check
python -u free_surface/lion_inverse.py run --theta 9.273184 0.3912951704721983 -2.995732 --frames 8 --out out/smoke.npz
```

- 第一条是第 0 帧的剪影对真值掩码。本机结果：各视角 IoU 在 0.918（V01）到 0.977（V05）之间，重心差平均约 1.9 px。数字应该一致。
- 第二条是真值参数下跑到第 8 帧（0.4 s）。本机结果：
  - 第 7 帧打印 `com z -0.1179 m`、`balls touching 908`；
  - 第 8 帧打印 `com z -0.1224 m`；
  - 建场景约 7 s，正向 58 s。
- 第一次运行要编译 Warp 核函数，需要几分钟。
- 日志里会有几行 `Warp UserWarning: ... k_closP ... enable_backward=False`，这是正常的，旧 DEM 本来就会打印。

## 第一步：合成数据（检验能不能找回已知答案）

```bash
mkdir -p out
nohup python -u free_surface/lion_inverse.py lm --obs synth --truth 9.273184 0.3912951704721983 -2.525729 \
  --start 10.371796 0.30 -1.897120 --frames 15 --jobs 7 --out out/synth > out/synth.log 2>&1 &
```

- 观测由我们自己的模型生成，参数是 E = 10648.6、ν = 0.391、β = 0.08。起点是 E×3、ν = 0.30、β = 0.15。
- 通过的标准：E 误差在几个百分点以内（log10 误差 < 0.02），ν 误差 < 0.01，β 误差在 10% 以内。
- 每行 `it ... accepted` 是一次被接受的迭代。

## 第二步：真值视频

```bash
nohup python -u free_surface/lion_inverse.py lm --obs real --start 10.371796 0.30 -2.302585 \
  --frames 15 --jobs 7 --out out/real > out/real.log 2>&1 &
```

起点 E×3、ν = 0.30、β = 0.10。结束后用最终 θ（`out/real/final_theta.json` 里的 `theta`）做评估：

```bash
python -u free_surface/lion_inverse.py eval --theta <lnE> <nu> <lnbeta> --out out/real_eval > out/real_eval.log 2>&1
```

评估跑完全部 24 帧，打印并写入 `eval.json`：
- E、ν 误差；
- 拟合视角第 1–15 帧的 IoU，留出视角的 IoU；
- 未来帧第 16–23 帧的 IoU，分全部视角和留出视角两项。

## 续跑

`lm` 加 `--resume`，从 `lm.json` 里最后一次被接受的 θ 继续。

## 需要拷回来的

- `out/synth.log`、`out/synth/lm.json`；
- `out/real.log`、`out/real/lm.json`、`out/real/final_theta.json`；
- `out/real_eval/eval.json`、`out/real_eval/eval_frames.npz`。

`runs/` 下面是每次运行的特征和日志，出问题时才需要。

## 用时（估计）

- 本机（RTX 5070 Ti，双精度慢）单次正向：第 1–15 帧约 3.5 分钟，24 帧约 6.5 分钟。触地后每帧约 20 s。
- 一次迭代 ≈ 1 轮并行的 6 次差分运行 + 1 次试探运行，在服务器上约等于 2 次正向的时间。
- 每个阶段约 10–15 次迭代。
