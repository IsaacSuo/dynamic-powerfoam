# 狮子落地仿真：在服务器上跑

分支 `spring-moment`。只需要本仓库，不需要 fourier3d 或其它工作区。

## 跑的是什么

- 模型：`RingMoment`（`free_surface/ring_moment.py`）。弹簧从静止的切分生成后一直保留，不绑在墙上；每步重新切分（功率图 ∩ 球，解半径让每个格子体积等于 V0·e^L）；隐式中点法，局部–整体迭代加信赖域 CG。
- 材料：纯弹性，Vid2Sim 狮子的参数（E/ρ = 10.65，ν = 0.391），β = 0.05 的粘性。
- 接触：
  - 自接触：静止颜色加跳跃份额判断是否同一块材料，异材料的格子对由对数律柱子推开。
  - 地面：镜像接触，每个格子对自己的镜像，地面同时也是切分的边界。
  - 两种接触都按平均向量场（AVF）计算，接触项的能量严格守恒。
- 输入泡沫：`foams/fourier_lion_8k_round_minsp_only_clean/dem_exact_single/`，7460 个格子，h = 13.88 mm。
- 地面在 z = −0.2，初速度 (0, 0, −0.245) m/s，dt = 1/1200 s。

## 硬件

- NVIDIA GPU。整套计算都是双精度（float64）。数据中心卡（A100、H100、V100 等）的双精度是单精度的 1/2；消费卡（RTX 系列）只有 1/64。所以尽量租数据中心卡。双精度快的卡上实际能快多少，还没测过。
- 显存 8 GB 以上，内存 16 GB 以上，磁盘 10 GB 以上（每个存档约 0.4 GB）。

## 装环境

```bash
git clone -b spring-moment https://github.com/IsaacSuo/dynamic-powerfoam.git
cd dynamic-powerfoam
conda create -n lion python=3.10 -y
conda activate lion
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128   # 按服务器驱动选 CUDA 版本
pip install -r requirements-sim.txt
export PYTHONPATH=$PWD                                                          # 每次开新终端都要设
```

## 先自检（几分钟）

```bash
python free_surface/avf_contact_check.py
python free_surface/ring_yield_lion.py --foam foams/fourier_lion_8k_round_minsp_only_clean --save out/lion --tag smoke \
  --moment --true-blocks --colour 1.19 --jump --ground-cut --T 0.01 --every 6
```

- 第一条检查两种接触的核函数，各项误差应在 1e-7 以下（本机：势能对积分 1e-9，力对差分 4e-8，功的恒等式 1e-7）。
- 第二条跑 12 步自由落体，最后打印 `done`。
- 第一次运行要编译 Warp 核函数，需要几分钟，之后会缓存在 `~/.cache/warp`。

## 正式跑（t = 0 → 1.2 s）

```bash
mkdir -p out
nohup python -u free_surface/ring_yield_lion.py \
  --foam foams/fourier_lion_8k_round_minsp_only_clean --save out/lion --tag full \
  --moment --true-blocks --colour 1.19 --jump --ground-cut \
  --tol 1e-2 --fit-floor 1e-2 --T 1.2 --every 6 \
  --save-state 336 480 600 720 840 960 1080 1200 1320 1440 \
  > out/lion_full.log 2>&1 &
```

- `--tol 1e-2`：求解容差，默认是 1e-3。实测 1e-2 和 1e-3 跑 24 步后位置差的均方根约 1e-4 h，速度、质心、动能、能量平衡都一致；落地后快约 2.6 倍。
- `--fit-floor 1e-2`：拟合皮的梯度拟合在墙看不清的方向上垫底，那个方向改用格子自己弹簧的拟合（11-03 起是默认值，写出来只为清楚）。不加这一项时，泡沫里少数只靠一个平面挂着的表面格子会让拟合系数放大到上万倍，服务器那次从 t = 0.595 起每步十几万次乘积。
- `--every 6`：每 6 步（0.005 s）存一帧。
- `--save-state`：在这些步存完整状态，用于续跑。336 在第一次受地面推力之前，之后每 120 步（0.1 s）一个。

## 看进度和输出

- `out/lion_full.log`：每 6 步一行 `t ...`，内容有动能、各项能量、能量平衡（balance / start）、质心高度、最低点、每步起点和求解各用的时间。另外几行 `contact:`、`ground:`、`share:` 是接触统计。
- `out/lion/frames_full.npz`：每帧的 x（中心点）、F、L、t，运行中每分钟写一次。
- `out/lion/state_full_<步>.pt`：完整状态。
- 退出码：0 表示跑完；1 表示某一步求解不收敛，日志里会有 `did not converge`，同时存 `fail_full.npz`。

## 续跑

```bash
nohup python -u free_surface/ring_yield_lion.py \
  --foam foams/fourier_lion_8k_round_minsp_only_clean --save out/lion --tag full2 \
  --moment --true-blocks --colour 1.19 --jump --ground-cut \
  --tol 1e-2 --fit-floor 1e-2 --T 1.2 --every 6 --load-state out/lion/state_full_720.pt \
  --save-state 840 960 1080 1200 1320 1440 \
  > out/lion_full2.log 2>&1 &
```

续跑的帧文件从载入的那一步开始，前面多一帧 t = 0 的静止状态。

## 速度

- A800 实测（tol 1e-2，还没有 `--fit-floor` 时）：t = 0 到 0.57 共 440 s，落地后每步约 0.75 s；从 0.595 起拟合退化，每步要 80～130 s。
- 加了 `--fit-floor 1e-2` 后，在同一个 t = 0.6 状态上，原来 10.5 万次乘积的那一步变成 222 次。按 0.75 s 一步估计，A800 上跑完 1.2 s 约 20～30 分钟；这个数还没有整段实跑验证过。
- 本机 RTX 5070 Ti Laptop（加 `--fit-floor` 之前）：自由落体约 0.26 s 一步，冲击段（tol 1e-3）约 3～4 s 一步。
- 同一个设置跑两次，耗时可能差很多：GPU 上双精度原子加法的累加顺序不固定，冲击段对舍入很敏感，轨迹会分岔。

## 注意

- 长跑不要加 `--dense-from`：它每一步都把整个逐步数组重写一遍，越跑越慢。
- `--watch-ground`（每步检查中心点是否穿过地面）和 `--trace-solves`（每步打印求解过程）只用于诊断。
- 渲染需要 fourier3d 的模型，不在这个仓库里。把 `out/lion/frames_full.npz` 拷回本地，用 `free_surface/dem_gt_compare.py` 渲染。
