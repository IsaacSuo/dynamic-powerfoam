# 橡皮泥落地（k1）：在服务器上跑

分支 `plasticine-k1`。只需要本仓库，不需要 PAC-NeRF 的数据。

## 跑的是什么

- **模型**：k1，即 11-01 验收的橡皮泥落地，代码是 `free_surface/merged_ul.py`（MergedUL / MergedStep）。
  - 每步重新切分 power 图。
  - 格子能量和弹簧余量合在一起，弹簧刚度取 DEM 键的 1/6。
  - 隐式中点法。
  - 模型代码就是提交 d4180b6，没有改动。
- **地面**：y = −0.05。赫兹球加 Mindlin 黏着，摩擦系数 μ = 10，不加"拉住"。
- **材料与初速度**：PAC-NeRF 橡皮泥第 k 号场景的 E、ν、屈服应力、初速度，按单位密度（ρ = 1000）换算。
- **泡沫**：`foams/fourier8k_clean`，7690 个格子，h = 14.78 mm，就是第 0 号场景的形状。
- **其他场景的初始位置**：第 1–9 号场景的初始形状是第 0 号整体转动后的结果。转动和平移已经用 MPM 第 0 帧粒子算好，连同各场景的参数存在 `foams/fourier8k_clean/plasticine_scenes.json`。
- **时间**：dt = 1/1200 s，共 15 帧（每帧 1/24 s），750 步，t = 0 → 0.625 s。

和 11-01 的 k1 相比，只改了驱动脚本 `free_surface/merged_drop.py`：
- 加了 `--scene`；
- 加了 `--data`。没有 MPM 粒子时，每帧的 `FRAME` 行只给我们自己的形状，末尾标 `no MPM here`；
- 求解不收敛时退出码为 1。

## 环境

和 `RUN_LION_SERVER.md` 的 `lion` 环境相同，k1 当时就是用这套版本跑的：Python 3.10、torch 2.9.1 + cu128、warp-lang 1.10.0。已经装过的直接用。

```bash
git clone https://github.com/IsaacSuo/dynamic-powerfoam.git     # 已有仓库就 git fetch origin
cd dynamic-powerfoam
git checkout plasticine-k1
conda activate lion                                               # 没有就按下面两行建
# conda create -n lion python=3.10 -y && conda activate lion
# pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128 && pip install -r requirements-k1.txt
export PYTHONPATH=$PWD                                            # 每次开新终端都要设
```

第一次运行要编译 Warp 核函数，需要几分钟，之后缓存在 `~/.cache/warp`。

## 先自检（约 1 分钟）

```bash
python -u free_surface/merged_drop.py --foam foams/fourier8k_clean --save out/plasticine --tag smoke --steps 40
```

最后打印 `done`，退出码为 0。

## 第 0 号场景（先跑这个，和本机对照）

```bash
mkdir -p out/plasticine
tmux new-session -d -s k1 -c $PWD "env PYTHONPATH=$PWD python -u free_surface/merged_drop.py \
  --foam foams/fourier8k_clean --scene 0 --save out/plasticine --tag s0 > out/plasticine/s0.log 2>&1"
```

本机（RTX 5070 Ti Laptop，11-01）的结果：
- 用时 679 s；
- 最后一行 `FRAME 15 t 0.625: height 0.148; x width 0.753, z width 0.711`；
- 落地后每步 30 多次 Newton。

同样的设置跑两次可能有差别：GPU 上双精度原子加法的累加顺序不固定，撞击段对舍入很敏感。只要量级一致就对。

## 第 1–9 号场景

```bash
tmux new-session -d -s k1all -c $PWD "for k in 1 2 3 4 5 6 7 8 9; do env PYTHONPATH=$PWD python -u free_surface/merged_drop.py \
  --foam foams/fourier8k_clean --scene \$k --save out/plasticine --tag s\$k > out/plasticine/s\$k.log 2>&1; done"
```

- 这些场景用 k1 都**还没跑过**，只在本机做过冒烟测试（几步能正常启动）。
- 第 1、2、7、8 号的 E 是 1.6–5.6 MPa，比第 0 号（0.17 MPa）硬 10–30 倍。这几个可能明显更慢，也可能某一步不收敛。某个场景失败时退出码为 1，循环会继续跑下一个场景。
- 某一步不收敛时，日志末尾会有 `the solve did not converge`，同时存 `fail_s<k>.npz`。

## 输出

- `out/plasticine/s<k>.log`：
  - 每 20 步一行 `t ...`：能量各项、塑性功、接触数、每步 Newton 和乘积；
  - 每帧一行 `FRAME`；
  - 最后是 Newton 和乘积的统计，以及 `done`。
- `out/plasticine/frames_s<k>.npz`：每 10 步一帧，包括 x（中心点）、半径、L、ep、屈服标记、t，以及每帧的 FRAME 数据（rows）。运行中至少每分钟写一次。
- 没有续跑功能，中断了就从头重跑那个场景。

## 和 MPM 对比（拷回本机后）

服务器上没有 MPM 粒子。把 `frames_s<k>.npz` 拷回本机，在有 `data/pacnerf/simulation_data.zip` 的地方运行：

```bash
python free_surface/flow_spreads.py --foam foams/fourier8k_clean --scene <k> out/plasticine/frames_s<k>.npz --frames 6 9 12 15
```

`free_surface/flow_spreads.py` 从 `spring-moment` 分支拷来，读的是本机写死的 `data/pacnerf` 路径。

如果把 PAC-NeRF 的 `simulation_data.zip` 放到服务器的 `data/pacnerf/` 下，`FRAME` 行会直接带 MPM 对照和 Chamfer 距离，这时需要另外 `pip install plyfile`。
