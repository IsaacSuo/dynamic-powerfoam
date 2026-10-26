# 需求：去掉新 foam 里的碎片 cell，重新划分（2026-10-23）

## 一句话
新训的 Power Foam（round01_8k）里有一批体积几乎为 0 的材料 cell（碎片）。物理模拟把每个材料 cell 当作一个 body 做显式时间积分，这些碎片质量接近 0、刚度却不小，导致稳定时间步小了 27 倍。希望删掉这些点、重新划分 power diagram（必要时短暂微调），得到一份没有碎片、几何和渲染基本不变的新 foam。

## 为什么是个问题（物理侧）
- 物理模型：每个材料 cell = 自己的 power cell ∩ 自己的球；cell 之间按公共墙连 bond；body 质量 = cell 体积 × 密度。
- 显式积分（辛欧拉）稳定条件 dt < 2/ω_max，ω_max 是"刚度 / 质量"的最大频率。体积→0 的 cell，ω→∞。
- 实测（`free_surface/omega_max.py`，日志在对话里）：ω_max = 2.6e5 rad/s，100% 集中在一个 cell 上（model.pt 下标 7531，体积为中位数的 2e-6）。稳定 dt < 7.6e-6 s；模拟要用 1/4800 s（2.1e-4 s），差 27 倍。
- 原来的 DEM 是把小 cell 合并成 body（612 个合并 body）来绕开的；用户希望改成删点、重新划分，从几何上解决。

## 现在的数据
- run：`/home/fangsuo/py/dynamic-powerfoam/.claude/worktrees/thin-surface/output/pacnerf_elastic0_f0_round01_8k/model.pt`（7,999 个点；半径 = softplus(radii, beta=100)）
- DEM：`.../thin-surface/output/pacnerf_dem_round01_8k/elastic_0_zup/`（`cells.npz` 的 `source_index` = 7,722 个材料点在 model.pt 里的下标；z-up 帧只是 model.pt 帧的 y→z 旋转，无缩放）
- 精确几何（物理侧生成）：`.../talk-pages/output/round01_exact/dem_exact_single/cells.npz`，`masses` = 每个材料 cell 的精确体积，顺序与 `source_index` 一致。
- 体积分布（材料 cell，中位 3.145e-6 m³）：小于中位数的 1e-4 有 21 个；1e-3：56；1e-2：156；5%：322；10%：497。
  最小的 8 个（model.pt 下标）：4611、1364、7531、5410、462、7128、269、2549（体积/中位：0、1e-6、2e-6、3e-6、3e-6、6e-6、2.1e-5、2.3e-5）。

## 要做的
1. 找出碎片点。**不要拍一个体积比例阈值**（用户的规则：不设阈值、不加修正系数）。由验收标准驱动：按"最危险"的顺序（ω_max 最大的、或体积/邻居体积最小的）逐个删，删到满足验收为止。
   - 快速筛选的代理量（不需要物理代码）：每个材料 cell 的 ω_c ≈ √(μ/ρ · Σ_墙 A/L ÷ V_c)，A 为墙面积，L 为两个点的距离，μ/ρ = 8.846 m²/s²（E = 23008 Pa，ν = 0.3008，ρ = 1000）。稳定步长约为 2 / max ω_c。只用来排序和预估，验收以第 4 节的脚本为准。
2. 删除这些点后重新算 power diagram（材料 / 空气分类方式与 round01 相同）。如果删点让渲染或表面明显变差，在 Power Foam 训练里对剩下的点做短暂微调（位置 / 半径 / 颜色），但不要再长出新的碎片。
3. 空气点里的碎片要不要删由你判断：物理只用材料 cell，但空气点参与划分、决定材料 cell 的形状。原则是材料区域（材料球的并集）基本不变。
4. 不要覆盖 round01：新 run 和新 DEM 放新目录（例如 `..._round01_8k_pruned`），DEM 目录格式与现有 `elastic_0_zup` 相同（尤其 `cells.npz` 的 `source_index`）。

## 约束
- 坐标系、尺度、半径的含义、材料 / 空气的分类方式都与 round01 一致。
- 物体表面 = 材料球的并集（物理用的就是它）。删一个点，如果它的球有露在外面的部分，表面就会变，需要量化（见验收）。
- 渲染流程按 Power Foam 自己的方式，不改。

## 验收（物理侧的命令，在 talk-pages worktree 里跑）
环境：`PYTHONPATH=/home/fangsuo/py/dynamic-powerfoam:/home/fangsuo/py/dynamic-powerfoam/.claude/worktrees/talk-pages`，Python `/home/fangsuo/anaconda3/envs/om/bin/python`。
1. 生成精确几何和皮（约 1–2 分钟；FEM 约 11 分钟，可先跳过）：
   `python free_surface/foam_exact.py --run <新 model.pt 所在目录> --dem <新 DEM 目录> --out output/<新目录> --fem-h`
   （`--fem-h` 后面不跟数 = 不算 FEM）。看打印的材料体积：round01 是 0.035699 m³（真值 0.0316）。
2. 稳定步长：`python free_surface/omega_max.py --out output/<新目录>`
   **目标：稳定 dt ≥ 1/4800 s**（打印 "fine"），目标值由用户最终确认。同时报告最快模式在哪些 cell 上、最小体积分位数。
3. 几何变化：材料体积、露出的球面总面积（`foam_exact.py` 打印 exposed area；round01 是 0.69 m²）与 round01 的相对变化。
4. 渲染：训练 / 测试视图的 PSNR（或 Power Foam 自己的指标），新旧对比。
5. 报告删了哪些点（model.pt 下标）、删除顺序和每一步后的 ω_max。

## 交付
- 新 model.pt 目录、新 DEM 目录的路径；
- 第 4 节 1–5 项的数字；
- 删除 / 微调用到的代码改动说明（在哪个 worktree、哪些文件）。
