# Memory Affordance — 项目进度与路线图

> 最后更新：2026-04-07

## 初心（核心 Task）

让机器人能从**过去的视觉记忆**（几小时甚至几天前走过房间的视频流）中检索完成当前任务所需的工具/物体，并在该历史帧上做精细的 affordance mask 预测。

> **典型场景**：你在客厅说"帮我拿个能拧螺丝的东西"，电钻在车库（不在当前视野），机器人需要回忆起之前路过车库时见过电钻 → 找到那一帧 → 在那帧上做精细的 affordance mask 预测。

要做的三件事：**新 task → benchmark → baseline method**。论文 presentation 视 baseline 效果决定侧重 benchmark 还是 method。

## 任务定义

- **输入**：N 张图片组成的 memory episode（50–1000 帧）+ 一个自然语言 task instruction
- **输出**：图片索引（哪一帧）+ 该帧上正确工具的 mask
- **约束**：标准答案必须唯一（episode 内不能有其他物体也能完成同一任务）

## 整体路线图 & 当前位置

| # | 步骤 | 工具/脚本 | 状态 |
|---|------|----------|------|
| 1 | 选数据源 | Ego4D（ModelScope 镜像 `loongfei/Ego4D`）/ EPIC-KITCHENS | ✅ |
| 2 | 关键帧抽取 | `benchmark/scripts/extract_keyframes.py` | ✅ 已写 + 已跑通 |
| 3 | 自动 QA 生成 | `benchmark/scripts/episode_qa_generation.py` + Qwen 397B | ✅ 已跑通（P01_03 → 8 QA） |
| 4 | QA 质量校验（跨帧唯一性） | `benchmark/scripts/qa_uniqueness_check.py` + `prompts/episode_qa_check.md` | ✅ 已写 + 已跑通 |
| 5 | Mask 自动标注 | `benchmark/scripts/generate_masks.py` (Rex-Omni + SAM2) | ✅ 已写，⏳ 等 GPU 跑 |
| 6 | Mask 校验 + 人工抽查 | `Affordance_Annotator/check/mask_check.py` | ⏳ 待做 |
| 7 | Benchmark 打包 | `benchmark/scripts/build_benchmark.py` | ✅ 已写 |
| 8 | Baseline 实现 | （见下方"Baseline 设计"） | ⏳ 待做 |
| 9 | 评测 | `benchmark/scripts/evaluate.py` (Acc@1/Acc@5/Mask IoU) | ✅ 已写 |

## 已完成

### 代码框架（`benchmark/`）
```
benchmark/
├── prompts/
│   └── memory_affordance_qa.md      # 多帧 episode QA 生成 prompt
├── scripts/
│   ├── extract_keyframes.py         # 视频→关键帧（uniform/scene_change/combined）
│   ├── episode_qa_generation.py     # 调 VLM 生成 affordance QA
│   ├── build_benchmark.py           # 打包最终 benchmark 格式
│   └── evaluate.py                  # 评测（Acc@1/Acc@5/Mask IoU/Combined）
└── PROGRESS.md                      # 本文件
```

### 数据
- **EPIC-KITCHENS-100**: P01_01/02/03 已下载（共约 8GB），P01_03 已抽 50 关键帧
- **Ego4D（ModelScope `loongfei/Ego4D`）**:
  - `annotations.zip` (550MB) ✅ 已下载并解压（含 fho/vq/nlq/moments 等所有 benchmark）
  - `ego4d.json` (89MB) ✅ 已下载（含 9821 个视频元数据）
  - `Ego4D_info.pth` (1.5MB) ✅ 已下载
  - **`full_scale_val_0000.tar` (7.2GB) ⏳ 后台下载中** ← 验证集 + benchmark 视频
- 完整 Ego4D 共 153 个 train tar (~7.17 TB)，目前不下

### 端到端测试
**测试 1 — EPIC-KITCHENS P01_03**（2 分钟厨房，20 帧）
- QA 生成：8 个，28 秒
- 唯一性校验：5/8 valid (62.5%)，4 分 40 秒
- 拒绝都合理：trash bin / sink / granola 在相邻帧重复 → 暴露抽帧过密问题

**测试 2 — Ego4D 024713b7 cooking video**（17.7 分钟，60 帧抽样到 30 帧）
- QA 生成：8 个，77 秒，物体描述更精细
- 唯一性校验：**8/8 valid (100%)**，~7 分钟
- 证实 Ego4D 视频抽帧间隔 17.7s 时，跨帧唯一性问题自动消失
- 已升级 `extract_keyframes.py`：加入 perceptual hash 去重选项

### 工具与依赖
- `.venv/`：Python 虚拟环境（opencv-python-headless / openai / pillow / tqdm / numpy / requests / torch / modelscope / aria2c）
- **API**：DashScope `qwen3.5-397b-a17b`（OpenAI 兼容协议）
  - Endpoint: `https://dashscope.aliyuncs.com/compatible-mode/v1`
  - 注意：本地代理 `127.0.0.1:7890` 对部分服务有 SSL 问题，必要时 `--noproxy '*'`

## 下一步具体做什么（按优先级）

### 1. 等 Ego4D val tar 下完（约 1 小时，7.2GB）
拿到正经的 ego-view 数据，比 EPIC-KITCHENS 厨房场景多样性更好（含 Cooking 1353、Cleaning 1065、Indoor Navigation 260 等）。

### 2. 跑一个稍大规模的 QA 生成测试
- 在 Ego4D val 数据上跑 5–10 个 episode
- 关注：Qwen 397B 在更长 episode（50–100 帧）下是否稳定？QA 唯一性是否有问题？单 episode 时间和成本？

### 3. ~~加 QA 唯一性校验~~ ✅ 已完成
- `qa_uniqueness_check.py` 已实现 episode 级别校验
- 4 步检查：物体存在 → instruction 合理 → **跨帧唯一性** → 表达清晰
- 支持多模型 intersect 模式（paranoid）
- 实测 Qwen 397B 判断准确，拒绝理由具体（指出哪些其他 image_id 也能答）

### 3.5 关键帧抽取改进（新发现的问题）
- 当前 uniform interval=60 在 2 分钟视频上抽 50 帧，连续帧高度相似
- 改进方案：
  - (a) 降低 interval / max_frames
  - (b) 加 perceptual hash 或 CLIP embedding 去重
  - (c) 用 scene_change 策略代替 uniform

### 4. 跑通 Mask 标注（需 GPU）
- Rex-Omni + SAM2 在 CPU 上不现实
- 等服务器账号；本地先准备好脚本，将生成的 QA 喂给 `label-sam2.py`

### 5. 写 Baseline 方法
- **A. Naive VLM**: 一次性把所有帧塞给 VLM 选帧 → 接 affordance method 出 mask
- **B. Per-frame**: 对每一帧独立跑 affordance prediction
- **我们的 method**：**离线 memory 索引 + 在线两阶段检索**
  - 离线：每帧 Detect Everything → 每个物体存 embedding (DINO + CLIP) + VLM 生成的 affordance proposal 文本标签
  - 在线粗排：query 与 affordance proposal 做 text similarity，召回 Top-K 候选
  - 在线重排：让 MLLM 精读 Top-K 候选帧（远好于一次性读 1000 张）

## Benchmark 设计要点

- **Episode 大小**：50–1000 帧（依实验调整）
- **总规模目标**：100–1000 个 episode，每个 episode 几个 task instruction
- **难度分级**：
  - Easy: 几乎不需要 reasoning
  - Medium: 需要 affordance 推理
  - Hard: 组合/多步推理
- **唯一性约束**：标准答案必须唯一，需在 QA check 阶段强制
- **测试时建议**：图文交错输入，每张图片前加 ID 编号（如 `[Image 0]`）

## 关键决策记录

- **2026-04-06**：选择 video-first 路径而非 PAP-12K 全景拆分（视频 episode 更自然，PAP-12K 备用）
- **2026-04-06**：先用 EPIC-KITCHENS 原型验证（开放下载，无需审批）
- **2026-04-07**：发现 Ego4D 在 ModelScope 有 `loongfei/Ego4D` 镜像（字节内部镜像，无需 Meta 审批，国内速度更快）
- **2026-04-07**：决定先下 `full_scale_val_0000.tar` (7.2GB) 而非 50GB train tar — val 集正是 benchmark 视频
- **VLM 选型**：先试 Qwen3.5 397B (DashScope)；后续可对比 Gemini / Gemma 长上下文能力

## 待用户确认

1. **GPU 服务器账号** — Mask 自动标注需要 GPU（详见下方"GPU 需求"）
2. **Ego4D 官方申请** — https://ego4ddataset.com 是否已申请？（作为 ModelScope 镜像断流时的备份）
3. **数据规模目标** — 100 / 500 / 1000 episode？这影响 pipeline 自动化程度

## GPU 需求清单

| 步骤 | 模型 | GPU 需求 |
|---|---|---|
| Mask 自动标注 | Rex-Omni (7B) + SAM2 hiera-large | **必须**，单卡 ≥24GB 显存（A100/L40S/4090） |
| Mask 校验（可选） | Qwen3-VL-8B / InternVL3.5-14B via vLLM | **可选**，一张 24GB 即可（用 API 则不需要） |
| VLM QA 生成 / QA 校验 | Qwen3.5 397B | **不需要**，走 DashScope API |
| 抽帧 / 评测 / 数据准备 | — | **不需要**，纯 CPU |

**最少需求**：1 张 ≥24GB 卡即可启动 mask 标注流程。
**理想配置**：2 张卡，1 卡跑 Rex-Omni+SAM2 标注，1 卡跑 vLLM 部署校验模型。

## 参考资源

- **Affordance_Annotator**: https://github.com/zixinzhang02/Affordance_Annotator （现有 4 阶段标注 pipeline，可复用）
- **PAP-12K**: https://github.com/EnVision-Research/PAP （之前工作的全景 affordance 数据集）
- **Ego4D 官方**: https://ego4d-data.org/
- **EPIC-KITCHENS**: https://epic-kitchens.github.io/
- **ModelScope Ego4D 镜像**: https://modelscope.cn/datasets/loongfei/Ego4D
