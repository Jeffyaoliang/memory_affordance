# GPU 资源申请说明

## 项目：Memory Affordance Benchmark & Method

**一句话**：做一个新的研究任务 + benchmark + baseline，让机器人能从过去的视觉记忆（几小时前路过的房间）中检索完成当前任务所需的工具/物体，并在该历史帧上做 affordance mask 预测。

> **典型场景**：用户跟机器人说"帮我拿个能拧螺丝的东西"，电钻在车库（不在当前视野），机器人需要回忆起之前路过车库时见过电钻 → 找到那一帧 → 在那帧上做精细的 mask 预测。

---

## 为什么需要 GPU

整条 pipeline 共 5 步，前 3 步**已用 CPU + API 跑通**，后 2 步必须 GPU：

```
[1] 视频抽关键帧            CPU              ✅ 已跑通
[2] VLM 自动生成 affordance QA  Qwen API     ✅ 已跑通 (28s/episode)
[3] 跨帧唯一性校验          Qwen API         ✅ 已跑通 (4min/episode)
[4] Rex-Omni + SAM2 自动标 mask  ⚠ 必须 GPU  ⏳ 阻塞中
[5] VLM 校验 mask 质量      可 GPU 可 API   ⏳ 阻塞中
```

第 4 步需要：
- **Rex-Omni**（7B 视觉模型）做物体检测 + 指点
- **SAM2 hiera-large** 做精细分割
- 这两个模型本身就要 ~20GB 显存，并且每个 episode 几十张图都要跑两遍

CPU 跑这两个模型不现实（一张图要分钟级），无法完成 100-1000 个 episode 的标注工作量。

---

## 已经完成的工作（截至 2026-04-07）

### 1. Pipeline 代码框架（5 个脚本，全部跑通）

```
benchmark/
├── prompts/
│   ├── memory_affordance_qa.md      # 多帧 episode QA 生成 prompt（4 种 phrasing 风格）
│   └── episode_qa_check.md          # 4 步唯一性校验 prompt
├── scripts/
│   ├── extract_keyframes.py         # 视频→关键帧 (uniform/scene_change/combined + phash 去重)
│   ├── episode_qa_generation.py     # VLM 多图输入生成 affordance QA
│   ├── qa_uniqueness_check.py       # 跨帧唯一性校验（支持多模型 intersect）
│   ├── build_benchmark.py           # 打包最终 benchmark 格式
│   └── evaluate.py                  # Acc@1 / Acc@5 / Mask IoU / Combined Score
└── PROGRESS.md                      # 项目进度文档
```

### 2. 数据准备（已下载）

| 数据集 | 规模 | 用途 | 状态 |
|---|---|---|---|
| **EPIC-KITCHENS-100 (P01_01/02/03)** | 8GB, 3 个厨房 ego-view 视频 | 原型验证 | ✅ |
| **Ego4D 完整 v2 标注** (`annotations.zip`) | 550MB, 6.4GB 解压 | 含 fho/vq/nlq/moments 等所有 benchmark | ✅ |
| **Ego4D `ego4d.json` 元数据** | 89MB, 9821 个视频信息 | 元数据查询 | ✅ |
| **Ego4D 验证集视频** (`full_scale_val_0000.tar`) | 7.2GB, 12 个真实视频 | 主要标注源 | ✅ |
| **Ego4D 完整训练集** | 7.17 TB（153 个 tar） | 后续大规模扩展 | 💤 暂不下 |

12 个 Ego4D val 视频涵盖丰富场景：**Cooking ×2**、Cleaning、Crafting、Construction、Carpenter、Farmer、Puzzle、Board games、Music、Talking、Workout。

### 3. 端到端测试结果（已跑通）

**测试数据**：EPIC-KITCHENS P01_03（2 分钟厨房 ego-view 视频）

| 步骤 | 输入 | 输出 | 耗时 | 模型 |
|---|---|---|---|---|
| 抽关键帧 | 7124 帧视频 | 50 张关键帧 | 5s | OpenCV |
| QA 生成 | 20 张图 | 8 个 affordance QA 对 | 28s | Qwen3.5-397B (API) |
| 唯一性校验 | 8 QA × 20 图 | 5 个有效 QA | 4min 40s | Qwen3.5-397B (API) |

**生成的 QA 示例**（已通过校验）：
- Image 2 / refrigerator → "I have a perishable item that needs to be kept cold. Which large appliance should I open to store it?"
- Image 8 / kitchen cabinet with pots → "I need to find a pot to boil water. Which storage unit contains the cookware?"
- Image 13 / electric stove → "I need to cook a meal using direct heat. Which appliance with four burners is available?"

**校验拒绝示例**（说明校验有效）：
- Image 5 / trash bin → 拒绝。Qwen 指出："Image 6 和 9 也有垃圾桶，无法唯一确定"
- Image 14 / kitchen sink → 拒绝。Qwen 指出："Image 15 也是同一个水槽"

校验逻辑严格、判断准确，正是我们需要的质量保障。

### 4. API 与基础设施

- **DashScope Qwen3.5-397B (`qwen3.5-397b-a17b`)**：已验证可用，OpenAI 兼容协议
- **Python 虚拟环境**：opencv / pillow / openai / modelscope / aria2c / torch 等依赖全部就绪
- **代码风格**：完全兼容现有 `Affordance_Annotator` 仓库，可无缝接入

---

## GPU 需求

### 最小可用配置

| 项 | 要求 |
|---|---|
| **GPU 数量** | **1 张** |
| **显存** | **≥24 GB**（A100 / L40S / 4090 / 6000 Ada 都行） |
| **CUDA** | 12.1 |
| **PyTorch** | 2.5（已知兼容） |
| **磁盘** | ≥50GB 用于模型权重 + 中间结果 |

### 理想配置

| 项 | 要求 |
|---|---|
| **GPU 数量** | **2 张** |
| **用途** | 1 卡跑 Rex-Omni+SAM2 标注，1 卡跑 vLLM 部署校验模型并行加速 |

### 软件栈

```bash
# 已确认可装的依赖
conda create -n a4-agent python=3.11
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install flash-attention==2.7.4
pip install git+https://github.com/IDEA-Research/Rex-Omni.git
pip install git+https://github.com/facebookresearch/sam2.git
huggingface-cli download facebook/sam2.1-hiera-large
huggingface-cli download IDEA-Research/Rex-Omni
```

---

## 时间预估（拿到 GPU 后）

| 任务 | 工作量估计 |
|---|---|
| 配环境（Rex-Omni + SAM2 + Flash Attention） | 0.5–1 天 |
| 跑通 100 个 episode 自动标注 | 1–2 天 |
| 人工抽查 + 修复 corner case | 2–3 天 |
| 扩到 500–1000 episode | 3–5 天 |
| **合计**：可发表 benchmark 数据 | **约 1–2 周** |

---

## 工作量证据

- 代码：`benchmark/` 目录共 9 个文件，`Affordance_Annotator/` 已通读
- 文档：`PROGRESS.md`（项目进度）、`GPU_REQUEST.md`（本文档）
- 数据：本地已存约 22GB（EPIC-KITCHENS 8GB + Ego4D val 7.2GB + 标注 6.4GB + 元数据）
- 验证：完整端到端测试已跑通 5/8 valid QA，校验逻辑工作正常

**结论**：所有不需要 GPU 的部分都已完成。Pipeline 后半段（mask 自动标注）是当前唯一阻塞点，需要 1 张 ≥24GB GPU 来推进。
