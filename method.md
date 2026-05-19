# GetForYeast 流程速查

## 核心脚本

| 用途 | 脚本 | 说明 |
|------|------|------|
| KM 训练 | `train_yeast_single_peak_km.py` | KM 酵母，5 个 npz（C1/C3/O2/O3 + ATAC1），支持数据集加权采样 |
| SC 训练 | `train_yeast_single_peak_sc.py` | SC 酵母，单样本 `ATAC1.npz` |
| 推理 | `infer_yeast_single_peak.py` | 加载 `best_model.pth`，对新 npz 做 peak 级预测 |
| 模型定义 | `get_model/model/yeast_model.py` | `YeastModel`（Transformer，单 peak 输入） |

训练输出目录：`output/{km\|sc}_atac_single_peak_training_<时间戳>/`，内含 `best_model.pth`、`experiment_config.yaml`、指标 CSV 等。

---

## 配置文件（是否还在用）

| 文件 | 训练 | 推理 |
|------|------|------|
| `get_model/config/yeast_training_km.yaml` | ✅ 通过 Hydra 加载 | 一般不用 |
| `get_model/config/yeast_training_sc.yaml` | ✅ 通过 Hydra 加载 | 一般不用 |

- **训练**：必须用 yaml。Hydra 装饰器分别绑定 `config_name=yeast_training_km` / `yeast_training_sc`。
- **推理**：**默认不需要 yaml**。`best_model.pth` 里已保存完整 `config` + `model_state_dict`，推理脚本从 checkpoint 读模型结构。
- 推理脚本顶部 `CONFIG_PATH = None`；仅作备选，非主路径。

改训练超参、数据路径 → 改对应 yaml 后重新训练。改推理输入/模型路径 → 改 `infer_yeast_single_peak.py` 顶部常量或命令行参数。

---

## 当前使用的权重（best_model.pth）

### KM 模型（推荐用于 KM 四样本数据）

- **路径**：`output/km_atac_single_peak_training_20260210_110601/best_model.pth`
- **训练数据**：`input/260128SCKM/` 下 C1/C3/O2/O3 + ATAC1（见 `yeast_training_km.yaml`）
- **测试指标**（该次训练报告）：Pearson r ≈ 0.967
- **yaml 中 `inference.model_path`** 已指向该路径（作记录用，推理脚本不自动读 yaml）

### SC 模型（当前推理默认）

- **路径**：`output/sc_atac_single_peak_training_20251209_142840/best_model.pth`
- **训练数据**：`input/train_251208/ATAC1.npz`
- **验证指标**（checkpoint 内）：Pearson r ≈ 0.923，MAE ≈ 0.699
- **`infer_yeast_single_peak.py` 中 `MODEL_PATH`** 当前指向该 SC 权重

### 实际推理选用情况（日志）

- 多数推理（含最近 2026-05）：**SC** 权重 `...20251209_142840/best_model.pth`
- KM 四矩阵推理（2026-03-09）：**KM** 权重 `...20260210_110601/best_model.pth`

> 对新物种/新矩阵：KM 数据优先用 KM 权重，SC/通用 ATAC1 风格数据用 SC 权重；在推理脚本里改 `MODEL_PATH` 或 `--model_path`。

---

## 常用命令

```bash
# 在 GetForYeast 目录下

# KM 训练
python train_yeast_single_peak_km.py

# SC 训练
python train_yeast_single_peak_sc.py

# 推理（先改脚本顶部 DATA_PATHS / MODEL_PATH）
python infer_yeast_single_peak.py

# 或命令行覆盖
python infer_yeast_single_peak.py \
  --data_path input/260511/LEM3_31x15_from_matrix511.npz \
  --model_path output/sc_atac_single_peak_training_20251209_142840/best_model.pth
```

推理结果：`output/inference_<时间戳>/`，含 `predictions_*.csv` 与 `inference.log`。

---

## KM vs SC 训练脚本差异（简要）

- **km**：多数据集混合 + `dataset_weights` + `WeightedRandomSampler`；强制 5 个 input key。
- **sc**：仅 `atac1` 单文件；无加权采样。
- 二者架构相同（`YeastModel`），超参结构类似，主要差在数据与 yaml。
