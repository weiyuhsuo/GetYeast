"""

酵母基因表达预测模型训练脚本 - 单Peak版本 (Single Peak Version)

主要特性:
1. 单Peak训练: 每次训练处理单个peak，不考虑peak之间的相关性
2. 独立peak处理: 每个peak独立预测表达量
3. 简单高效: 训练速度快，适合大规模数据集
4. 完整的训练流程: 包含训练、验证、测试的完整流程

单Peak训练优势:
- 训练速度快，内存占用小
- 每个peak独立训练，简单直接
- 适合数据量大的情况
- 易于并行化处理

使用方法:
- 在config.py中设置训练参数
- 运行脚本进行训练
- 训练完成后使用测试脚本进行评估
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from get_model.model.yeast_model import YeastModel
import logging
from pathlib import Path
import os
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR, ExponentialLR, SequentialLR, LinearLR
import signal
import sys
import atexit
import time
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, linregress, spearmanr
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import warnings
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

# =============================================================
# 工具函数：加载真实基因TPM CSV并聚合为 log2(TPM+1)
# =============================================================
def load_real_gene_tpm_csv(csv_path: str, reduce: str = 'mean', gene_col: str = 'Gene', logger: logging.Logger = None) -> dict:
    """加载真实基因TPM CSV文件，按行聚合为 log2(TPM+1)。

    参数:
      - csv_path: 真实TPM CSV路径
      - reduce: 对多列TPM的聚合方式: mean/median/max/min
      - gene_col: 基因ID列名
      - logger: 可选logger

    返回:
      - dict[str, float]: 基因ID -> log2(聚合TPM + 1)
    """
    lg = logger or logging.getLogger(__name__)
    if not csv_path or not os.path.exists(csv_path):
        lg.warning(f"真实TPM文件不存在: {csv_path}")
        return {}
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        lg.error(f"读取真实TPM CSV失败: {e}")
        return {}
    if gene_col not in df.columns:
        lg.error(f"真实TPM CSV缺少'{gene_col}'列，跳过加载")
        return {}
    value_cols = [c for c in df.columns if c != gene_col]
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df_valid = df.dropna(subset=value_cols, how='all').copy()
    if df_valid.empty:
        lg.warning("真实TPM CSV中没有有效的数值列，返回空映射")
        return {}
    reduce = (reduce or 'mean').lower()
    out: dict[str, float] = {}
    for _, row in df_valid.iterrows():
        vals = row[value_cols].dropna().astype(float).values
        if vals.size == 0:
            continue
        if reduce == 'median':
            tpm_val = float(np.median(vals))
        elif reduce == 'max':
            tpm_val = float(np.max(vals))
        elif reduce == 'min':
            tpm_val = float(np.min(vals))
        else:
            tpm_val = float(np.mean(vals))
        gene_id = str(row[gene_col])
        
        # 移除 Kmarxianus_ 前缀，统一命名格式
        if gene_id.startswith('Kmarxianus_'):
            gene_id = gene_id.replace('Kmarxianus_', '', 1)
        
        out[gene_id] = float(np.log2(tpm_val + 1.0))
    lg.info(f"真实TPM加载完成: {len(out)} 个基因 (reduce={reduce}, 值=log2(TPM+1))")
    return out
# ============================================================================
# 全局变量（用于优雅中断处理）
# ============================================================================
# 这些全局变量用于在收到中断信号（Ctrl+C）时，能够保存当前训练的模型
# 并执行测试集评估，而不是直接退出导致训练成果丢失
interrupted = False  # 是否收到中断信号
training_completed = False  # 标记训练是否正常完成（非中断）
current_model = None  # 当前训练的模型（用于中断时测试）
current_test_loader = None  # 当前测试集数据加载器
current_device = None  # 当前使用的设备（CPU/GPU）
current_logger = None  # 当前日志记录器
current_output_dir = None  # 当前输出目录
current_peak_ids = None  # 保存用于CSV导出的peak_id列表


def signal_handler(signum, frame):
    """处理中断信号（SIGINT/SIGTERM）。"""
    global interrupted, current_logger
    interrupted = True
    if current_logger:
        current_logger.warning(f"收到中断信号 {signum}，准备保存模型并执行测试...")
    else:
        print(f"收到中断信号 {signum}，准备保存模型并执行测试...")

def cleanup_and_test():
    """清理并执行测试：在被中断时做一次快速测试并存盘。"""
    global current_model, current_test_loader, current_device, current_logger, current_output_dir, training_completed
    # 如果训练已经正常完成，不执行中断测试
    if training_completed:
        if current_logger:
            current_logger.info("训练已正常完成，跳过中断测试")
        return
    
    if current_model is not None and current_test_loader is not None and current_output_dir is not None:
        try:
            if current_logger:
                current_logger.info("开始执行中断后的测试...")
            # 确保模型在评估模式
            current_model.eval()
            
            # 测试集评估（按原文：正负链拼接后计算指标）
            test_eval = evaluate_model_with_meta(current_model, current_test_loader, current_device, current_logger, "test")
            test_loss, test_mae, test_slope, test_intercept, test_preds, test_targets, test_p, \
                test_peak_indices, test_sample_indices, test_strands = test_eval
            
            # 重新计算需要的统计量（仅测试集整体的相关性和回归，不做TPM域聚合，这部分留在正式评估函数中）
            from scipy.stats import spearmanr as _spearmanr
            test_spearman, _ = _spearmanr(test_targets, test_preds)
            test_r2 = r2_score(test_targets, test_preds)
            
            # 保存预测结果
            try:
                # 优先带peak_id和sample_id
                min_len = min(len(test_preds), len(test_peak_indices), len(test_sample_indices), len(test_strands))
                preds_np = np.array(test_preds)[:min_len]
                trues_np = np.array(test_targets)[:min_len]
                peak_idx_np = np.array(test_peak_indices)[:min_len]
                sample_idx_np = np.array(test_sample_indices)[:min_len]
                strand_np = np.array(test_strands)[:min_len]
                
                # 映射peak_id
                if current_peak_ids is not None:
                    peak_id_vals = [str(current_peak_ids[int(i)]) if int(i) < len(current_peak_ids) else '' for i in peak_idx_np]
                else:
                    peak_id_vals = [''] * min_len
                
                # 尝试获取sample_ids（从数据加载器）
                sample_id_vals = [str(i) for i in sample_idx_np]  # 默认使用sample_idx
                try:
                    if current_test_loader is not None:
                        dataset = current_test_loader.dataset
                        # 尝试获取底层数据集的sample_ids
                        base_ds = dataset
                        while hasattr(base_ds, 'dataset') or hasattr(base_ds, 'base'):
                            if hasattr(base_ds, 'base'):
                                base_ds = base_ds.base
                            elif hasattr(base_ds, 'dataset'):
                                base_ds = base_ds.dataset
                            else:
                                break
                        if hasattr(base_ds, 'sample_ids') and base_ds.sample_ids is not None:
                            sample_id_vals = [str(base_ds.sample_ids[int(i)]) if int(i) < len(base_ds.sample_ids) else str(i) for i in sample_idx_np]
                except Exception:
                    pass  # 如果获取失败，使用默认的sample_idx
                
                strand_str = np.where(strand_np == 0, 'pos', 'neg')
                df_test = pd.DataFrame({
                    'sample_idx': sample_idx_np,
                    'sample_id': sample_id_vals,
                    'peak_idx': peak_idx_np,
                    'peak_id': peak_id_vals,
                    'strand': strand_str,
                    'true': trues_np,
                    'pred': preds_np,
                    'error': preds_np - trues_np,
                    'abs_error': np.abs(preds_np - trues_np),
                })
            except Exception:
                df_test = pd.DataFrame({'pred': test_preds, 'true': test_targets, 'split': 'test'})
            df_test.to_csv(current_output_dir / 'interrupted_test_predictions.csv', index=False)
            
            # 生成测试集散点图（混合：concat pos/neg）
            test_mse = mean_squared_error(test_targets, test_preds)
            plt.figure(figsize=(10, 8))
            plt.scatter(test_targets, test_preds, alpha=0.01, s=0.1, c='green')
            plt.plot([0, 20], [0, 20], 'r--', linewidth=2, label='y=x')
            
            plt.xlabel('True expression (log2)')
            plt.ylabel('Predicted expression (log2)')
            plt.title('Test set - interrupted training evaluation (concat pos/neg)')
            plt.xlim(0, 20)
            plt.ylim(0, 20)
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
            
            # 增加Pearson r为首要展示
            from scipy.stats import pearsonr as _pearsonr
            _pr, _ = _pearsonr(test_targets, test_preds)
            test_stats_text = f'Test Set (Single Peak):\nPearson r = {_pr:.4f}\nSpearman ρ = {test_spearman:.4f}\nR² = {test_r2:.4f}\nMAE = {test_mae:.4f}\nMSE = {test_mse:.4f}\nN = {len(test_targets):,}'
            plt.text(0.02, 0.98, test_stats_text, transform=plt.gca().transAxes, 
                    fontsize=12, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(current_output_dir / 'interrupted_test_evaluation.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # 生成中断测试报告
            report_path = current_output_dir / 'interrupted_test_report.md'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"# 中断训练测试报告 (单Peak版本)\n\n")
                f.write(f"**测试时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**测试原因**: 训练被中断\n")
                f.write(f"**训练方法**: 单Peak训练\n\n")
                f.write(f"## 测试集结果\n")
                # 这里补充Pearson r为首要指标
                from scipy.stats import pearsonr as __pearsonr
                __pr, __ = __pearsonr(test_targets, test_preds)
                f.write(f"- **测试集Pearson r**: {__pr:.6f}\n")
                f.write(f"- **测试集Spearman ρ**: {test_spearman:.6f}\n")
                f.write(f"- **测试集R²**: {test_r2:.6f}\n")
                f.write(f"- **测试集MAE**: {test_mae:.6f}\n")
                f.write(f"- **测试集MSE**: {test_mse:.6f}\n")
                f.write(f"- **测试集样本数**: {len(test_targets):,}\n\n")
            
            if current_logger:
                current_logger.info(f"中断测试完成！结果已保存到: {current_output_dir}")
                current_logger.info(f"测试集结果: 损失={test_loss:.6f}, MAE={test_mae:.6f}, Spearman ρ={test_spearman:.6f}, 样本数={len(test_targets):,}")
            else:
                print(f"中断测试完成！测试集结果: 损失={test_loss:.6f}, MAE={test_mae:.6f}, Spearman ρ={test_spearman:.6f}, 样本数={len(test_targets):,}")
                
        except Exception as e:
            if current_logger:
                current_logger.error(f"中断测试失败: {e}")
            else:
                print(f"中断测试失败: {e}")

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup_and_test)

# ========== 配置将从YAML文件加载 ==========
# 所有配置参数现在都从 get_model/config/yeast_training.yaml 加载
# 包括数据路径、实验配置、训练参数等

# ============================================================================
# 数据集类：单Peak数据集
# ============================================================================
class YeastPeakSingleDataset(Dataset):
    """
    单Peak数据集 - 每个peak独立训练
    
    核心思想：
    - 每个peak作为独立的训练样本，不考虑peak之间的相关性
    - 输入特征：motif (470维) + accessibility (1维) + condition (74维) = 545维
    - 输出标签：正链表达 (1个值) + 负链表达 (1个值) = 2个值
    
    数据格式（KM酵母）：
    - 原始数据 shape: (num_samples, num_peaks, 547)
    - 前545列：输入特征
    - 列545：正链表达量 (label_pos_idx)
    - 列546：负链表达量 (label_neg_idx)
    
    Args:
        data_path: .npz数据文件路径
    """
    def __init__(self, data_path: str):
        """
        初始化数据集
        
        Args:
            data_path: .npz格式数据文件路径
        """
        # 以内存映射方式加载数据文件（避免一次性加载到内存）
        # mmap_mode='r' 表示只读模式，节省内存
        npz_file = np.load(data_path, mmap_mode='r', allow_pickle=True)
        
        # 数据格式: (samples, peaks, features)
        # 例如：(100, 5000, 360) 表示100个样本，每个样本5000个peaks，每个peak 360个特征
        self.data = npz_file['data']
        self.num_samples, self.num_peaks, self.num_features_all = self.data.shape
        
        # 读取peak_ids（用于CSV结果导出）
        try:
            self.peak_ids = npz_file['peak_ids']
            # 兼容object数组
            if hasattr(self.peak_ids, 'tolist'):
                self.peak_ids = self.peak_ids.tolist()
        except Exception:
            self.peak_ids = None
        
        # 读取sample_ids（用于CSV结果导出）
        try:
            self.sample_ids = npz_file['sample_ids']
            # 兼容object数组
            if hasattr(self.sample_ids, 'tolist'):
                self.sample_ids = self.sample_ids.tolist()
        except Exception:
            # 如果没有sample_ids，使用顺序编号
            self.sample_ids = [str(i) for i in range(self.num_samples)]
        
        logging.info(f"加载训练数据: {data_path}")
        logging.info(f"训练模式: 单Peak训练（每个peak独立处理）")
        
        num_samples, num_peaks, num_features = self.data.shape
        
        # ========== 特征维度定义 ==========
        # 每个peak的输入特征由三部分组成：
        # KM酵母特征维度定义
        # 1. Motif特征（0-469列）：470个TF motif强度
        # 2. Accessibility特征（470列）：染色质可及性
        # 3. Condition特征（471-544列）：74个实验条件特征
        self.motif_dim = 470  # Motif特征维度
        self.accessibility_dim = 1  # 可及性特征维度
        self.condition_dim = 74  # 条件特征维度
        
        # 标签在数据末尾：列545=正链表达，列546=负链表达
        # 数据布局: [0:470)=motif, [470]=accessibility, [471:545)=condition, [545]=expr_pos_log2, [546]=expr_neg_log2
        
        # 验证数据格式是否正确（KM酵母）
        if num_features == 547:
            # 正确！特征545=pos, 546=neg
            self.feature_dim = 545  # 前545列是输入特征
            self.label_pos_idx = 545
            self.label_neg_idx = 546
            logging.info(f"✅ 数据格式正确: 547列 = 545特征 + 2标签(pos@545, neg@546)")
        else:
            logging.error(f"❌ 数据维度错误！期望547列（545特征+2标签），实际={num_features}")
            raise ValueError(f"数据维度不匹配: 期望547，实际{num_features}")
        
        logging.info(f"特征维度: motif={self.motif_dim}, accessibility={self.accessibility_dim}, "
                    f"condition={self.condition_dim}, 总计={self.feature_dim}")
        logging.info(f"标签位置: pos={self.label_pos_idx}, neg={self.label_neg_idx}")
        
        # 生成所有有效的peak索引（正负链标签都不是NaN或inf）
        self.valid_indices = []
        for sample_idx in range(num_samples):
            for peak_idx in range(num_peaks):
                # 从data数组的最后两列读取标签
                label_pos = self.data[sample_idx, peak_idx, self.label_pos_idx]
                label_neg = self.data[sample_idx, peak_idx, self.label_neg_idx]
                if not (np.isnan(label_pos) or np.isinf(label_pos) or 
                        np.isnan(label_neg) or np.isinf(label_neg)):
                    self.valid_indices.append((sample_idx, peak_idx))
        
        self.total = len(self.valid_indices)
        
        logging.info(f"训练数据集: 样本={num_samples}, peaks/样本={num_peaks}, "
                    f"特征数={num_features}, 有效peaks总数={self.total:,}")

        # ========== 加载 Gene-to-Peak 映射矩阵 ==========
        # BuildNumpy 输出同时包含历史命名 p2g_* 与直观命名 g2p_*（内容相同）。
        # 这里做兼容处理：优先使用 g2p_*，若缺失则回退 p2g_*。
        # 另外：对每行权重做归一化检查，必要时自动重新归一化，
        # 确保后续 gene-level 聚合时不被异常放大/缩小。

        def _load_or_none(key: str):
            try:
                return npz_file[key]
            except Exception:
                return None

        def _choose(prefix: str, fallback_prefix: str):
            """选择一套 g2p 或 p2g 组件，返回 dict。"""
            def _first_non_none(a, b):
                return a if a is not None else b
            indices = _first_non_none(_load_or_none(f"{prefix}_indices"), _load_or_none(f"{fallback_prefix}_indices"))
            indptr = _first_non_none(_load_or_none(f"{prefix}_indptr"), _load_or_none(f"{fallback_prefix}_indptr"))
            data = _first_non_none(_load_or_none(f"{prefix}_data"), _load_or_none(f"{fallback_prefix}_data"))
            shape = _first_non_none(_load_or_none(f"{prefix}_shape"), _load_or_none(f"{fallback_prefix}_shape"))
            gene_ids = _first_non_none(_load_or_none(f"{prefix}_gene_ids"), _load_or_none(f"{fallback_prefix}_gene_ids"))
            shape = tuple(shape) if shape is not None else None
            return {
                'indices': indices,
                'indptr': indptr,
                'data': data,
                'shape': shape,
                'gene_ids': gene_ids
            }

        self.g2p_pos = _choose('g2p_pos', 'p2g_pos')
        self.g2p_neg = _choose('g2p_neg', 'p2g_neg')

        def _standardize_gene_ids(gene_ids_array):
            if gene_ids_array is None:
                return None
            out = []
            for g in gene_ids_array:
                if isinstance(g, bytes):
                    try:
                        out.append(g.decode('utf-8'))
                    except Exception:
                        out.append(str(g))
                else:
                    out.append(str(g))
            return np.array(out, dtype=object)

        self.g2p_pos['gene_ids'] = _standardize_gene_ids(self.g2p_pos['gene_ids'])
        self.g2p_neg['gene_ids'] = _standardize_gene_ids(self.g2p_neg['gene_ids'])

        def _to_csr(mapping: dict):
            if mapping is None:
                return None
            comps = (mapping.get('indices'), mapping.get('indptr'), mapping.get('data'), mapping.get('shape'))
            # 注意：不能用 `None in comps`，因为 numpy.ndarray 的比较会返回数组，导致布尔值歧义
            if any(x is None for x in comps):
                return None
            try:
                csr = csr_matrix((mapping['data'], mapping['indices'], mapping['indptr']), shape=mapping['shape'])
            except Exception as e:
                logging.warning(f"构建CSR失败: {e}")
                return None
            # 归一化检查与修正
            try:
                row_sums = np.array(csr.sum(axis=1)).ravel()
                need_fix = (row_sums > 0) & (np.abs(row_sums - 1.0) > 1e-3)
                if np.any(need_fix):
                    data = csr.data
                    indptr = csr.indptr
                    for r in np.where(need_fix)[0]:
                        s, e = indptr[r], indptr[r+1]
                        total = row_sums[r]
                        if total > 0:
                            data[s:e] /= total
                    csr = csr_matrix((data, csr.indices, indptr), shape=csr.shape)
                    logging.info(f"已重新归一化 {np.sum(need_fix)} 行 gene→peak 权重 (|sum-1|>1e-3)")
            except Exception as e:
                logging.warning(f"归一化检查失败: {e}")
            return csr

        self.g2p_pos_csr = _to_csr(self.g2p_pos)
        self.g2p_neg_csr = _to_csr(self.g2p_neg)

        if self.g2p_pos_csr is not None and self.g2p_neg_csr is not None:
            logging.info(
                f"已加载 gene→peak 映射: pos_genes={self.g2p_pos_csr.shape[0]}, neg_genes={self.g2p_neg_csr.shape[0]}, peaks={self.g2p_pos_csr.shape[1]}"
            )
            # 简单统计偏差（归一化后应接近1）
            pos_row_sums = np.array(self.g2p_pos_csr.sum(axis=1)).ravel()
            neg_row_sums = np.array(self.g2p_neg_csr.sum(axis=1)).ravel()
            logging.info(
                f"权重行和偏差: pos(mean|sum-1|={np.mean(np.abs(pos_row_sums-1)):.2e}), neg(mean|sum-1|={np.mean(np.abs(neg_row_sums-1)):.2e})"
            )
        else:
            logging.warning("未发现可用的 gene→peak 权重 (g2p_/p2g_)，将跳过基因级评估。")

    def has_gene_mapping(self) -> bool:
        return self.g2p_pos_csr is not None and self.g2p_neg_csr is not None

    def aggregate_gene_expression(self, peak_pred_pos: np.ndarray, peak_pred_neg: np.ndarray):
        """将peak级预测聚合为基因级预测。
        参数:
          peak_pred_pos/neg: 形状 (n_peaks,) 或 (batch, n_peaks)
        返回:
          dict: { 'pos': np.ndarray(n_pos_genes,), 'neg': np.ndarray(n_neg_genes,), 'gene_ids_pos': [...], 'gene_ids_neg': [...] }
        """
        if not self.has_gene_mapping():
            raise RuntimeError("gene→peak 映射缺失，无法聚合")
        # 支持批次：若二维仅取第一维（当前单peak训练评估阶段一般是展开后再聚合）
        if peak_pred_pos.ndim == 2:
            peak_pred_pos = peak_pred_pos.mean(axis=0)
        if peak_pred_neg.ndim == 2:
            peak_pred_neg = peak_pred_neg.mean(axis=0)
        pos_vals = self.g2p_pos_csr @ peak_pred_pos
        neg_vals = self.g2p_neg_csr @ peak_pred_neg
        return {
            'pos': np.asarray(pos_vals).ravel(),
            'neg': np.asarray(neg_vals).ravel(),
            'gene_ids_pos': self.g2p_pos.get('gene_ids'),
            'gene_ids_neg': self.g2p_neg.get('gene_ids')
        }

    def get_item_by_pair(self, sample_idx: int, peak_idx: int, dataset_idx: int | None = None):
        # 获取单个peak的所有数据（总列=547；前545特征 + 2标签）
        peak_data = self.data[sample_idx, peak_idx]
        # 特征与标签
        features = peak_data[:self.feature_dim]
        label_pos = peak_data[self.label_pos_idx]
        label_neg = peak_data[self.label_neg_idx]
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        labels_tensor = torch.tensor([label_pos, label_neg], dtype=torch.float32).unsqueeze(0)
        return {
            'motif_features': features_tensor,
            'labels': labels_tensor,
            'sample_idx': sample_idx,
            'peak_idx': peak_idx,
            'dataset_idx': 0 if dataset_idx is None else int(dataset_idx)
        }

    def __len__(self):
        return self.total

    def __getitem__(self, idx: int):
        # 获取peak的样本索引和peak索引
        sample_idx, peak_idx = self.valid_indices[idx]
        return self.get_item_by_pair(sample_idx, peak_idx)

class FilteredBySamplesDataset(Dataset):
    """按样本子集过滤后的单Peak数据集视图（用于样本级划分以支持gene级聚合）。"""
    def __init__(self, base_dataset: YeastPeakSingleDataset, allowed_samples: list):
        self.base = base_dataset
        self.allowed_samples = set(allowed_samples)
        self.valid_indices = []
        for p in self.base.valid_indices:
            if len(p) == 3:
                _, sidx, _ = p
            else:
                sidx, _ = p
            if sidx in self.allowed_samples:
                self.valid_indices.append(p)
        self.num_peaks = self.base.data.shape[1]
        # 暴露聚合相关
        self.g2p_pos_csr = self.base.g2p_pos_csr
        self.g2p_neg_csr = self.base.g2p_neg_csr
        self.g2p_pos = self.base.g2p_pos
        self.g2p_neg = self.base.g2p_neg
        self.parent_samples = sorted(list(self.allowed_samples))
        self.peak_ids = self.base.peak_ids
        self.sample_ids = getattr(self.base, 'sample_ids', None)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        key = self.valid_indices[idx]
        if len(key) == 3:
            ds_idx, sidx, pidx = key
            return self.base.get_item_by_pair(sidx, pidx, ds_idx)
        sidx, pidx = key
        return self.base.get_item_by_pair(sidx, pidx)


class FilteredByIndicesDataset(Dataset):
    """按indices列表过滤的数据集（用于Peak级划分）"""
    def __init__(self, base_dataset, valid_indices: list):
        self.base = base_dataset
        self.valid_indices = valid_indices
        
        # 提取所有涉及的samples（用于gene-level评估）
        all_samples = set()
        for idx in valid_indices:
            if len(idx) == 3:
                _, sidx, _ = idx
            else:
                sidx, _ = idx
            all_samples.add(sidx)
        self.parent_samples = sorted(list(all_samples))
        
        # 暴露必要属性
        if hasattr(base_dataset, 'g2p_pos_csr'):
            self.g2p_pos_csr = base_dataset.g2p_pos_csr
            self.g2p_neg_csr = base_dataset.g2p_neg_csr
            self.g2p_pos = base_dataset.g2p_pos
            self.g2p_neg = base_dataset.g2p_neg
        
        if hasattr(base_dataset, 'peak_ids'):
            self.peak_ids = base_dataset.peak_ids
        
        if hasattr(base_dataset, 'sample_ids'):
            self.sample_ids = base_dataset.sample_ids

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        key = self.valid_indices[idx]
        if len(key) == 3:
            ds_idx, sidx, pidx = key
            return self.base.get_item_by_pair(sidx, pidx, ds_idx)
        sidx, pidx = key
        return self.base.get_item_by_pair(sidx, pidx)

class MultiSinglePeakDataset(Dataset):
    """多数据集合并类 - 单Peak版本"""
    
    def __init__(self, data_paths: list):
        self.datasets = []
        self.dataset_sizes = []
        self.total_size = 0
        
        for path in data_paths:
            dataset = YeastPeakSingleDataset(path)
            self.datasets.append(dataset)
            self.dataset_sizes.append(len(dataset))
            self.total_size += len(dataset)
        
        # 添加num_samples属性（使用第一个数据集的样本数）
        # 假设所有数据集的样本数相同
        self.num_samples = self.datasets[0].num_samples if self.datasets else 0
        
        # 合并所有数据集的valid_indices
        # valid_indices格式(多数据集): [(dataset_idx, sample_idx, peak_idx), ...]
        # 同时创建映射：(sample_idx, peak_idx) → dataset_idx (仅兼容旧逻辑)
        self.valid_indices = []
        self.pair_to_dataset = {}  # 映射：(sample_idx, peak_idx) → dataset_idx
        
        for dataset_idx, dataset in enumerate(self.datasets):
            for valid_pair in dataset.valid_indices:
                sidx, pidx = valid_pair
                self.valid_indices.append((dataset_idx, sidx, pidx))
                # 记录这个(sample_idx, peak_idx)属于哪个数据集（可能覆盖）
                self.pair_to_dataset[(sidx, pidx)] = dataset_idx
        
        # 暴露第一个数据集的属性（用于FilteredBySamplesDataset访问）
        # 注意：这里假设所有数据集具有相同的结构，只是peaks不同
        if self.datasets:
            first_ds = self.datasets[0]
            self.data = first_ds.data  # 保留第一个数据集的data引用（用于获取shape等）
            self.g2p_pos_csr = first_ds.g2p_pos_csr
            self.g2p_neg_csr = first_ds.g2p_neg_csr
            self.g2p_pos = first_ds.g2p_pos
            self.g2p_neg = first_ds.g2p_neg
            self.peak_ids = first_ds.peak_ids  # 第一个数据集的peak_ids
            self.sample_ids = first_ds.sample_ids  # 第一个数据集的sample_ids（假设所有数据集样本数相同）
            
            # 计算所有数据集中最大的peak数（用于gene-level评估时分配足够大的数组）
            self.max_num_peaks = max([ds.data.shape[1] for ds in self.datasets])
            logging.info(f"  各数据集peaks数: {[ds.data.shape[1] for ds in self.datasets]}")
            logging.info(f"  最大peaks数: {self.max_num_peaks}")
            
        logging.info(f"多数据集单Peak加载完成，总Peak数: {self.total_size:,}")
        for i, (path, size) in enumerate(zip(data_paths, self.dataset_sizes)):
            logging.info(f"  数据集 {i+1}: {os.path.basename(path)} - {size:,} 个有效peaks")

    def __len__(self):
        return len(self.valid_indices)

    def get_item_by_pair(self, sample_idx: int, peak_idx: int, dataset_idx: int | None = None):
        """根据(sample_idx, peak_idx)获取数据
        
        由于多数据集合并，需要找到这个pair属于哪个数据集
        """
        if dataset_idx is not None:
            return self.datasets[int(dataset_idx)].get_item_by_pair(sample_idx, peak_idx, dataset_idx)

        pair = (sample_idx, peak_idx)
        if pair in self.pair_to_dataset:
            dataset_idx = self.pair_to_dataset[pair]
            return self.datasets[dataset_idx].get_item_by_pair(sample_idx, peak_idx, dataset_idx)

        # 如果找不到映射，尝试在所有数据集中查找（后备）
        for dataset in self.datasets:
            if (sample_idx, peak_idx) in dataset.valid_indices:
                return dataset.get_item_by_pair(sample_idx, peak_idx)

        raise ValueError(f"无法找到pair ({sample_idx}, {peak_idx}) 在任何数据集中")

    def __getitem__(self, idx: int):
        key = self.valid_indices[idx]
        if len(key) == 3:
            ds_idx, sidx, pidx = key
            return self.get_item_by_pair(sidx, pidx, ds_idx)
        sidx, pidx = key
        return self.get_item_by_pair(sidx, pidx)
    
    def get_dataset_idx(self, idx: int) -> int:
        """获取指定索引对应的数据集索引
        
        Args:
            idx: 样本索引
            
        Returns:
            dataset_idx: 数据集索引（0, 1, 2, ...）
        """
        key = self.valid_indices[idx]
        if len(key) == 3:
            return key[0]  # (dataset_idx, sample_idx, peak_idx)
        # 单数据集模式，返回0
        return 0

# =========================
# 辅助函数：创建加权采样器
# =========================
def create_weighted_sampler(train_indices, dataset, config, logger):
    """
    为训练集创建加权随机采样器
    
    Args:
        train_indices: 训练集索引列表，格式可能是:
                      - 多数据集: [(dataset_idx, sample_idx, peak_idx), ...]
                      - 单数据集: [(sample_idx, peak_idx), ...]
        dataset: 数据集对象
        config: 全局配置对象
        logger: 日志记录器
    
    Returns:
        WeightedRandomSampler 或 None（如果未启用权重）
    """
    # 检查是否启用数据集权重
    use_weights = getattr(config.data, 'use_dataset_weights', False)
    if not use_weights:
        return None
    
    # 获取数据集权重配置
    dataset_weights = getattr(config.data, 'dataset_weights', {})
    if not dataset_weights:
        logger.warning("⚠️ 启用了数据集权重但未配置权重值，使用均匀采样")
        return None
    
    # 从配置中获取数据集名称列表
    # 检查是否为多数据集（通过检查 dataset 是否有 datasets 属性）
    is_multi_dataset = hasattr(dataset, 'datasets') and len(dataset.datasets) > 1
    
    # 获取数据集名称列表（从 config.data.input_files 的键）
    if hasattr(config.data, 'input_files'):
        if hasattr(config.data.input_files, 'keys'):
            dataset_names = list(config.data.input_files.keys())
        else:
            dataset_names = list(config.data.input_files.__dict__.keys())
    else:
        dataset_names = []
    
    if not is_multi_dataset:
        # 单数据集：所有样本使用相同权重
        if dataset_names:
            dataset_name = dataset_names[0]
            weight = dataset_weights.get(dataset_name, 1.0)
        else:
            weight = 1.0
        weights = [weight] * len(train_indices)
        logger.info(f"✅ 单数据集加权采样: 权重 = {weight}")
    else:
        # 多数据集：根据每个索引所属的数据集分配权重
        weights = []
        for idx in train_indices:
            if len(idx) == 3:  # 多数据集格式: (dataset_idx, sample_idx, peak_idx)
                dataset_idx = idx[0]
                if dataset_idx < len(dataset_names):
                    dataset_name = dataset_names[dataset_idx]
                    weight = dataset_weights.get(dataset_name, 1.0)
                else:
                    weight = 1.0
            else:  # 单数据集格式: (sample_idx, peak_idx)
                # 这种情况不应该出现在多数据集模式下，但为了安全起见
                weight = 1.0
            weights.append(weight)
        
        # 统计各数据集的权重使用情况
        weight_stats = {}
        for idx, weight in zip(train_indices, weights):
            if len(idx) == 3:
                dataset_idx = idx[0]
                if dataset_idx < len(dataset_names):
                    dataset_name = dataset_names[dataset_idx]
                    if dataset_name not in weight_stats:
                        weight_stats[dataset_name] = {'count': 0, 'weight': weight}
                    weight_stats[dataset_name]['count'] += 1
        
        logger.info("✅ 多数据集加权采样配置:")
        for name, stats in weight_stats.items():
            logger.info(f"   {name}: 权重 = {stats['weight']:.2f}, 样本数 = {stats['count']:,}")
    
    # 创建加权随机采样器
    # 注意：WeightedRandomSampler 需要权重张量，且长度必须等于数据集大小
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    sampler = WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=len(train_indices),
        replacement=True  # 允许重复采样以平衡不同权重的数据集
    )
    
    return sampler

# =========================
# 划分策略：Peak 级
# =========================
def create_data_loaders_peak(dataset, config, logger):
    """
    按Peak级划分数据集（避免motif特征泄露）
    
    核心策略：
    1. 识别所有unique peaks（来自4个ATAC样本）
    2. 随机划分peaks为train/val/test（70%/15%/15%）
    3. 同一个peak的所有54个实验条件都跟随该peak归属
    
    优势：
    - 测试模型对新基因组位点的预测能力
    - Motif特征（235维）严格隔离，无泄露
    - 更符合实际应用（预测新调控元件的表达）
    
    注意：
    - 实验条件会在train/val/test中重复（但只有6维，影响相对小）
    - Gene-level评估在多数据集模式下禁用（因g2p矩阵维度不同）
    
    Args:
        dataset: 原始数据集
        config: 配置对象
        logger: 日志记录器
    
    Returns:
        train_loader, val_loader, test_loader: 三个数据加载器
    """
    
    # 获取底层数据集
    base_ds = dataset
    try:
        while hasattr(base_ds, 'dataset'):
            base_ds = base_ds.dataset
    except Exception:
        pass
    
    logger.info("=" * 70)
    logger.info("📊 数据划分策略：Peak级划分（避免motif特征泄露）")
    logger.info("=" * 70)
    
    # 收集所有unique peaks及其对应的(sample_idx, peak_idx)对
    peak_to_pairs = {}  # peak_identifier → [(sample_idx, peak_idx), ...]
    
    if hasattr(base_ds, 'datasets'):  # MultiSinglePeakDataset
        logger.info(f"✅ 多数据集模式：{len(base_ds.datasets)}个ATAC样本")
        
        for dataset_idx, ds in enumerate(base_ds.datasets):
            for sample_idx, peak_idx in ds.valid_indices:
                # Peak identifier: (dataset_idx, peak_idx)
                peak_id = (dataset_idx, peak_idx)

                if peak_id not in peak_to_pairs:
                    peak_to_pairs[peak_id] = []
                # 记录三元组，避免跨数据集冲突
                peak_to_pairs[peak_id].append((dataset_idx, sample_idx, peak_idx))
    else:  # 单数据集
        logger.info("✅ 单数据集模式")
        for sample_idx, peak_idx in base_ds.valid_indices:
            peak_id = peak_idx  # 单数据集直接用peak_idx

            if peak_id not in peak_to_pairs:
                peak_to_pairs[peak_id] = []
            peak_to_pairs[peak_id].append((sample_idx, peak_idx))
    
    # 随机划分peaks
    all_peak_ids = list(peak_to_pairs.keys())
    rng = np.random.default_rng(config.experiment.seed)
    rng.shuffle(all_peak_ids)
    
    n_total_peaks = len(all_peak_ids)
    n_train_peaks = int(0.7 * n_total_peaks)
    n_val_peaks = int(0.15 * n_total_peaks)
    n_test_peaks = n_total_peaks - n_train_peaks - n_val_peaks
    
    train_peak_ids = set(all_peak_ids[:n_train_peaks])
    val_peak_ids = set(all_peak_ids[n_train_peaks:n_train_peaks + n_val_peaks])
    test_peak_ids = set(all_peak_ids[n_train_peaks + n_val_peaks:])
    
    # 根据peak归属，收集所有(sample_idx, peak_idx)对
    train_indices = []
    val_indices = []
    test_indices = []
    
    for peak_id, pairs in peak_to_pairs.items():
        if peak_id in train_peak_ids:
            train_indices.extend(pairs)  # 该peak的所有54个条件都进入训练集
        elif peak_id in val_peak_ids:
            val_indices.extend(pairs)
        elif peak_id in test_peak_ids:
            test_indices.extend(pairs)
    
    logger.info(f"\n📈 Peak级划分统计:")
    logger.info(f"  总unique peaks: {n_total_peaks:,}")
    logger.info(f"  ├─ 训练peaks: {n_train_peaks:,} ({n_train_peaks/n_total_peaks*100:.1f}%)")
    logger.info(f"  ├─ 验证peaks: {n_val_peaks:,} ({n_val_peaks/n_total_peaks*100:.1f}%)")
    logger.info(f"  └─ 测试peaks: {n_test_peaks:,} ({n_test_peaks/n_total_peaks*100:.1f}%)")
    logger.info(f"")
    logger.info(f"📈 训练样本数（peak×条件）:")
    logger.info(f"  ├─ 训练集: {len(train_indices):,} 个样本")
    logger.info(f"  ├─ 验证集: {len(val_indices):,} 个样本")
    logger.info(f"  └─ 测试集: {len(test_indices):,} 个样本")
    logger.info(f"")
    logger.info(f"✅ 划分特点：同一peak的所有条件在同一集合（无motif泄露）")
    logger.info("=" * 70)
    
    # 创建基于indices过滤的数据集
    train_dataset = FilteredByIndicesDataset(base_ds, train_indices)
    val_dataset = FilteredByIndicesDataset(base_ds, val_indices)
    test_dataset = FilteredByIndicesDataset(base_ds, test_indices)

    # 创建加权采样器（如果启用）
    weighted_sampler = create_weighted_sampler(train_indices, dataset, config, logger)
    
    # 创建DataLoader
    # 如果使用加权采样器，则不能同时使用 shuffle
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=(weighted_sampler is None),  # 只有不使用采样器时才shuffle
        sampler=weighted_sampler,  # 如果启用权重，使用加权采样器
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

# =========================
# 划分策略：Sample 级
# =========================
def create_data_loaders_sample(dataset, config, logger):
    """按样本级划分（每个样本的全部peaks留在同一集合，避免 gene 聚合泄露）。

    比较：
      - peak: 测试新位点泛化能力，gene-level需要列子矩阵修正。
      - sample: 测试跨样本条件泛化，gene-level更直接（完整峰集合）。
    返回: train_loader, val_loader, test_loader
    """
    base_ds = dataset
    try:
        while hasattr(base_ds, 'dataset'):
            base_ds = base_ds.dataset
    except Exception:
        pass

    logger.info("=" * 70)
    logger.info("📊 数据划分策略：Sample级划分（保持基因完整峰集合）")
    logger.info("=" * 70)

    num_samples = getattr(base_ds, 'num_samples', base_ds.data.shape[0])
    all_samples = list(range(num_samples))
    rng = np.random.default_rng(config.experiment.seed)
    rng.shuffle(all_samples)

    n_train = int(0.7 * num_samples)
    n_val = int(0.15 * num_samples)
    n_test = num_samples - n_train - n_val
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train + n_val]
    test_samples = all_samples[n_train + n_val:]

    logger.info(f"总样本: {num_samples} | 训练: {len(train_samples)} | 验证: {len(val_samples)} | 测试: {len(test_samples)}")

    # 生成索引列表
    def collect_indices(sample_list):
        out = []
        if hasattr(base_ds, 'datasets'):  # multi
            for ds_idx, ds in enumerate(base_ds.datasets):
                for sidx, pidx in ds.valid_indices:
                    if sidx in sample_list:
                        out.append((ds_idx, sidx, pidx))
        else:
            for sidx, pidx in base_ds.valid_indices:
                if sidx in sample_list:
                    out.append((sidx, pidx))
        return out

    train_indices = collect_indices(train_samples)
    val_indices = collect_indices(val_samples)
    test_indices = collect_indices(test_samples)

    logger.info(f"训练样本(peak×条件)数: {len(train_indices):,}; 验证: {len(val_indices):,}; 测试: {len(test_indices):,}")
    logger.info("✅ 同一样本的所有peaks保持在同一集合，gene-level评估无泄露。")

    train_dataset = FilteredByIndicesDataset(base_ds, train_indices)
    val_dataset = FilteredByIndicesDataset(base_ds, val_indices)
    test_dataset = FilteredByIndicesDataset(base_ds, test_indices)

    # 创建加权采样器（如果启用）
    weighted_sampler = create_weighted_sampler(train_indices, dataset, config, logger)
    
    # 创建DataLoader
    # 如果使用加权采样器，则不能同时使用 shuffle
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=(weighted_sampler is None),  # 只有不使用采样器时才shuffle
        sampler=weighted_sampler,  # 如果启用权重，使用加权采样器
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    return train_loader, val_loader, test_loader

# =========================
# 包装：根据配置选择划分策略
# =========================
def create_data_loaders(dataset, config, logger):
    strategy = getattr(config.data, 'split_strategy', 'peak')
    if strategy not in {'peak', 'sample'}:
        logger.warning(f"未知划分策略 '{strategy}'，回退使用 'peak'")
        strategy = 'peak'
    if strategy == 'sample':
        return create_data_loaders_sample(dataset, config, logger)
    return create_data_loaders_peak(dataset, config, logger)

def evaluate_model(model, data_loader, device, logger, split_name="validation"):
    """
    评估模型性能（单Peak版本，按原文拼接正负链计算指标）

    - 模型输出与标签均为 [batch, 1, 2]（最后维 2 表示正/负链）
    - 评估时将正负链直接展平拼接，得到一条长向量，用于计算 Pearson、Spearman、R²、MAE、MSE 等指标。
    - 返回：avg_loss, mae, slope, intercept, all_preds, all_targets, spearman_p
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    # 仅保留主指标收集（concat pos/neg）
    
    with torch.no_grad():
        for batch_x in tqdm(data_loader, desc=f'评估 {split_name}'):
            # 单Peak模式的数据结构
            batch_labels = batch_x['labels'].to(device)  # [batch_size, 1, 2] - 正负链表达
            features_for_model = {'motif_features': batch_x['motif_features'].to(device)}  # [batch_size, 1, features]
            
            # 前向传播
            outputs = model(features_for_model)  # [batch_size, 1, 2] - 正负链表达预测
            
            # 计算损失
            loss = model.compute_loss(outputs, batch_labels)
            total_loss += loss.item()
            
            # 收集预测和真实值 - 将正负链拼接（flatten）作为总体表达
            preds = outputs.squeeze(1).detach().cpu().numpy()  # [batch_size, 2]
            targets = batch_labels.squeeze(1).detach().cpu().numpy()  # [batch_size, 2]

            # 主指标：直接拼接正负链（flatten）
            all_preds.extend(preds.flatten())
            all_targets.extend(targets.flatten())
    
    # 计算平均损失
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    
    # 计算统计指标
    all_preds_np = np.array(all_preds)
    all_targets_np = np.array(all_targets)

    try:
        # ===== 主要指标 (按重要性排序) =====

        # 1. MAE - 平均绝对误差 (主要指标)
        mae = mean_absolute_error(all_targets_np, all_preds_np)

        # 2. Spearman相关系数 - 排序相关性
        spearman_rho, spearman_p = spearmanr(all_targets_np, all_preds_np)

        # 3. R² - 决定系数 (预测能力)
        r2 = r2_score(all_targets_np, all_preds_np)

        # 4. Pearson相关系数 - 线性相关性
        pearson_r, pearson_p = pearsonr(all_targets_np, all_preds_np)

        # 5. Kendall's Tau - 排序一致性
        from scipy.stats import kendalltau
        kendall_tau, kendall_p = kendalltau(all_targets_np, all_preds_np)

        # 6. 回归分析 - 斜率和截距
        reg = linregress(all_targets_np, all_preds_np)
        slope = reg.slope
        intercept = reg.intercept

        # 7. MSE - 均方误差
        mse = mean_squared_error(all_targets_np, all_preds_np)

        # 8. MAPE - 平均绝对百分比误差
        mape = np.mean(np.abs((all_targets_np - all_preds_np) / (all_targets_np + 1e-8))) * 100

        # 9. 额外有用的指标
        # RMSE - 均方根误差
        rmse = np.sqrt(mse)

        # 中位数绝对误差
        median_ae = np.median(np.abs(all_targets_np - all_preds_np))

        # 预测值范围
        pred_range = all_preds_np.max() - all_preds_np.min()
        true_range = all_targets_np.max() - all_targets_np.min()

        # 预测值分布统计
        pred_mean = all_preds_np.mean()
        pred_std = all_preds_np.std()

        true_mean = all_targets_np.mean()
        true_std = all_targets_np.std()

        # 误差分布统计
        errors = all_preds_np - all_targets_np
        error_mean = errors.mean()
        error_std = errors.std()
        error_median = np.median(errors)

        # 分位数误差 (关注不同表达水平)
        q25_error = np.percentile(np.abs(errors), 25)
        q75_error = np.percentile(np.abs(errors), 75)

        # 高表达区域误差 (表达值 > 中位数)
        high_expr_mask = all_targets_np > np.median(all_targets_np)
        high_expr_mae = mean_absolute_error(all_targets_np[high_expr_mask], all_preds_np[high_expr_mask]) if np.sum(high_expr_mask) > 0 else float('nan')

        # 低表达区域误差 (表达值 <= 中位数)
        low_expr_mask = all_targets_np <= np.median(all_targets_np)
        low_expr_mae = mean_absolute_error(all_targets_np[low_expr_mask], all_preds_np[low_expr_mask]) if np.sum(low_expr_mask) > 0 else float('nan')

    except Exception as e:
        # 如果计算失败，设置默认值
        mae = spearman_rho = r2 = pearson_r = kendall_tau = float('nan')
        slope = intercept = mse = mape = rmse = median_ae = float('nan')
        pred_range = true_range = pred_mean = pred_std = true_mean = true_std = float('nan')
        error_mean = error_std = error_median = q25_error = q75_error = float('nan')
        high_expr_mae = low_expr_mae = float('nan')
        logger.warning(f"{split_name}集指标计算失败: {e}")
    
    # 简洁日志（统一 peak-level 风格）
    logger.info(
        f"[PEAK] {split_name} | r={pearson_r:.6f} | rho={spearman_rho:.6f} | R2={r2:.6f} | "
        f"MAE={mae:.6f} | loss={avg_loss:.6f} | N={len(all_targets):,}"
    )
    
    # 返回主要指标 (MAE作为主要指标)
    return avg_loss, mae, slope, intercept, all_preds, all_targets, spearman_p

def evaluate_model_with_meta(model, data_loader, device, logger, split_name="test"):
    """评估模型并同时返回逐条预测的元信息（sample_idx、peak_idx、strand）。

    返回：avg_loss, mae, slope, intercept, all_preds, all_targets, spearman_p,
         peak_indices, sample_indices, strands
    其中 strands 为0/1，0=pos, 1=neg，对应拼接顺序。
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    peak_indices = []
    sample_indices = []
    strands = []  # 0=pos, 1=neg

    with torch.no_grad():
        for batch_x in tqdm(data_loader, desc=f'评估 {split_name} (with meta)'):
            batch_labels = batch_x['labels'].to(device)
            features_for_model = {'motif_features': batch_x['motif_features'].to(device)}
            outputs = model(features_for_model)
            loss = model.compute_loss(outputs, batch_labels)
            total_loss += loss.item()

            preds = outputs.squeeze(1).detach().cpu().numpy()  # [B,2]
            targets = batch_labels.squeeze(1).detach().cpu().numpy()  # [B,2]
            all_preds.extend(preds.flatten())
            all_targets.extend(targets.flatten())

            # 追加元信息，顺序与flatten一致：[pos, neg]
            bsz = preds.shape[0]
            b_peak_idx = batch_x['peak_idx']  # 长度 B
            b_sample_idx = batch_x['sample_idx']  # 长度 B
            # 展平为2*B，并填充strand标识
            for i in range(bsz):
                peak_indices.append(int(b_peak_idx[i]))
                sample_indices.append(int(b_sample_idx[i]))
                strands.append(0)  # pos
                peak_indices.append(int(b_peak_idx[i]))
                sample_indices.append(int(b_sample_idx[i]))
                strands.append(1)  # neg

    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0

    # 统计指标沿用 evaluate_model
    try:
        mae = mean_absolute_error(all_targets, all_preds)
        spearman_rho, spearman_p = spearmanr(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)
        pearson_r, pearson_p = pearsonr(all_targets, all_preds)
        reg = linregress(all_targets, all_preds)
        slope, intercept = reg.slope, reg.intercept
        _ = (r2, pearson_r, pearson_p)  # 防未使用告警
    except Exception as e:
        mae = float('nan'); spearman_p = float('nan'); slope = float('nan'); intercept = float('nan')
        logger.warning(f"{split_name}集指标计算失败(with meta): {e}")

    return avg_loss, mae, slope, intercept, all_preds, all_targets, spearman_p, peak_indices, sample_indices, strands

def evaluate_model_gene_level(model, data_loader, device, logger, split_name="validation"):
    """在保留原peak级拼接指标的同时，基于g2p权重计算gene级指标。

    返回:
      {
        'avg_loss': float,
        'peak': {'metrics': dict, 'preds': np.ndarray, 'targets': np.ndarray},
        'gene': {'metrics': dict, 'detail': {'pred': np.ndarray, 'true': np.ndarray, 'gene_ids': np.ndarray|None, 'strands': np.ndarray|None, 'sample_indices': np.ndarray|None}}
      }
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    # 兼容多种数据集包装：
    # - 原始数据集：YeastPeakSingleDataset
    # - 视图数据集：FilteredBySamplesDataset（带有base属性）
    # - 子集包装：torch.utils.data.Subset（带有dataset属性）
    base_ds = getattr(data_loader.dataset, 'base', data_loader.dataset)
    # 递归向下剥离 Subset 包装，拿到真正的底层数据集
    try:
        while hasattr(base_ds, 'dataset'):
            base_ds = base_ds.dataset
    except Exception:
        pass

    # 从底层数据集上获取 g2p 信息
    g2p_pos = getattr(base_ds, 'g2p_pos_csr', None)
    g2p_neg = getattr(base_ds, 'g2p_neg_csr', None)
    
    # 检查是否是多数据集（MultiSinglePeakDataset）
    is_multi_dataset = hasattr(base_ds, 'datasets') and len(getattr(base_ds, 'datasets', [])) > 1
    
    # 多数据集也需要映射：若是多数据集，采用“每个数据集内部用自己的矩阵聚合，再拼接指标”的策略
    have_gene_eval = False
    if is_multi_dataset:
        # 只要每个子数据集都有自己的 g2p，就允许做 gene-level 评估
        have_gene_eval = True
        for ds in getattr(base_ds, 'datasets', []):
            if getattr(ds, 'g2p_pos_csr', None) is None or getattr(ds, 'g2p_neg_csr', None) is None:
                have_gene_eval = False
                break
        if not have_gene_eval:
            logger.info("  ⚠️  多数据集模式：存在子数据集缺失 g2p 映射，gene-level 评估被跳过")
    else:
        have_gene_eval = (g2p_pos is not None) and (g2p_neg is not None)

    # 样本索引集合：
    # - 若数据集有明确的 parent_samples（样本级视图），优先使用
    # - 否则回退到底层数据的样本范围 [0, num_samples)
    split_samples = getattr(data_loader.dataset, 'parent_samples', None)
    if split_samples is None:
        split_samples = list(range(base_ds.data.shape[0]))
    sample_to_local = {s: i for i, s in enumerate(split_samples)}
    n_split = len(split_samples)
    
    # 对于多数据集，使用max_num_peaks；单数据集使用data.shape[1]
    n_peaks = getattr(base_ds, 'max_num_peaks', base_ds.data.shape[1])

    # 收集当前split实际包含的peak索引（用于峰级划分时的列子集与行内再归一化）
    included_peaks = None
    if hasattr(data_loader.dataset, 'valid_indices') and isinstance(getattr(data_loader.dataset, 'valid_indices'), list):
        try:
            included_peaks = []
            for item in data_loader.dataset.valid_indices:
                if len(item) == 3:
                    _, _, p = item
                else:
                    _, p = item
                included_peaks.append(int(p))
            included_peaks = sorted(set(included_peaks))
        except Exception:
            included_peaks = None
    # 若未提供，默认使用全部列
    if included_peaks is None:
        included_peaks = list(range(n_peaks))

    preds_pos = np.full((n_split, n_peaks), np.nan, dtype=np.float32)
    preds_neg = np.full((n_split, n_peaks), np.nan, dtype=np.float32)
    trues_pos = np.full((n_split, n_peaks), np.nan, dtype=np.float32)
    trues_neg = np.full((n_split, n_peaks), np.nan, dtype=np.float32)

    with torch.no_grad():
        for batch_x in tqdm(data_loader, desc=f'评估 {split_name} (gene-level)'):
            batch_labels = batch_x['labels'].to(device)
            features_for_model = {'motif_features': batch_x['motif_features'].to(device)}
            outputs = model(features_for_model)
            loss = model.compute_loss(outputs, batch_labels)
            total_loss += loss.item()

            preds = outputs.squeeze(1).detach().cpu().numpy()  # [B,2]
            targets = batch_labels.squeeze(1).detach().cpu().numpy()
            all_preds.extend(preds.reshape(-1))
            all_targets.extend(targets.reshape(-1))

            # 写入矩阵
            bsz = preds.shape[0]
            for i in range(bsz):
                sidx = int(batch_x['sample_idx'][i])
                pidx = int(batch_x['peak_idx'][i])
                lidx = sample_to_local.get(sidx, None)
                if lidx is None:
                    continue
                preds_pos[lidx, pidx] = preds[i, 0]
                preds_neg[lidx, pidx] = preds[i, 1]
                trues_pos[lidx, pidx] = targets[i, 0]
                trues_neg[lidx, pidx] = targets[i, 1]

    avg_loss = total_loss / max(1, len(data_loader))
    all_preds_np = np.array(all_preds)
    all_targets_np = np.array(all_targets)

    # peak级
    peak_metrics = {}
    try:
        peak_metrics = {
            'mae': float(mean_absolute_error(all_targets_np, all_preds_np)),
            'spearman_rho': float(spearmanr(all_targets_np, all_preds_np)[0]),
            'r2': float(r2_score(all_targets_np, all_preds_np)),
            'pearson_r': float(pearsonr(all_targets_np, all_preds_np)[0]),
            'mse': float(mean_squared_error(all_targets_np, all_preds_np)),
        }
    except Exception as e:
        logger.warning(f"peak级指标计算失败: {e}")

    # gene级
    gene_metrics = {}
    gene_detail = None
    if have_gene_eval:
        # 重要说明：当前gene级真值来源于“将峰级真值使用相同的g2p矩阵做加权聚合”。
        # 这不是独立的监督信号，聚合会降低噪声、提高相关性，因此gene级指标通常显著高于peak级，
        # 更接近一个“上界”估计。若希望更严格评估，应提供独立的基因级真值（如TPM）并直接对比。
        # gene-level 仅作为补充参考，日志保持简洁
        # 若是peak级划分，仅使用当前split包含的列构造子矩阵，并在行内重新归一化
        def _row_normalize(csr_mat):
            if csr_mat is None:
                return None
            row_sums = np.array(csr_mat.sum(axis=1)).ravel()
            data = csr_mat.data.copy()
            indptr = csr_mat.indptr.copy()
            fix_rows = (row_sums > 0)
            for r in np.where(fix_rows)[0]:
                s, e = indptr[r], indptr[r+1]
                if e > s:
                    data[s:e] /= (row_sums[r] + 1e-12)
            return csr_matrix((data, csr_mat.indices.copy(), indptr), shape=csr_mat.shape)

        if not is_multi_dataset:
            try:
                col_mask = np.zeros(n_peaks, dtype=bool)
                col_mask[np.array(included_peaks, dtype=int)] = True
                g2p_pos_eval = _row_normalize(g2p_pos[:, col_mask] if g2p_pos is not None else None)
                g2p_neg_eval = _row_normalize(g2p_neg[:, col_mask] if g2p_neg is not None else None)
            except Exception as e:
                logger.warning(f"构造按split列的gene映射失败，将退回使用完整矩阵（缺失峰视为0）：{e}")
                g2p_pos_eval = g2p_pos
                g2p_neg_eval = g2p_neg
        else:
            # 多数据集：为每个子数据集单独准备列子矩阵与归一化
            ds_col_mask = {}
            ds_g2p_eval = {}
            # 收集当前split涉及到的每个数据集的峰索引集合
            pair_to_dataset = getattr(base_ds, 'pair_to_dataset', {})
            ds_peaks_map = {}
            if hasattr(data_loader.dataset, 'valid_indices'):
                for item in data_loader.dataset.valid_indices:
                    if len(item) == 3:
                        ds_idx, _, pidx = item
                    else:
                        sidx, pidx = item
                        ds_idx = pair_to_dataset.get((sidx, pidx), None)
                    if ds_idx is None:
                        continue
                    ds_peaks_map.setdefault(int(ds_idx), set()).add(int(pidx))
            # 为每个子数据集构造掩码和子矩阵（无列时跳过，避免生成(n_gene,0)导致错误聚合）
            for ds_idx, ds in enumerate(getattr(base_ds, 'datasets', [])):
                n_peaks_ds = ds.data.shape[1]
                mask = np.zeros(n_peaks_ds, dtype=bool)
                for p in ds_peaks_map.get(ds_idx, set()):
                    if 0 <= p < n_peaks_ds:
                        mask[p] = True
                ds_col_mask[ds_idx] = mask
                pos_csr = getattr(ds, 'g2p_pos_csr', None)
                neg_csr = getattr(ds, 'g2p_neg_csr', None)
                has_cols = bool(mask.any())
                ds_g2p_eval[ds_idx] = (
                    _row_normalize(pos_csr[:, mask]) if (pos_csr is not None and has_cols) else None,
                    _row_normalize(neg_csr[:, mask]) if (neg_csr is not None and has_cols) else None
                )

        preds_pos_f = np.nan_to_num(preds_pos, nan=0.0)
        preds_neg_f = np.nan_to_num(preds_neg, nan=0.0)
        trues_pos_f = np.nan_to_num(trues_pos, nan=0.0)
        trues_neg_f = np.nan_to_num(trues_neg, nan=0.0)

        pos_gene_ids = getattr(base_ds, 'g2p_pos', {}).get('gene_ids', None)
        neg_gene_ids = getattr(base_ds, 'g2p_neg', {}).get('gene_ids', None)

        gene_pred_list = []
        gene_true_list = []
        # 额外保存TPM域再log的聚合结果（可选）
        gene_pred_list_tpm = []
        gene_true_list_tpm = []
        gene_ids_list = []
        strand_list = []
        sample_list = []
        for i in range(n_split):
            # 默认初始化为空，避免在多数据集路径下变量未定义
            pos_pred_vec = np.array([])
            pos_true_vec = np.array([])
            neg_pred_vec = np.array([])
            neg_true_vec = np.array([])
            # TPM域变量默认空
            pos_pred_vec_tpm = np.array([])
            pos_true_vec_tpm = np.array([])
            neg_pred_vec_tpm = np.array([])
            neg_true_vec_tpm = np.array([])
            if not is_multi_dataset:
                # 单数据集逻辑
                if g2p_pos_eval is not None and g2p_pos_eval.shape[1] < n_peaks:
                    vec_pos = preds_pos_f[i, col_mask]
                    vec_pos_true = trues_pos_f[i, col_mask]
                    pos_pred_vec = g2p_pos_eval.dot(vec_pos)
                    pos_true_vec = g2p_pos_eval.dot(vec_pos_true)
                    # TPM域：先逆变换，再聚合，最后log
                    vec_pos_tpm = np.power(2.0, vec_pos) - 1.0
                    vec_pos_true_tpm = np.power(2.0, vec_pos_true) - 1.0
                    pos_pred_vec_tpm = np.log2(g2p_pos_eval.dot(vec_pos_tpm) + 1.0)
                    pos_true_vec_tpm = np.log2(g2p_pos_eval.dot(vec_pos_true_tpm) + 1.0)
                else:
                    if g2p_pos is not None:
                        pos_pred_vec = g2p_pos.dot(preds_pos_f[i])
                        pos_true_vec = g2p_pos.dot(trues_pos_f[i])
                        # TPM域
                        vec_pos = preds_pos_f[i]
                        vec_pos_true = trues_pos_f[i]
                        vec_pos_tpm = np.power(2.0, vec_pos) - 1.0
                        vec_pos_true_tpm = np.power(2.0, vec_pos_true) - 1.0
                        pos_pred_vec_tpm = np.log2(g2p_pos.dot(vec_pos_tpm) + 1.0)
                        pos_true_vec_tpm = np.log2(g2p_pos.dot(vec_pos_true_tpm) + 1.0)

                if g2p_neg_eval is not None and g2p_neg_eval.shape[1] < n_peaks:
                    vec_neg = preds_neg_f[i, col_mask]
                    vec_neg_true = trues_neg_f[i, col_mask]
                    neg_pred_vec = g2p_neg_eval.dot(vec_neg)
                    neg_true_vec = g2p_neg_eval.dot(vec_neg_true)
                    # TPM域
                    vec_neg_tpm = np.power(2.0, vec_neg) - 1.0
                    vec_neg_true_tpm = np.power(2.0, vec_neg_true) - 1.0
                    neg_pred_vec_tpm = np.log2(g2p_neg_eval.dot(vec_neg_tpm) + 1.0)
                    neg_true_vec_tpm = np.log2(g2p_neg_eval.dot(vec_neg_true_tpm) + 1.0)
                else:
                    if g2p_neg is not None:
                        neg_pred_vec = g2p_neg.dot(preds_neg_f[i])
                        neg_true_vec = g2p_neg.dot(trues_neg_f[i])
                        # TPM域
                        vec_neg = preds_neg_f[i]
                        vec_neg_true = trues_neg_f[i]
                        vec_neg_tpm = np.power(2.0, vec_neg) - 1.0
                        vec_neg_true_tpm = np.power(2.0, vec_neg_true) - 1.0
                        neg_pred_vec_tpm = np.log2(g2p_neg.dot(vec_neg_tpm) + 1.0)
                        neg_true_vec_tpm = np.log2(g2p_neg.dot(vec_neg_true_tpm) + 1.0)
            else:
                # 多数据集逻辑：对每个数据集分别聚合，然后拼接
                # 先为该样本构建一个 dataset_idx -> (vec_pos, vec_neg) 的视图
                # 使用预先准备的掩码与子矩阵
                for ds_idx, ds in enumerate(getattr(base_ds, 'datasets', [])):
                    pos_eval, neg_eval = ds_g2p_eval.get(ds_idx, (None, None))
                    mask = ds_col_mask.get(ds_idx, None)
                    if mask is None:
                        continue
                    if pos_eval is not None:
                        vec_pos = preds_pos_f[i, :ds.data.shape[1]][mask]
                        vec_pos_true = trues_pos_f[i, :ds.data.shape[1]][mask]
                        pp = pos_eval.dot(vec_pos)
                        pt = pos_eval.dot(vec_pos_true)
                        if mask.any() and pp.size:
                            gene_pred_list.append(pp)
                            gene_true_list.append(pt)
                            # TPM域
                            vec_pos_tpm = np.power(2.0, vec_pos) - 1.0
                            vec_pos_true_tpm = np.power(2.0, vec_pos_true) - 1.0
                            pp_tpm = np.log2(pos_eval.dot(vec_pos_tpm) + 1.0)
                            pt_tpm = np.log2(pos_eval.dot(vec_pos_true_tpm) + 1.0)
                            gene_pred_list_tpm.append(pp_tpm)
                            gene_true_list_tpm.append(pt_tpm)
                            gids = getattr(ds, 'g2p_pos', {}).get('gene_ids', None)
                            if gids is not None:
                                gene_ids_list.extend([str(g) for g in gids])
                                strand_list.extend(['+'] * len(pp))
                                sample_list.extend([split_samples[i]] * len(pp))
                    if neg_eval is not None:
                        vec_neg = preds_neg_f[i, :ds.data.shape[1]][mask]
                        vec_neg_true = trues_neg_f[i, :ds.data.shape[1]][mask]
                        pn = neg_eval.dot(vec_neg)
                        tn = neg_eval.dot(vec_neg_true)
                        if mask.any() and pn.size:
                            gene_pred_list.append(pn)
                            gene_true_list.append(tn)
                            # TPM域
                            vec_neg_tpm = np.power(2.0, vec_neg) - 1.0
                            vec_neg_true_tpm = np.power(2.0, vec_neg_true) - 1.0
                            pn_tpm = np.log2(neg_eval.dot(vec_neg_tpm) + 1.0)
                            tn_tpm = np.log2(neg_eval.dot(vec_neg_true_tpm) + 1.0)
                            gene_pred_list_tpm.append(pn_tpm)
                            gene_true_list_tpm.append(tn_tpm)
                            gids = getattr(ds, 'g2p_neg', {}).get('gene_ids', None)
                            if gids is not None:
                                gene_ids_list.extend([str(g) for g in gids])
                                strand_list.extend(['-'] * len(pn))
                                sample_list.extend([split_samples[i]] * len(pn))

            if pos_pred_vec.size:
                gene_pred_list.append(pos_pred_vec)
                gene_true_list.append(pos_true_vec)
                if pos_pred_vec_tpm.size:
                    gene_pred_list_tpm.append(pos_pred_vec_tpm)
                    gene_true_list_tpm.append(pos_true_vec_tpm)
                if pos_gene_ids is not None:
                    gene_ids_list.extend([str(g) for g in pos_gene_ids])
                    strand_list.extend(['+'] * len(pos_pred_vec))
                    sample_list.extend([split_samples[i]] * len(pos_pred_vec))
            if neg_pred_vec.size:
                gene_pred_list.append(neg_pred_vec)
                gene_true_list.append(neg_true_vec)
                if neg_pred_vec_tpm.size:
                    gene_pred_list_tpm.append(neg_pred_vec_tpm)
                    gene_true_list_tpm.append(neg_true_vec_tpm)
                if neg_gene_ids is not None:
                    gene_ids_list.extend([str(g) for g in neg_gene_ids])
                    strand_list.extend(['-'] * len(neg_pred_vec))
                    sample_list.extend([split_samples[i]] * len(neg_pred_vec))

        if gene_pred_list:
            gene_pred_concat_logspace = np.concatenate(gene_pred_list)
            gene_true_concat_logspace = np.concatenate(gene_true_list)
            
            # ========== 两种聚合域：直接log空间 & TPM域再log ==========
            agg_mode = getattr(base_ds, 'tpm_aggregation', 'logspace_direct')
            
            # 构建TPM域聚合结果
            tpm_pred_concat = None
            tpm_true_concat = None
            if gene_pred_list_tpm:
                tpm_pred_concat = np.concatenate(gene_pred_list_tpm)
                tpm_true_concat = np.concatenate(gene_true_list_tpm)
            
            # ✅ 根据配置选择使用哪种聚合结果
            if agg_mode == 'tpm_then_log' and tpm_pred_concat is not None:
                # 使用TPM域聚合（生物学正确）
                gene_pred_concat = tpm_pred_concat
                gene_true_concat = tpm_true_concat
                pass
            else:
                # 使用log空间直接聚合（默认）
                gene_pred_concat = gene_pred_concat_logspace
                gene_true_concat = gene_true_concat_logspace
                pass
            
            try:
                gene_metrics = {
                    'mae': float(mean_absolute_error(gene_true_concat, gene_pred_concat)),
                    'spearman_rho': float(spearmanr(gene_true_concat, gene_pred_concat)[0]),
                    'r2': float(r2_score(gene_true_concat, gene_pred_concat)),
                    'pearson_r': float(pearsonr(gene_true_concat, gene_pred_concat)[0]),
                    'mse': float(mean_squared_error(gene_true_concat, gene_pred_concat)),
                    'agg_mode': agg_mode,
                }
                gene_detail = {
                    'pred': gene_pred_concat,  # 使用选定的聚合结果
                    'true': gene_true_concat,
                    'gene_ids': np.array(gene_ids_list) if len(gene_ids_list) else None,
                    'strands': np.array(strand_list) if len(strand_list) else None,
                    'sample_indices': np.array(sample_list) if len(sample_list) else None,
                }
                # 同时保存两种聚合结果供对比
                gene_detail['pred_logspace'] = gene_pred_concat_logspace
                gene_detail['true_logspace'] = gene_true_concat_logspace
                if tpm_pred_concat is not None:
                    gene_detail['pred_tpm_mode'] = tpm_pred_concat
                    gene_detail['true_tpm_mode'] = tpm_true_concat
            except Exception as e:
                logger.warning(f"gene级指标计算失败: {e}")

            # ===== 真实TPM对比 =====
            real_truth_map = getattr(base_ds, 'real_gene_truth', None)
            if real_truth_map and gene_detail and gene_detail.get('gene_ids') is not None:
                try:
                    gids_all = gene_detail['gene_ids']
                    preds_all = gene_detail['pred']
                    # 汇总正负链重复：取平均
                    accum = {}
                    counts = {}
                    for gid, val in zip(gids_all, preds_all):
                        accum[gid] = accum.get(gid, 0.0) + float(val)
                        counts[gid] = counts.get(gid, 0) + 1
                    pred_reduced = {g: accum[g] / counts[g] for g in accum}
                    overlap = [g for g in pred_reduced if g in real_truth_map]
                    
                    if len(overlap) > 0:
                        # 计算真实TPM对比指标
                        pred_vals = np.array([pred_reduced[g] for g in overlap])
                        true_vals = np.array([real_truth_map[g] for g in overlap])
                        
                        real_metrics = {
                            'pearson_r': float(pearsonr(true_vals, pred_vals)[0]),
                            'spearman_rho': float(spearmanr(true_vals, pred_vals)[0]),
                            'mae': float(mean_absolute_error(true_vals, pred_vals)),
                            'r2': float(r2_score(true_vals, pred_vals)),
                            'n_genes': len(overlap)
                        }
                        
                        gene_detail['real_truth_metrics'] = real_metrics
                        logger.info(f"[GENE-LEVEL_REAL] {split_name} Pearson={real_metrics['pearson_r']:.6f}, MAE={real_metrics['mae']:.6f}, Genes={real_metrics['n_genes']}")
                    else:
                        logger.warning("真实TPM对比：无重叠基因，跳过")
                except Exception as e:
                    logger.warning(f"真实TPM基因级对比失败: {e}")

    if gene_metrics:
        logger.info(
            f"[{split_name}] peak: r={peak_metrics.get('pearson_r', float('nan')):.6f}, "
            f"mae={peak_metrics.get('mae', float('nan')):.6f}, r2={peak_metrics.get('r2', float('nan')):.6f}, "
            f"n={len(all_preds_np):,} | gene: r={gene_metrics.get('pearson_r', float('nan')):.6f}, "
            f"mae={gene_metrics.get('mae', float('nan')):.6f}, r2={gene_metrics.get('r2', float('nan')):.6f}"
        )
    else:
        logger.info(
            f"[{split_name}] peak: r={peak_metrics.get('pearson_r', float('nan')):.6f}, "
            f"mae={peak_metrics.get('mae', float('nan')):.6f}, r2={peak_metrics.get('r2', float('nan')):.6f}, "
            f"n={len(all_preds_np):,} | gene: N/A"
        )

    # 汇总更详细的统计信息，便于外层日志输出
    out = {
        'avg_loss': avg_loss,
        'peak': {
            'metrics': peak_metrics,
            'preds': all_preds_np,
            'targets': all_targets_np,
            'count': len(all_preds_np)
        },
        'gene': {
            'metrics': gene_metrics,
            'detail': gene_detail,
            'count': int(gene_detail['true'].size) if (gene_detail and gene_detail.get('true') is not None) else 0
        }
    }
    return out

def train_experiment(experiment_name: str, experiment_config: dict, config: DictConfig):
    """训练单个实验 - 单Peak版本"""
    
    global current_model, current_test_loader, current_device, current_logger, current_output_dir, training_completed
    
    # 创建输出目录
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(config.data.output_base_dir) / f"{experiment_config.output_dir}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置全局变量
    current_output_dir = output_dir
    
    # 提取超参数（在函数开始时定义，后续会用到）
    hyperparams = {
        'training/learning_rate': config.training.learning_rate,
        'training/batch_size': config.training.batch_size,
        'training/max_epochs': config.training.max_epochs,
        'training/weight_decay': config.training.weight_decay,
        'training/clip_grad': config.training.get('clip_grad', 0),
        'training/warmup_epochs': config.training.get('warmup_epochs', 0),
        'training/early_stopping_patience': config.training.get('early_stopping_patience', None),
        'training/early_stopping_min_delta': config.training.get('early_stopping_min_delta', 0.001),
        'experiment/seed': config.experiment.seed,
    }
    
    # 记录调度器参数
    if hasattr(config, 'scheduler') and hasattr(config.scheduler, 'type'):
        hyperparams['scheduler/type'] = config.scheduler.type
    
    # 记录模型参数（如果有）
    if hasattr(config.model, 'model'):
        model_cfg = config.model.model
        if hasattr(model_cfg, 'hidden_dim'):
            hyperparams['model/hidden_dim'] = model_cfg.hidden_dim
        if hasattr(model_cfg, 'num_layers'):
            hyperparams['model/num_layers'] = model_cfg.num_layers
    
    # 设置日志
    # 清除之前的handlers，避免重复
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / 'train_single_peak.log', mode='w', encoding='utf-8')
        ],
        force=True  # 强制重新配置
    )
    logger = logging.getLogger(__name__)
    current_logger = logger
    
    logger.info(f"开始实验: {experiment_config.name}")
    logger.info(f"实验描述: {experiment_config.description}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"训练方法: 单Peak方法 (每个peak独立训练)")
    
    # 打印配置信息用于调试
    logger.info(f"=== 配置信息 ===")
    logger.info(f"max_epochs: {config.training.max_epochs}")
    logger.info(f"batch_size: {config.training.batch_size}")
    logger.info(f"learning_rate: {config.training.learning_rate}")
    logger.info(f"weight_decay: {config.training.weight_decay}")
    logger.info(f"clip_grad: {config.training.clip_grad}")
    logger.info(f"early_stopping_patience: {config.training.early_stopping_patience}")
    logger.info(f"early_stopping_min_delta: {config.training.early_stopping_min_delta}")
    logger.info(f"=================")
    
    # 设置随机种子
    torch.manual_seed(config.experiment.seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    current_device = device
    
    # 准备数据集 - 单Peak版本
    if len(experiment_config.input_files) == 1:
        # 单数据集
        data_path = config.data.input_files[experiment_config.input_files[0]]
        logger.info(f"使用单数据集: {data_path}")
        dataset = YeastPeakSingleDataset(data_path)
    else:
        # 多数据集
        data_paths = [config.data.input_files[f] for f in experiment_config.input_files]
        logger.info(f"使用多数据集: {len(data_paths)} 个文件")
        for i, path in enumerate(data_paths):
            logger.info(f"  数据集 {i+1}: {path}")
        dataset = MultiSinglePeakDataset(data_paths)

    # 记录并附加聚合模式（peak->gene）
    agg_mode = getattr(config.data, 'tpm_aggregation', 'logspace_direct')
    setattr(dataset, 'tpm_aggregation', agg_mode)
    logger.info(f"Gene聚合模式(tpm_aggregation): {agg_mode}")

    # 加载真实TPM (若启用)
    if getattr(config.data, 'use_real_gene_truth', False):
        real_csv = getattr(config.data, 'real_gene_tpm_csv', None)
        reduce_method = getattr(config.data, 'gene_truth_reduce', 'mean')
        real_map = load_real_gene_tpm_csv(real_csv, reduce_method, logger=logger)
        if real_map:
            setattr(dataset, 'real_gene_truth', real_map)
            logger.info(f"✅ 已附加真实基因TPM: {len(real_map)} genes")
        else:
            logger.warning("⚠️ 真实TPM加载失败或为空，继续使用聚合的峰级真值作为参考")
    else:
        logger.info("未启用真实TPM评价 (use_real_gene_truth = false)")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(dataset, config, logger)
    current_test_loader = test_loader
    
    # 创建模型
    model = YeastModel(config.model.model)
    model = model.to(device)
    current_model = model
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # 创建学习率调度器（按配置选择），支持可选warmup
    def create_scheduler_from_config(cfg, opt):
        # 读取调度器类型
        sched_type = None
        try:
            if hasattr(cfg, 'scheduler') and 'type' in cfg.scheduler:
                sched_type = cfg.scheduler.type
        except Exception:
            sched_type = None

        base_lr = cfg.training.learning_rate
        max_epochs = cfg.training.max_epochs

        # 缺省：余弦退火
        if sched_type is None or sched_type == 'cosine':
            eta_min = None
            try:
                eta_min = cfg.scheduler.cosine.get('eta_min', None)
            except Exception:
                eta_min = None
            if eta_min is None:
                eta_min = base_lr * 1e-2
            main_sched = CosineAnnealingLR(opt, T_max=max_epochs, eta_min=eta_min)

        elif sched_type == 'cosine_warm_restarts':
            # 使用配置或默认参数
            T_0 = 50
            T_mult = 2
            eta_min = 1e-7
            try:
                T_0 = cfg.scheduler.cosine_warm_restarts.get('T_0', T_0)
                T_mult = cfg.scheduler.cosine_warm_restarts.get('T_mult', T_mult)
                eta_min = cfg.scheduler.cosine_warm_restarts.get('eta_min', eta_min)
            except Exception:
                pass
            main_sched = CosineAnnealingWarmRestarts(opt, T_0=T_0, T_mult=T_mult, eta_min=eta_min)

        elif sched_type == 'step':
            step_size = 30
            gamma = 0.5
            try:
                step_size = cfg.scheduler.step.get('step_size', step_size)
                gamma = cfg.scheduler.step.get('gamma', gamma)
            except Exception:
                pass
            main_sched = StepLR(opt, step_size=step_size, gamma=gamma)

        elif sched_type == 'exponential':
            gamma = 0.98
            try:
                gamma = cfg.scheduler.exponential.get('gamma', gamma)
            except Exception:
                pass
            main_sched = ExponentialLR(opt, gamma=gamma)
        else:
            # 回退到余弦
            main_sched = CosineAnnealingLR(opt, T_max=max_epochs, eta_min=base_lr * 1e-2)

        # 可选warmup（线性预热），在 training.warmup_epochs > 0 时启用
        warmup_epochs = 0
        try:
            warmup_epochs = cfg.training.get('warmup_epochs', 0)
        except Exception:
            warmup_epochs = 0

        if warmup_epochs and warmup_epochs > 0:
            warmup = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
            scheduler = SequentialLR(opt, schedulers=[warmup, main_sched], milestones=[warmup_epochs])
            return scheduler
        else:
            return main_sched

    scheduler = create_scheduler_from_config(config, optimizer)
    logger.info(f"学习率调度器: {type(scheduler).__name__}")
    
    # 创建TensorBoard writer
    tensorboard_dir = output_dir / 'tensorboard_logs'
    tensorboard_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)
    logger.info(f"TensorBoard日志目录: {tensorboard_dir}")
    
    # ========== 记录超参数配置 ==========
    logger.info(f"")
    logger.info(f"{'='*100}")
    logger.info(f"📋 实验超参数配置")
    logger.info(f"{'='*100}")
    
    # 打印到日志
    for key, value in hyperparams.items():
        logger.info(f"  {key:.<50} {value}")
    
    logger.info(f"{'='*100}")
    logger.info(f"")
    
    # 将超参数保存到YAML文件
    config_output_file = output_dir / 'experiment_config.yaml'
    OmegaConf.save(config, config_output_file)
    logger.info(f"✅ 实验配置已保存到: {config_output_file}")
    
    # 记录模型结构到TensorBoard
    try:
        # 创建一个示例输入来记录模型结构 - 单Peak模式
        # 特征维度: 470 motif + 1 accessibility + 74 condition = 545 (KM酵母)
        sample_input = torch.randn(1, 1, 545).to(device)  # [batch_size, 1, features]
        sample_features = {'motif_features': sample_input}

        # 记录模型计算图
        writer.add_graph(model, sample_features)
        logger.info("模型结构已记录到TensorBoard")
    except Exception as e:
        logger.warning(f"记录模型结构失败: {e}")
    
    # ============================================================================
    # 训练主循环
    # ============================================================================
    # 关键指标初始化
    best_val_pearson = -1.0  # 最佳验证集Pearson相关系数（越大越好）
    best_val_mae = float('inf')  # 最佳验证集MAE（越小越好）
    
    # 训练历史记录（用于可视化）
    train_losses = []  # 训练损失历史
    val_losses = []  # 验证损失历史
    train_maes = []  # 训练MAE历史
    val_maes = []  # 验证MAE历史
    val_pearsons = []  # 验证集Pearson相关系数历史
    val_spearmans = []  # 验证集Spearman相关系数历史
    lr_history = []  # 学习率历史
    
    # 早停相关参数
    patience_counter = 0  # 连续无改善轮数
    early_stopping_patience = config.training.get('early_stopping_patience', None)
    early_stopping_min_delta = config.training.get('early_stopping_min_delta', 0.001)
    
    logger.info(f"开始训练，共 {config.training.max_epochs} 个epoch")
    if early_stopping_patience:
        logger.info(f"早停耐心值: {early_stopping_patience}, 最小改善阈值: {early_stopping_min_delta}")
        logger.info(f"早停监控指标: peak-level Pearson r")
    
    # 详细验证历史记录（用于事后分析peak/gene级指标差异）
    val_history_rows = []

    # ========== 训练循环开始 ==========
    for epoch in range(config.training.max_epochs):
        if interrupted:
            logger.warning("训练被中断")
            break
            
        # ========== 训练阶段 (Training Phase) ==========
        # 设置模型为训练模式（启用dropout、batch normalization等）
        model.train()
        epoch_train_loss = 0  # 累加本轮的训练损失
        epoch_train_mae = 0  # 累加本轮的训练MAE
        train_batches = 0  # 统计batch数量
        
        # 遍历训练集所有batch
        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.training.max_epochs}')):
            if interrupted:
                break
                
            # ========== 数据准备 ==========
            # 单Peak模式：每个样本是一个peak
            # batch_labels shape: [batch_size, 1, 2] (2 = 正链+负链表达)
            # features shape: [batch_size, 1, 545] (545 = 470 motif + 1 accessibility + 74 condition, KM酵母)
            batch_labels = batch_data['labels'].to(device)
            features_for_model = {'motif_features': batch_data['motif_features'].to(device)}
            
            # ========== 前向传播 (Forward Pass) ==========
            # 清零梯度（重要！否则梯度会累积）
            optimizer.zero_grad()
            
            # 模型预测：shape [batch_size, 1, 2]
            outputs = model(features_for_model)
            
            # ========== 损失计算与反向传播 ==========
            # 计算损失（MSE loss）
            loss = model.compute_loss(outputs, batch_labels)
            
            # 反向传播计算梯度
            loss.backward()
            
            # ========== 梯度裁剪 (Gradient Clipping) ==========
            # 防止梯度爆炸，提高训练稳定性
            if config.training.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_grad)
            
            # ========== 优化器更新参数 ==========
            optimizer.step()
            
            # ========== 记录训练指标 ==========
            epoch_train_loss += loss.item()  # 累加损失
            
            # 计算并累加MAE（不计算梯度，节省计算）
            with torch.no_grad():
                batch_mae = torch.mean(torch.abs(outputs - batch_labels)).item()
                epoch_train_mae += batch_mae
            train_batches += 1
        
        # ========== 计算训练集平均指标 ==========
        # 除以batch数量得到平均值
        if train_batches > 0:
            avg_train_loss = epoch_train_loss / train_batches
            avg_train_mae = epoch_train_mae / train_batches
        else:
            avg_train_loss = 0
            avg_train_mae = 0
        
        # ========== 验证阶段 (Validation Phase) ==========
        # 同时计算peak级和gene级指标
        # 这允许我们在两个层级评估模型性能：
        # 1. Peak级：直接评估peak预测的准确性
        # 2. Gene级：将peak预测聚合为gene表达后评估（更符合生物学意义）
        val_eval = evaluate_model_gene_level(model, val_loader, device, logger, "validation")
        val_loss = val_eval['avg_loss']
        
        # ========== 选择主要监控指标 ==========
        # 按用户需求：训练/早停主指标固定使用 peak-level Pearson r
        # gene-level 指标仍保留记录
        val_pearson = val_eval['peak']['metrics'].get('pearson_r', float('nan'))
        val_mae = val_eval['peak']['metrics'].get('mae', float('nan'))
        val_r2 = val_eval['peak']['metrics'].get('r2', float('nan'))
        val_spearman = val_eval['peak']['metrics'].get('spearman_rho', float('nan'))

        # 用 peak 级数据计算回归斜率和截距（用于可视化）
        try:
            _vt = np.array(val_eval['peak']['targets'])
            _vp = np.array(val_eval['peak']['preds'])
            reg = linregress(_vt, _vp)
            val_slope, val_intercept = reg.slope, reg.intercept
        except Exception:
            val_slope, val_intercept = float('nan'), float('nan')
        
        # 记录到TensorBoard - 混合指标 + Gene指标
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('MAE/Train', avg_train_mae, epoch)
        writer.add_scalar('MAE/Validation', val_mae, epoch)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        lr_history.append(current_lr)
        # 主监控指标（peak-level）
        writer.add_scalar('Pearson/Peak', val_pearson, epoch)
        writer.add_scalar('Spearman/Peak', val_spearman, epoch)
        writer.add_scalar('R2/Peak', val_r2, epoch)

        # 同时记录peak级指标，便于对比
        if val_eval['peak']['metrics']:
            writer.add_scalar('Peak/Pearson_r', val_eval['peak']['metrics'].get('pearson_r', float('nan')), epoch)
            writer.add_scalar('Peak/MAE', val_eval['peak']['metrics'].get('mae', float('nan')), epoch)
            writer.add_scalar('Peak/R2', val_eval['peak']['metrics'].get('r2', float('nan')), epoch)

        if val_eval['gene']['metrics']:
            writer.add_scalar('Gene/Pearson_r', val_eval['gene']['metrics'].get('pearson_r', float('nan')), epoch)
            writer.add_scalar('Gene/MAE', val_eval['gene']['metrics'].get('mae', float('nan')), epoch)
            writer.add_scalar('Gene/R2', val_eval['gene']['metrics'].get('r2', float('nan')), epoch)

        # 保存到内存历史表
        val_history_rows.append({
            'epoch': epoch+1,
            'lr': current_lr,
            'val_loss': val_loss,
            'peak_pearson': val_eval['peak']['metrics'].get('pearson_r', float('nan')),
            'peak_mae': val_eval['peak']['metrics'].get('mae', float('nan')),
            'peak_r2': val_eval['peak']['metrics'].get('r2', float('nan')),
            'peak_count': val_eval['peak'].get('count', 0),
            'gene_pearson': val_eval['gene']['metrics'].get('pearson_r', float('nan')) if val_eval['gene']['metrics'] else float('nan'),
            'gene_mae': val_eval['gene']['metrics'].get('mae', float('nan')) if val_eval['gene']['metrics'] else float('nan'),
            'gene_r2': val_eval['gene']['metrics'].get('r2', float('nan')) if val_eval['gene']['metrics'] else float('nan'),
            'gene_count': val_eval['gene'].get('count', 0)
        })
        
        # 验证散点图输出频率：前10轮每2轮，之后每10轮
        if ((epoch + 1) <= 10 and (epoch + 1) % 2 == 0) or ((epoch + 1) > 10 and (epoch + 1) % 10 == 0):
            try:
                # 固定使用 peak-level 数据生成散点图
                targets_np = np.array(val_eval['peak']['targets'])
                preds_np = np.array(val_eval['peak']['preds'])
                level_name = 'peak'
                scatter_pearson = val_eval['peak']['metrics'].get('pearson_r', float('nan'))
                scatter_spearman = val_eval['peak']['metrics'].get('spearman_rho', float('nan'))
                scatter_r2 = val_eval['peak']['metrics'].get('r2', float('nan'))
                scatter_mae = val_eval['peak']['metrics'].get('mae', float('nan'))
                
                max_points = 100000
                lo, hi = 0.0, 20.0
                scatter_dir = output_dir / 'validation_plots'
                scatter_dir.mkdir(parents=True, exist_ok=True)
                
                fig, ax = plt.subplots(figsize=(10, 10))
                if len(targets_np) > max_points:
                    idx = np.random.choice(len(targets_np), max_points, replace=False)
                    x_plot = targets_np[idx]
                    y_plot = preds_np[idx]
                else:
                    x_plot = targets_np
                    y_plot = preds_np

                ax.scatter(x_plot, y_plot, s=1.5, alpha=0.15, c='#1f77b4', edgecolors='none', zorder=1)
                ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1, label='y=x', zorder=10)
                try:
                    x_line = np.linspace(lo, hi, 200)
                    # 使用实际数据的回归线
                    reg = linregress(targets_np, preds_np)
                    y_line = reg.slope * x_line + reg.intercept
                    fit_label = f'fit: y={reg.slope:.3f}x+{reg.intercept:.3f}'
                    ax.plot(x_line, y_line, color='blue', linewidth=1.2, zorder=3, label=fit_label)
                except Exception:
                    pass
                ax.set_xlabel('True expression', fontsize=14)
                ax.set_ylabel('Predicted expression', fontsize=14)
                ax.set_title(f'Validation predictions (peak-level) - Epoch {epoch+1}\nPearson r={scatter_pearson:.4f}, Spearman ρ={scatter_spearman:.4f}, R²={scatter_r2:.4f}', fontsize=16)
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                ax.tick_params(axis='both', labelsize=12)
                ax.legend(loc='lower right', fontsize=12, framealpha=0.95, edgecolor='#444444')
                ax.grid(True, alpha=0.3)

                # 确保N值和指标都来自同一个级别
                stats_text = (
                    f'N = {len(targets_np):,}\n'
                    f'Pearson r = {scatter_pearson:.4f}\n'
                    f'Spearman ρ = {scatter_spearman:.4f}\n'
                    f'R² = {scatter_r2:.4f}\n'
                    f'MAE = {scatter_mae:.4f}'
                )
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                        fontsize=12, va='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0e6c8', alpha=0.9, edgecolor='#a98f5a'))

                plt.tight_layout()
                writer.add_figure('Validation/Scatter', fig, epoch)
                writer.flush()
                scatter_path = scatter_dir / f'val_scatter_epoch_{epoch+1:04d}.png'
                fig.savefig(scatter_path, dpi=300, bbox_inches='tight')
                logger.info(f"✅ Saved validation scatter (peak-level) to: {scatter_path}")
                plt.close(fig)

            except Exception as e:
                logger.warning(f"⚠️ 验证散点图保存失败 (epoch {epoch+1}): {e}")
                import traceback
                logger.warning(traceback.format_exc())
                plt.close('all')  # 确保清理

        
        # 更新学习率
        scheduler.step()
        
        # ========== 模型保存与早停逻辑 ==========
        # 使用Pearson相关系数作为主要判断指标（越大越好）
        # early_stopping_min_delta: 只有改善超过这个阈值才认为是"真正的改善"
        # 这样可以避免因微小的随机波动而保存过多的模型
        if val_pearson > best_val_pearson + early_stopping_min_delta:
            # 发现更好的模型，更新最佳指标
            best_val_pearson = val_pearson
            best_val_mae = val_mae
            patience_counter = 0  # 重置耐心计数器
            
            # 确保输出目录存在
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存完整的checkpoint（用于恢复训练和推理）
            torch.save({
                'epoch': epoch,  # 当前训练轮数
                'model_state_dict': model.state_dict(),  # 模型参数
                'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态
                'val_pearson': val_pearson,  # 验证集Pearson相关系数
                'val_spearman': val_spearman,  # 验证集Spearman相关系数
                'val_r2': val_r2,  # 验证集R²
                'val_mae': val_mae,  # 验证集MAE
                'val_loss': val_loss,  # 验证集损失
                'config': config  # 完整配置（用于恢复实验）
            }, output_dir / 'best_model.pth')
            
            # 根据使用哪个级别指标显示不同的日志
            logger.info(f"✨ Epoch {epoch+1}: 保存最佳模型 (peak-level 监控)")
            logger.info(f"   Pearson r: {val_pearson:.6f} | Spearman ρ: {val_spearman:.6f}")
            logger.info(f"   MAE: {val_mae:.6f} | R²: {val_r2:.6f}")
        else:
            # 模型没有改善，增加耐心计数器
            patience_counter += 1
            
            # ========== 早停判断 ==========
            # 如果连续N轮（early_stopping_patience）都没有明显改善，
            # 则提前停止训练（避免过拟合和资源浪费）
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                logger.info(f"⛔ 早停触发！连续 {patience_counter} 轮Pearson r无改善，停止训练")
                logger.info(f"📊 最佳验证指标: Pearson r={best_val_pearson:.6f}, MAE={best_val_mae:.6f}")
                
                # 早停是正常结束（不是中断），标记为完成
                training_completed = True
                break
        
        # 记录历史
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        train_maes.append(avg_train_mae)
        val_maes.append(val_mae)
        val_pearsons.append(val_pearson)
        val_spearmans.append(val_spearman)
        
        # 增强的epoch日志输出 - 分类清晰展示
        peak_count = val_eval['peak'].get('count', 0)
        gene_count = val_eval['gene'].get('count', 0)
        gene_pearson = val_eval['gene']['metrics'].get('pearson_r', float('nan')) if val_eval['gene']['metrics'] else float('nan')
        logger.info(
            f"Epoch {epoch+1}/{config.training.max_epochs} | train_loss={avg_train_loss:.6f} | "
            f"train_mae={avg_train_mae:.6f} | val_peak_r={val_pearson:.6f} | "
            f"val_peak_mae={val_mae:.6f} | val_peak_r2={val_r2:.6f} | "
            f"val_gene_r={gene_pearson:.6f} | n_peak={peak_count:,} | n_gene={gene_count:,} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f} | early_stop={patience_counter}/{early_stopping_patience or 'N/A'}"
        )
    
    # 测试阶段：同时获得 peak 级（带meta）与 gene 级评估
    logger.info("开始测试阶段...")
    test_eval_peak = evaluate_model_with_meta(model, test_loader, device, logger, "test")
    test_loss, test_mae, test_slope, test_intercept, test_preds, test_targets, test_p, \
        test_peak_indices, test_sample_indices, test_strands = test_eval_peak
    test_eval_gene = evaluate_model_gene_level(model, test_loader, device, logger, "test")
    
    # 保存最终结果
    try:
        # 尝试获取peak_ids和sample_ids（用于CSV）
        try:
            peak_ids_for_csv = None
            sample_ids_for_csv = None
            if isinstance(dataset, YeastPeakSingleDataset):
                peak_ids_for_csv = dataset.peak_ids
                sample_ids_for_csv = dataset.sample_ids
            elif hasattr(dataset, 'peak_ids'):
                # 多数据集或过滤后的数据集
                peak_ids_for_csv = dataset.peak_ids
                sample_ids_for_csv = getattr(dataset, 'sample_ids', None)
            # 更新全局变量，便于中断路径使用
            global current_peak_ids
            current_peak_ids = peak_ids_for_csv
        except Exception:
            peak_ids_for_csv = None
            sample_ids_for_csv = None

        save_final_results(output_dir, train_losses, val_losses, train_maes, val_maes,
                          test_loss, test_mae, test_slope, test_intercept, test_preds, test_targets,
                          experiment_config, test_p, val_pearsons, val_spearmans, lr_history,
                          test_peak_indices=test_peak_indices, test_sample_indices=test_sample_indices,
                          test_strands=test_strands, test_peak_ids=peak_ids_for_csv,
                          test_sample_ids=sample_ids_for_csv,
                          test_gene_detail=test_eval_gene.get('gene', {}).get('detail'),
                          test_gene_metrics=test_eval_gene.get('gene', {}).get('metrics'))
        logger.info("✅ 最终结果保存成功")
    except Exception as e:
        logger.error(f"❌ 保存最终结果失败: {e}")
        import traceback
        logger.error(f"详细错误信息:\n{traceback.format_exc()}")
        logger.warning("⚠️ 尝试保存部分结果...")
        # 尝试至少保存预测结果CSV
        try:
            # 最低限度保存：不含peak_id的扁平结果
            df_test = pd.DataFrame({'pred': test_preds, 'true': test_targets, 'split': 'test'})
            df_test.to_csv(output_dir / 'test_predictions.csv', index=False)
            logger.info("✅ 已保存测试预测结果CSV")
        except Exception as e2:
            logger.error(f"❌ 连CSV也保存失败: {e2}")
    
    # 训练结束后保存详细验证历史表
    try:
        if val_history_rows:
            val_hist_df = pd.DataFrame(val_history_rows)
            val_hist_df.to_csv(output_dir / 'validation_history_detailed.csv', index=False)
            logger.info(f"✅ 已保存详细验证历史: {output_dir / 'validation_history_detailed.csv'}")
    except Exception as e:
        logger.warning(f"⚠️ 验证历史保存失败: {e}")

    # 记录超参数和最终指标到TensorBoard
    try:
        # 获取最终的测试指标（优先使用gene级，否则使用peak级）
        if test_eval_gene['gene']['metrics']:
            metrics_dict = {
                'hparam/test_gene_pearson_r': test_eval_gene['gene']['metrics'].get('pearson_r', float('nan')),
                'hparam/test_gene_mae': test_eval_gene['gene']['metrics'].get('mae', float('nan')),
                'hparam/test_gene_r2': test_eval_gene['gene']['metrics'].get('r2', float('nan')),
            }
        else:
            metrics_dict = {
                'hparam/test_peak_pearson_r': test_eval_gene['peak']['metrics'].get('pearson_r', float('nan')),
                'hparam/test_peak_mae': test_eval_gene['peak']['metrics'].get('mae', float('nan')),
                'hparam/test_peak_r2': test_eval_gene['peak']['metrics'].get('r2', float('nan')),
            }
        
        # 添加验证集最佳指标
        metrics_dict['hparam/best_val_pearson_r'] = best_val_pearson
        metrics_dict['hparam/best_val_mae'] = best_val_mae
        
        # 添加到TensorBoard（需要将hyperparams转换为扁平字典）
        flat_hparams = {k: v for k, v in hyperparams.items() if v is not None}
        writer.add_hparams(flat_hparams, metrics_dict)
        logger.info(f"✅ 超参数和指标已记录到 TensorBoard")
    except Exception as e:
        logger.warning(f"⚠️ 记录超参数到 TensorBoard 失败: {e}")

    # 关闭TensorBoard writer
    writer.close()
    logger.info(f"TensorBoard日志已保存到: {tensorboard_dir}")
    
    # 标记训练正常完成（如果还没标记的话）
    if not training_completed:
        training_completed = True
    
    logger.info("训练完成！")
    
    return {
        'best_val_mae': best_val_mae,
        'test_mae': test_mae,
        'test_loss': test_loss,
        'output_dir': output_dir
    }

def save_final_results(output_dir, train_losses, val_losses, train_maes, val_maes,
                      test_loss, test_mae, test_slope, test_intercept, test_preds, test_targets,
                      experiment_config, test_p=None, val_pearsons=None, val_spearmans=None, lr_history=None,
                      test_peak_indices=None, test_sample_indices=None, test_strands=None, test_peak_ids=None,
                      test_sample_ids=None, test_gene_detail=None, test_gene_metrics=None):
    """保存最终结果（按原文：仅拼接正负链的混合指标与图）- 增强版，每个绘图块独立处理"""
    # 获取logger（避免局部未定义）
    logger = logging.getLogger(__name__)
    
    # 数据保护：确保所有输入都是可用的列表/数组
    def safe_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return list(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        return [x]
    
    train_losses = safe_list(train_losses)
    val_losses = safe_list(val_losses)
    train_maes = safe_list(train_maes)
    val_maes = safe_list(val_maes)
    val_pearsons = safe_list(val_pearsons)
    val_spearmans = safe_list(val_spearmans)
    lr_history = safe_list(lr_history)
    test_preds = safe_list(test_preds)
    test_targets = safe_list(test_targets)
    
    # 计算测试集指标
    from scipy.stats import spearmanr, pearsonr
    test_pearson, _ = pearsonr(test_targets, test_preds)
    test_spearman, _ = spearmanr(test_targets, test_preds)
    test_r2 = r2_score(test_targets, test_preds)
    test_mse = mean_squared_error(test_targets, test_preds)
    test_rmse = np.sqrt(test_mse)
    
    # 计算额外统计指标
    test_median_ae = np.median(np.abs(np.array(test_targets) - np.array(test_preds)))
    test_mape = np.mean(np.abs((np.array(test_targets) - np.array(test_preds)) / (np.array(test_targets) + 1e-8))) * 100
    
    # 分层分析
    median_val = np.median(test_targets)
    high_expr_mask = np.array(test_targets) > median_val
    low_expr_mask = np.array(test_targets) <= median_val
    high_expr_mae = mean_absolute_error(np.array(test_targets)[high_expr_mask], np.array(test_preds)[high_expr_mask])
    low_expr_mae = mean_absolute_error(np.array(test_targets)[low_expr_mask], np.array(test_preds)[low_expr_mask])
    high_expr_r, _ = pearsonr(np.array(test_targets)[high_expr_mask], np.array(test_preds)[high_expr_mask])
    low_expr_r, _ = pearsonr(np.array(test_targets)[low_expr_mask], np.array(test_preds)[low_expr_mask])
    
    # 保存预测结果CSV（带peak_id/strand）
    try:
        if test_peak_indices is not None and test_sample_indices is not None and test_strands is not None:
            # 构建带元信息的表
            n = len(test_preds)
            # 安全长度对齐
            min_len = min(n, len(test_peak_indices), len(test_sample_indices), len(test_strands))
            preds_np = np.array(test_preds)[:min_len]
            trues_np = np.array(test_targets)[:min_len]
            peak_idx_np = np.array(test_peak_indices)[:min_len]
            sample_idx_np = np.array(test_sample_indices)[:min_len]
            strand_np = np.array(test_strands)[:min_len]
            # 映射peak_id
            if test_peak_ids is not None:
                try:
                    peak_id_vals = [str(test_peak_ids[int(i)]) if int(i) < len(test_peak_ids) else '' for i in peak_idx_np]
                except Exception:
                    peak_id_vals = [''] * min_len
            else:
                peak_id_vals = [''] * min_len
            
            # 映射sample_id
            if test_sample_ids is not None:
                try:
                    sample_id_vals = [str(test_sample_ids[int(i)]) if int(i) < len(test_sample_ids) else str(i) for i in sample_idx_np]
                except Exception:
                    sample_id_vals = [str(i) for i in sample_idx_np]
            else:
                sample_id_vals = [str(i) for i in sample_idx_np]
            
            strand_str = np.where(strand_np == 0, 'pos', 'neg')
            df_test = pd.DataFrame({
                'sample_idx': sample_idx_np,
                'sample_id': sample_id_vals,
                'peak_idx': peak_idx_np,
                'peak_id': peak_id_vals,
                'strand': strand_str,
                'true': trues_np,
                'pred': preds_np,
                'error': preds_np - trues_np,
                'abs_error': np.abs(preds_np - trues_np),
            })
        else:
            df_test = pd.DataFrame({'pred': test_preds, 'true': test_targets, 'split': 'test'})
        df_test.to_csv(output_dir / 'test_predictions.csv', index=False)
        logger.info(f"✅ 测试预测CSV已保存: {len(df_test)} 行 (含peak_id={test_peak_ids is not None})")
    except Exception as e:
        logger.error(f"❌ 保存测试预测CSV失败: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # 额外保存 gene 级预测结果 CSV（若提供）
    try:
        if test_gene_detail and test_gene_detail.get('pred') is not None and test_gene_detail.get('true') is not None:
            g_pred = np.array(test_gene_detail['pred'])
            g_true = np.array(test_gene_detail['true'])
            g_gene_ids = test_gene_detail.get('gene_ids')
            g_strands = test_gene_detail.get('strands')
            g_samples = test_gene_detail.get('sample_indices')
            df = {
                'true': g_true,
                'pred': g_pred,
                'error': g_pred - g_true,
                'abs_error': np.abs(g_pred - g_true)
            }
            if g_gene_ids is not None:
                df['gene_id'] = g_gene_ids
            if g_strands is not None:
                df['strand'] = g_strands
            if g_samples is not None:
                df['sample_idx'] = g_samples
            df_gene = pd.DataFrame(df)
            df_gene.to_csv(output_dir / 'test_gene_predictions.csv', index=False)
            logger.info(f"✅ 已保存基因级预测CSV: {len(df_gene)} 行")
    except Exception as e:
        logger.error(f"❌ 保存基因级预测CSV失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # 生成基因级散点图（若提供 gene 级明细与指标）
    try:
        if test_gene_detail and test_gene_detail.get('pred') is not None and test_gene_detail.get('true') is not None:
            g_pred = np.array(test_gene_detail['pred'])
            g_true = np.array(test_gene_detail['true'])
            lo, hi = 0.0, 20.0
            max_points = 100000
            fig, ax = plt.subplots(figsize=(10, 10))
            if len(g_true) > max_points:
                idx = np.random.choice(len(g_true), max_points, replace=False)
                x_plot = g_true[idx]
                y_plot = g_pred[idx]
            else:
                x_plot = g_true
                y_plot = g_pred
            ax.scatter(x_plot, y_plot, s=1.5, alpha=0.15, c='#2a5599', edgecolors='none', zorder=1)
            ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1, label='y=x', zorder=10)
            # 指标文本
            gp = (test_gene_metrics or {}).get('pearson_r', np.nan)
            gs = (test_gene_metrics or {}).get('spearman_rho', np.nan)
            gr2 = (test_gene_metrics or {}).get('r2', np.nan)
            gmae = (test_gene_metrics or {}).get('mae', np.nan)
            ax.set_xlabel('True gene expression', fontsize=14)
            ax.set_ylabel('Predicted gene expression', fontsize=14)
            ax.set_title(f'[GENE-LEVEL] Test (gene-level)\nPearson r={gp:.4f}, Spearman ρ={gs:.4f}, R²={gr2:.4f}, MAE={gmae:.4f}', fontsize=16)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.tick_params(axis='both', labelsize=12)
            ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            out_path = output_dir / 'test_gene_evaluation.png'
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ 基因级散点图已保存: {out_path}")
            plt.close(fig)
    except Exception as e:
        logger.warning(f"⚠️ 基因级散点图保存失败: {e}")

    # 保存训练历史 - 更稳健的对齐策略（仅包含长度>=min_len的序列）
    try:
        raw_series = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_mae': train_maes,
            'val_mae': val_maes,
            'val_pearson': val_pearsons,
            'val_spearman': val_spearmans,
            'lr': lr_history,
        }

        # 仅保留非空序列
        non_empty = {k: list(v) for k, v in raw_series.items() if v and len(v) > 0}

        if not non_empty:
            logger.warning("⚠️ 训练历史为空，跳过 training_history.csv 保存")
        else:
            # 计算最小长度
            min_len = min(len(v) for v in non_empty.values())
            # 仅包含长度 >= min_len 的键，并统一截断到 min_len
            aligned = {k: v[:min_len] for k, v in non_empty.items() if len(v) >= min_len}
            aligned['epoch'] = list(range(1, min_len + 1))
            train_history = pd.DataFrame(aligned)
            train_history.to_csv(output_dir / 'training_history.csv', index=False)
            logger.info(f"✅ 训练历史已保存（{min_len} 轮）: {output_dir / 'training_history.csv'}")
    except Exception as e:
        logger.error(f"❌ 保存训练历史失败（将继续保存其他结果）: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # 保存指标汇总 - 增加更多指标（不再包含 per-strand 指标）
    try:
        metrics_list = ['Peak Pearson r', 'Peak Spearman ρ', 'Peak R²', 'Peak MAE', 'Peak MSE', 'RMSE', 'Median_AE', 'MAPE(%)', 
                       'Slope', 'Intercept', 'N_samples',
                       'High_Expr_MAE', 'Low_Expr_MAE', 'High_Expr_r', 'Low_Expr_r']
        values_list = [test_pearson, test_spearman, test_r2, test_mae, test_mse, test_rmse, test_median_ae, test_mape,
                      test_slope, test_intercept, len(test_targets),
                      high_expr_mae, low_expr_mae, high_expr_r, low_expr_r]
        # 合并gene级指标（如有）
        if test_gene_metrics:
            metrics_list.extend(['Gene Pearson r', 'Gene Spearman ρ', 'Gene R²', 'Gene MAE', 'Gene MSE'])
            values_list.extend([
                test_gene_metrics.get('pearson_r', float('nan')),
                test_gene_metrics.get('spearman_rho', float('nan')),
                test_gene_metrics.get('r2', float('nan')),
                test_gene_metrics.get('mae', float('nan')),
                test_gene_metrics.get('mse', float('nan')),
            ])
        
        metrics_summary = pd.DataFrame({
            'metric': metrics_list,
            'value': values_list
        })
        metrics_summary.to_csv(output_dir / 'test_metrics_summary.csv', index=False)
        logger.info(f"✅ 测试指标汇总已保存: {len(metrics_list)} 项指标")
    except Exception as e:
        logger.error(f"❌ 保存指标汇总失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # 生成测试集散点图 - 单图（concat pos/neg）
    logger.info("📊 生成测试集散点图（concat pos/neg）...")
    try:
        max_points = 100000
        lo, hi = 0.0, 20.0

        test_targets_np = np.array(test_targets)
        test_preds_np = np.array(test_preds)

        fig, ax = plt.subplots(figsize=(10, 10))
        if len(test_targets_np) > max_points:
            idx = np.random.choice(len(test_targets_np), max_points, replace=False)
            x_plot = test_targets_np[idx]
            y_plot = test_preds_np[idx]
        else:
            x_plot = test_targets_np
            y_plot = test_preds_np

        ax.scatter(x_plot, y_plot, s=1.5, alpha=0.15, c='#1b7c3d', edgecolors='none', zorder=1)
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1, label='y=x', zorder=10)

        # 回归拟合线
        try:
            x_line = np.linspace(lo, hi, 200)
            y_line = test_slope * x_line + test_intercept
            fit_label = f'fit: y={test_slope:.3f}x+{test_intercept:.3f}'
            ax.plot(x_line, y_line, color='green', linewidth=1.2, zorder=3, label=fit_label)
        except Exception:
            pass

        ax.set_xlabel('True expression', fontsize=14)
        ax.set_ylabel('Predicted expression', fontsize=14)
        ax.set_title(f'[PEAK-LEVEL] Test Set (concat pos/neg)\nPearson r={test_pearson:.4f}, Spearman ρ={test_spearman:.4f}, R²={test_r2:.4f}', fontsize=16, fontweight='bold')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3)

        stats_text = (
            f'N = {len(test_targets):,}\n'
            f'Pearson r = {test_pearson:.4f}\n'
            f'Spearman ρ = {test_spearman:.4f}\n'
            f'R² = {test_r2:.4f}\n'
            f'MAE = {test_mae:.4f}\n'
            f'RMSE = {test_rmse:.4f}'
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11, va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0e6c8', alpha=0.9, edgecolor='#a98f5a'))

        plt.tight_layout()
        test_scatter_path = output_dir / 'test_evaluation.png'
        plt.savefig(test_scatter_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ 测试集散点图已保存: {test_scatter_path}")
        plt.close()
    except Exception as e:
        logger.error(f"❌ 测试集散点图生成失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        plt.close('all')

    
    # 生成训练历史图表 - 增强版
    logger.info("📊 生成训练历史分析图...")
    try:
        # 定义坐标范围常量
        lo, hi = 0.0, 20.0
        max_points = 100000
        
        fig = plt.figure(figsize=(20, 12))
        # 总标题：本图中的测试指标均为峰级（peak-level）
        fig.suptitle('[PEAK-LEVEL] Training analysis (test metrics are peak-level)', fontsize=14, fontweight='bold')
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 损失曲线
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
        ax1.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('Training vs. validation loss', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # MAE曲线
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(train_maes, label='Training MAE', color='blue', linewidth=2)
        ax2.plot(val_maes, label='Validation MAE', color='red', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('MAE', fontsize=11)
        ax2.set_title('MAE over epochs', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Pearson相关系数曲线
        if val_pearsons:
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.plot(val_pearsons, label='Validation Pearson r', color='green', linewidth=2)
            if val_spearmans:
                ax3.plot(val_spearmans, label='Validation Spearman ρ', color='purple', linewidth=2, linestyle='--')
            ax3.set_xlabel('Epoch', fontsize=11)
            ax3.set_ylabel('Correlation Coefficient', fontsize=11)
            ax3.set_title('Correlation over epochs', fontsize=12, fontweight='bold')
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0.9, color='orange', linestyle=':', alpha=0.5, label='Target: 0.9')
            ax3.set_ylim([max(0, min(val_pearsons) - 0.05), 1.0])
        
        # 学习率曲线
        if lr_history:
            ax4 = fig.add_subplot(gs[0, 3])
            ax4.plot(lr_history, label='Learning Rate', color='purple', linewidth=2)
            ax4.set_xlabel('Epoch', fontsize=11)
            ax4.set_ylabel('Learning Rate', fontsize=11)
            ax4.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
            ax4.legend(fontsize=10)
            ax4.grid(True, alpha=0.3)
            ax4.set_yscale('log')  # 使用对数坐标更好地显示学习率变化
        
        # 误差分布
        errors = np.array(test_preds) - np.array(test_targets)
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.hist(errors, bins=60, alpha=0.7, color='skyblue', edgecolor='black', density=True)
        ax5.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax5.axvline(np.mean(errors), color='orange', linestyle='-', linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
        ax5.axvline(np.median(errors), color='green', linestyle='-.', linewidth=2, label=f'Median: {np.median(errors):.4f}')
        ax5.set_xlabel('Prediction Error', fontsize=11)
        ax5.set_ylabel('Density', fontsize=11)
        ax5.set_title('Prediction error distribution (normalized)', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)
        
        # 预测密度图（测试集）
        ax6 = fig.add_subplot(gs[1, 1])
        hb = ax6.hexbin(test_targets, test_preds, gridsize=40, cmap='YlOrRd', alpha=0.85, mincnt=1)
        ax6.plot([0, 20], [0, 20], 'b--', linewidth=2, label='y=x')
        cb = plt.colorbar(hb, ax=ax6)
        cb.set_label('Density', fontsize=10)
        ax6.set_xlabel('True expression (log2)', fontsize=11)
        ax6.set_ylabel('Predicted expression (log2)', fontsize=11)
        ax6.set_title('Prediction density (test set)', fontsize=12, fontweight='bold')
        ax6.set_xlim(0, 20)
        ax6.set_ylim(0, 20)
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)
        
        # 残差图
        ax7 = fig.add_subplot(gs[1, 2])
        ax7.scatter(test_targets, errors, alpha=0.3, s=2, c='purple', edgecolors='none')
        ax7.axhline(0, color='red', linestyle='--', linewidth=2)
        ax7.set_xlabel('True expression', fontsize=11)
        ax7.set_ylabel('Residual (pred - true)', fontsize=11)
        ax7.set_title('Residuals', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        
        # QQ图（检查误差正态性）
        ax8 = fig.add_subplot(gs[2, 0])
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=ax8)
        ax8.set_title('Q-Q plot (residual normality)', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3)
        
        # 累积误差分布
        ax9 = fig.add_subplot(gs[2, 1])
        sorted_abs_errors = np.sort(np.abs(errors))
        cumulative = np.arange(1, len(sorted_abs_errors) + 1) / len(sorted_abs_errors)
        ax9.plot(sorted_abs_errors, cumulative, linewidth=2, color='darkblue')
        ax9.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50% Quantile')
        ax9.axhline(0.9, color='orange', linestyle='--', alpha=0.5, label='90% Quantile')
        ax9.set_xlabel('Absolute error', fontsize=11)
        ax9.set_ylabel('Cumulative probability', fontsize=11)
        ax9.set_title('Cumulative error distribution', fontsize=12, fontweight='bold')
        ax9.legend(fontsize=9)
        ax9.grid(True, alpha=0.3)
        
        # 表达值分布对比
        ax10 = fig.add_subplot(gs[2, 2])
        ax10.hist(test_targets, bins=50, alpha=0.6, color='blue', label='True Values', density=True)
        ax10.hist(test_preds, bins=50, alpha=0.6, color='red', label='Predicted Values', density=True)
        ax10.set_xlabel('Expression', fontsize=11)
        ax10.set_ylabel('Density', fontsize=11)
        ax10.set_title('Distribution: true vs. predicted', fontsize=12, fontweight='bold')
        ax10.legend(fontsize=10)
        ax10.grid(True, alpha=0.3)
        
        # 为总标题留白
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_dir / 'training_analysis.png', dpi=150, bbox_inches='tight')
        logger.info(f"✅ 训练分析图已保存: {output_dir / 'training_analysis.png'}")
        plt.close()

        # 额外保存独立的学习率曲线图，便于快速查看LR日程
        try:
            if lr_history and len(lr_history) > 0:
                plt.figure(figsize=(7, 4))
                plt.plot(lr_history, color='purple', linewidth=2)
                plt.yscale('log')
                plt.xlabel('Epoch')
                plt.ylabel('Learning Rate (log)')
                plt.title('Learning Rate Schedule')
                plt.grid(True, alpha=0.3)
                lr_png = output_dir / 'learning_rate.png'
                plt.tight_layout()
                plt.savefig(lr_png, dpi=200, bbox_inches='tight')
                logger.info(f"✅ 学习率曲线已单独保存: {lr_png}")
                plt.close()
        except Exception as e:
            logger.warning(f"⚠️ 学习率曲线单图保存失败: {e}")
    except Exception as e:
        logger.error(f"❌ 训练分析图生成失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        plt.close('all')
    
    # 生成训练报告 - 增强版
    report_path = output_dir / 'training_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 🧬 单Peak训练报告（增强版）\n\n")
        f.write(f"**训练时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**实验名称**: {experiment_config.name}\n")
        f.write(f"**实验描述**: {experiment_config.description}\n")
        f.write(f"**训练方法**: 单Peak方法（每个peak独立训练）\n\n")
        
        f.write(f"---\n\n")
        f.write(f"## 📊 测试集核心指标\n\n")
        f.write(f"### 相关性指标（最重要）\n")
        f.write(f"- **Pearson相关系数 (r)**: {test_pearson:.6f} ⭐ 主要指标\n")
        f.write(f"- **Spearman相关系数 (ρ)**: {test_spearman:.6f}\n")
        f.write(f"- **决定系数 (R²)**: {test_r2:.6f}\n\n")
        
        f.write(f"### 误差指标\n")
        f.write(f"- **平均绝对误差 (MAE)**: {test_mae:.6f}\n")
        f.write(f"- **均方误差 (MSE)**: {test_mse:.6f}\n")
        f.write(f"- **均方根误差 (RMSE)**: {test_rmse:.6f}\n")
        f.write(f"- **中位数绝对误差**: {test_median_ae:.6f}\n")
        f.write(f"- **平均绝对百分比误差 (MAPE)**: {test_mape:.2f}%\n\n")
        
        f.write(f"### 回归分析\n")
        f.write(f"- **回归斜率**: {test_slope:.6f}\n")
        f.write(f"- **回归截距**: {test_intercept:.6f}\n")
        f.write(f"- **测试集损失**: {test_loss:.6f}\n")
        f.write(f"- **样本数**: {len(test_targets):,}\n\n")
        
        f.write(f"---\n\n")
        f.write(f"## 🎯 分层分析\n\n")
        f.write(f"**分层标准**: 以中位数 {median_val:.4f} 为界\n\n")
        f.write(f"### 高表达区域（> {median_val:.4f}）\n")
        f.write(f"- **MAE**: {high_expr_mae:.6f}\n")
        f.write(f"- **Pearson r**: {high_expr_r:.6f}\n")
        f.write(f"- **样本数**: {np.sum(high_expr_mask):,}\n\n")
        f.write(f"### 低表达区域（≤ {median_val:.4f}）\n")
        f.write(f"- **MAE**: {low_expr_mae:.6f}\n")
        f.write(f"- **Pearson r**: {low_expr_r:.6f}\n")
        f.write(f"- **样本数**: {np.sum(low_expr_mask):,}\n\n")
        
        # 移除 per-strand 分析部分，统一以 concat 指标为准
        
        f.write(f"---\n\n")
        f.write(f"## 📈 训练历史\n\n")
        f.write(f"- **总训练轮次**: {len(train_losses)}\n")
        f.write(f"- **最终训练损失**: {train_losses[-1]:.6f}\n")
        f.write(f"- **最终验证损失**: {val_losses[-1]:.6f}\n")
        f.write(f"- **最终训练MAE**: {train_maes[-1]:.6f}\n")
        f.write(f"- **最终验证MAE**: {val_maes[-1]:.6f}\n")
        if val_pearsons:
            f.write(f"- **最终验证Pearson r**: {val_pearsons[-1]:.6f}\n")
            f.write(f"- **最佳验证Pearson r**: {max(val_pearsons):.6f} (Epoch {val_pearsons.index(max(val_pearsons))+1})\n")
        if val_spearmans:
            f.write(f"- **最终验证Spearman ρ**: {val_spearmans[-1]:.6f}\n")
            f.write(f"- **最佳验证Spearman ρ**: {max(val_spearmans):.6f} (Epoch {val_spearmans.index(max(val_spearmans))+1})\n")
        f.write(f"\n")
        
        f.write(f"---\n\n")
        
        f.write(f"## 文件说明\n")
        f.write(f"- `best_model.pth`: 最佳模型权重\n")
        f.write(f"- `test_predictions.csv`: 测试集预测结果\n")
        f.write(f"- `test_evaluation.png`: 测试集散点图\n")
        f.write(f"- `training_analysis.png`: 训练分析图表（损失曲线、MAE曲线、误差分布、预测密度）\n")
        f.write(f"- `training_history.csv`: 训练历史数据\n")
        f.write(f"- `test_metrics_summary.csv`: 测试指标汇总\n")
        f.write(f"- `tensorboard_logs/`: TensorBoard日志目录\n")
        f.write(f"- `train_single_peak.log`: 训练日志\n")

@hydra.main(version_base=None, config_path="get_model/config", config_name="yeast_training_km")
def main(config: DictConfig):
    """主函数：运行ATAC单Peak训练"""
    
    logger = logging.getLogger(__name__)
    logger.info("开始ATAC单Peak训练")
    logger.info(f"配置路径: {config}")
    
    # 从YAML配置中获取数据路径和训练配置
    input_files = config.data.input_files
    output_base_dir = config.data.output_base_dir
    training_name = config.training_name
    training_description = config.training_description
    output_dir = config.output_dir
    
    # 检查输入文件（KM酵母多数据集：要求五份数据一起训练）
    data_paths = []
    if hasattr(input_files, 'keys'):
        input_keys = list(input_files.keys())
    else:
        input_keys = list(input_files.__dict__.keys())

    # 强制要求五份数据集并包含 ATAC1
    expected_keys = {'c1', 'c3', 'o2', 'o3', 'atac1'}
    missing_keys = expected_keys - set(input_keys)
    if missing_keys:
        logger.error(f"缺少必要数据集键: {sorted(list(missing_keys))}，当前仅有: {input_keys}")
        raise ValueError("KM训练需同时提供 c1/c3/o2/o3/atac1 五份数据")
    if len(input_keys) != 5:
        logger.error(f"当前输入文件数量为 {len(input_keys)}，期望 5 个 (c1/c3/o2/o3/atac1)")
        raise ValueError("KM训练需同时使用五份数据集")
    for sample_name in input_keys:
        path = getattr(input_files, sample_name)
        if not os.path.exists(path):
            logger.error(f"{sample_name.upper()}文件不存在: {path}")
            raise FileNotFoundError(f"{sample_name.upper()}文件缺失")
        data_paths.append(path)
        logger.info(f"找到数据文件 {sample_name.upper()}: {path}")
    
    logger.info(f"总计使用 {len(data_paths)} 个数据文件")
    
    # 创建输出基础目录
    Path(output_base_dir).mkdir(parents=True, exist_ok=True)
    
    # 创建训练配置对象
    training_config = type('TrainingConfig', (), {
        'name': training_name,
        'description': training_description,
        'output_dir': output_dir,
        'input_files': input_keys
    })()
    
    # 运行训练
    try:
        logger.info(f"开始训练: {training_name}")
        result = train_experiment('atac_training', training_config, config)
        logger.info("训练完成")
        return result
    except Exception as e:
        logger.error(f"训练失败: {e}")
        raise

if __name__ == "__main__":
    main()
