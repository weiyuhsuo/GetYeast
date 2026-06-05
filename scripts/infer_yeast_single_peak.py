#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理脚本 - 单Peak模型推理
用于加载训练好的模型对新的数据进行预测

使用方法:
    直接修改脚本开头的配置参数，然后运行:
    python infer_yeast_single_peak.py
    
    或者使用命令行参数覆盖:
    python infer_yeast_single_peak.py --data_path input/KM/matrix_C1.csv.npz

说明:
    - 推理时不需要yaml配置文件，模型配置会从checkpoint中自动读取
    - yaml文件主要用于训练阶段，推理时所有参数都在脚本开头配置
    - 如果设置了CONFIG_PATH，会作为备选配置（优先级低于脚本配置）
"""

# ============================================================================
# 📋 配置参数（请在此处修改）
# ============================================================================
# 说明：以下参数可以直接在脚本中修改，优先级高于配置文件和命令行参数

# 数据文件路径（支持单个文件或文件列表）
# 单个文件：使用字符串，例如: DATA_PATHS = 'input/KM/matrix_C1.csv.npz'
# 多个文件：使用列表，例如: DATA_PATHS = ['input/KM/matrix_C1.csv.npz', 'input/KM/matrix_C3.csv.npz']
DATA_PATHS = [
    'input/260511/LEM3_31x15_from_matrix511.npz',  # 260511 新输入（推理）
]

# 模型checkpoint路径（相对路径或绝对路径）
MODEL_PATH = 'output/sc_atac_single_peak_training_20251209_142840/best_model.pth'

# 批次大小（推理时建议使用较小的batch_size，如512或更小）
BATCH_SIZE = 512

# 计算设备: 'auto'（自动检测）/ 'cuda' / 'cpu'
DEVICE = 'auto'

# 输出基础目录（推理结果会保存在此目录下的带时间戳的子目录中）
OUTPUT_BASE_DIR = 'output'

# 配置文件路径（可选，推理时通常不需要，模型配置会从checkpoint中读取）
# 设为None则不使用配置文件，所有参数使用脚本中的配置
CONFIG_PATH = None  # 或设为 'get_model/config/yeast_training_km.yaml' 作为备选配置

# ============================================================================
# 导入库
# ============================================================================
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import os
import datetime
import yaml
from pathlib import Path
from tqdm import tqdm
import sys
import logging

# 添加模型路径到sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from get_model.model.yeast_model import YeastModel
from omegaconf import DictConfig, OmegaConf

def load_model(checkpoint_path: str, device: torch.device):
    """
    加载训练好的模型
    
    Args:
        checkpoint_path: 模型checkpoint路径
        device: 计算设备
    
    Returns:
        model: 加载好的模型
        config: 模型配置
    """
    logger = logging.getLogger(__name__)
    logger.info(f"加载模型: {checkpoint_path}")
    
    # 加载checkpoint
    # 注意：PyTorch 2.6默认weights_only=True，但checkpoint包含DictConfig等对象，需要设置为False
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 获取配置（checkpoint中保存了完整配置）
    if 'config' in checkpoint:
        config = checkpoint['config']
        # 如果是DictConfig，转换为字典
        if isinstance(config, DictConfig):
            config_dict = OmegaConf.to_container(config, resolve=True)
        else:
            config_dict = config
        
        # 创建模型
        model_cfg = config_dict['model']['model']
        use_lora = config_dict['training'].get('use_lora', False)
        lora_rank = config_dict['training'].get('lora_rank', 4)
        lora_alpha = config_dict['training'].get('lora_alpha', 16)
        lora_layers = config_dict['training'].get('lora_layers', None)
        
        model = YeastModel(
            cfg=model_cfg,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_layers=lora_layers
        )
    else:
        raise ValueError("Checkpoint中未找到config，无法创建模型")
    
    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("✅ 成功加载模型权重")
    else:
        raise ValueError("Checkpoint中未找到model_state_dict")
    
    # 移动到指定设备并设置为评估模式
    model = model.to(device)
    model.eval()
    
    # 打印模型信息
    if 'val_pearson' in checkpoint:
        logger.info(f"模型验证指标: Pearson r = {checkpoint['val_pearson']:.4f}")
    if 'val_mae' in checkpoint:
        logger.info(f"模型验证指标: MAE = {checkpoint['val_mae']:.4f}")
    
    return model, config_dict

def load_data(npz_path: str):
    """
    加载npz数据文件
    
    Args:
        npz_path: npz文件路径
    
    Returns:
        data: 数据数组 (samples, peaks, features)
        peak_ids: peak ID列表
    """
    logger = logging.getLogger(__name__)
    logger.info(f"加载数据: {npz_path}")
    
    # 检查文件是否存在
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"数据文件不存在: {npz_path}")
    
    # 检查文件大小
    file_size = os.path.getsize(npz_path)
    if file_size == 0:
        raise ValueError(f"数据文件为空: {npz_path}")
    logger.info(f"文件大小: {file_size / (1024*1024):.2f} MB")
    
    # 尝试加载文件（先尝试不使用mmap，如果失败再尝试mmap）
    npz_file = None
    try:
        # 方法1: 不使用mmap（更稳定，适合网络文件系统）
        npz_file = np.load(npz_path, allow_pickle=True)
        logger.info("使用标准模式加载文件（非mmap）")
    except EOFError:
        # 如果标准模式失败，尝试mmap模式
        try:
            logger.warning("标准模式加载失败，尝试mmap模式...")
            npz_file = np.load(npz_path, mmap_mode='r', allow_pickle=True)
            logger.info("使用mmap模式加载文件")
        except EOFError as e:
            raise ValueError(f"数据文件损坏或格式错误: {npz_path}\n"
                            f"错误详情: {e}\n"
                            f"可能原因: 文件传输不完整、网络文件系统问题、或文件损坏\n"
                            f"建议: 1) 检查文件MD5值 2) 重新复制文件 3) 重新生成npz文件")
        except Exception as e:
            raise ValueError(f"无法加载数据文件: {npz_path}\n"
                            f"错误详情: {e}")
    except Exception as e:
        raise ValueError(f"无法加载数据文件: {npz_path}\n"
                        f"错误详情: {e}")
    
    # 读取数据
    data = npz_file['data']
    
    # 检查数据是否为空
    if data is None:
        raise ValueError(f"数据文件中的'data'键为空: {npz_path}")
    
    # 获取数据形状
    if isinstance(data, np.ndarray):
        num_samples, num_peaks, num_features = data.shape
    else:
        # 如果是mmap，需要先获取形状
        num_samples, num_peaks, num_features = data.shape
    
    logger.info(f"数据形状: ({num_samples}, {num_peaks}, {num_features})")
    
    # 检查数据维度是否有效
    if num_samples == 0:
        raise ValueError(f"数据样本数为0: {npz_path}")
    if num_peaks == 0:
        raise ValueError(f"数据peaks数为0: {npz_path}")
    if num_features == 0:
        raise ValueError(f"数据特征数为0: {npz_path}")
    
    # 检查数据内容（采样检查，避免加载全部数据到内存）
    logger.info("检查数据内容...")
    try:
        # 检查第一个样本的第一个peak的数据
        sample_data = data[0, 0, :]
        non_zero_count = np.count_nonzero(sample_data)
        nan_count = np.isnan(sample_data).sum()
        inf_count = np.isinf(sample_data).sum()
        
        logger.info(f"  第一个样本第一个peak: 非零值={non_zero_count}/{len(sample_data)}, NaN={nan_count}, Inf={inf_count}")
        
        # 随机采样几个位置检查
        if num_samples > 0 and num_peaks > 0:
            check_indices = [
                (0, 0),
                (min(num_samples//2, num_samples-1), min(num_peaks//2, num_peaks-1)),
                (num_samples-1, num_peaks-1)
            ]
            for s_idx, p_idx in check_indices:
                if s_idx < num_samples and p_idx < num_peaks:
                    check_data = data[s_idx, p_idx, :]
                    check_non_zero = np.count_nonzero(check_data)
                    check_nan = np.isnan(check_data).sum()
                    logger.info(f"  样本[{s_idx}]峰值[{p_idx}]: 非零值={check_non_zero}/{len(check_data)}, NaN={check_nan}")
    except Exception as e:
        logger.warning(f"⚠️ 数据内容检查时出错: {e}，继续处理...")
    
    # 读取peak_ids
    try:
        peak_ids = npz_file['peak_ids']
        if hasattr(peak_ids, 'tolist'):
            peak_ids = peak_ids.tolist()
        logger.info(f"✅ 成功加载peak_ids: {len(peak_ids)} 个peaks")
    except Exception:
        peak_ids = [f"peak_{i}" for i in range(num_peaks)]
        logger.warning("⚠️ 未找到peak_ids，使用默认命名")

    try:
        sample_ids = npz_file['sample_ids']
        if hasattr(sample_ids, 'tolist'):
            sample_ids = sample_ids.tolist()
        logger.info(f"✅ 成功加载sample_ids: {len(sample_ids)} 个样本")
    except Exception:
        sample_ids = [str(i) for i in range(num_samples)]
        logger.warning("⚠️ 未找到sample_ids，使用顺序编号")
    
    # 验证数据格式
    if num_features == 547:
        # 标准格式：545特征 + 2标签
        feature_dim = 545
        label_pos_idx = 545
        label_neg_idx = 546
        logger.info(f"✅ 数据格式正确: 547列 = 545特征 + 2标签")
    elif num_features == 545:
        # 只有特征，没有标签（推理场景）
        feature_dim = 545
        label_pos_idx = None
        label_neg_idx = None
        logger.info(f"✅ 数据格式: 545列（仅特征，无标签）")
    else:
        raise ValueError(f"数据维度不匹配: 期望547或545，实际={num_features}")
    
    return {
        'data': data,
        'peak_ids': peak_ids,
        'sample_ids': sample_ids,
        'feature_dim': feature_dim,
        'label_pos_idx': label_pos_idx,
        'label_neg_idx': label_neg_idx,
        'num_samples': num_samples,
        'num_peaks': num_peaks
    }

def predict(model, data_info, device, batch_size=32):
    """
    使用模型进行预测
    
    Args:
        model: 训练好的模型
        data_info: 数据信息字典
        device: 计算设备
        batch_size: batch大小
    
    Returns:
        predictions: 预测结果 (samples, peaks, 2)
        true_labels: 真实标签 (samples, peaks, 2) 或 None（如果数据中没有标签）
    """
    logger = logging.getLogger(__name__)
    logger.info("开始推理...")
    
    data = data_info['data']
    num_samples = data_info['num_samples']
    num_peaks = data_info['num_peaks']
    feature_dim = data_info['feature_dim']
    label_pos_idx = data_info.get('label_pos_idx', None)
    label_neg_idx = data_info.get('label_neg_idx', None)
    
    # 验证输入数据
    logger.info(f"输入数据: {num_samples} 样本 × {num_peaks} peaks × {feature_dim} 特征")
    if num_samples == 0 or num_peaks == 0:
        raise ValueError(f"数据为空: num_samples={num_samples}, num_peaks={num_peaks}")
    
    # 检查第一个样本的数据
    try:
        first_sample_first_peak = data[0, 0, :feature_dim]
        logger.info(f"第一个样本第一个peak特征统计: min={np.min(first_sample_first_peak):.4f}, "
                   f"max={np.max(first_sample_first_peak):.4f}, "
                   f"mean={np.mean(first_sample_first_peak):.4f}, "
                   f"非零值={np.count_nonzero(first_sample_first_peak)}/{len(first_sample_first_peak)}")
    except Exception as e:
        logger.warning(f"⚠️ 无法检查数据统计: {e}")
    
    # 初始化预测结果数组
    predictions = np.zeros((num_samples, num_peaks, 2), dtype=np.float32)
    logger.info(f"初始化预测结果数组: shape={predictions.shape}")
    
    # 提取真实标签（如果存在）
    true_labels = None
    if label_pos_idx is not None and label_neg_idx is not None:
        logger.info("检测到数据中包含真实标签，提取真实值...")
        true_labels = np.zeros((num_samples, num_peaks, 2), dtype=np.float32)
        for sample_idx in range(num_samples):
            for peak_idx in range(num_peaks):
                true_labels[sample_idx, peak_idx, 0] = float(data[sample_idx, peak_idx, label_pos_idx])
                true_labels[sample_idx, peak_idx, 1] = float(data[sample_idx, peak_idx, label_neg_idx])
        
        # 验证真实标签
        non_zero_labels = np.count_nonzero(true_labels)
        total_labels = true_labels.size
        logger.info(f"  真实标签: 总数={total_labels}, 非零值={non_zero_labels}, "
                   f"非零比例={non_zero_labels/total_labels*100:.2f}%")
        logger.info(f"  真实值范围: min={np.min(true_labels):.6f}, max={np.max(true_labels):.6f}, "
                   f"mean={np.mean(true_labels):.6f}")
    else:
        logger.info("数据中不包含真实标签（仅特征，无标签）")
    
    model.eval()
    with torch.no_grad():
        # 遍历所有样本和peaks
        total_batches = 0
        for sample_idx in tqdm(range(num_samples), desc="推理进度"):
            # 收集当前样本的所有peaks
            sample_features = []
            peak_indices = []
            
            for peak_idx in range(num_peaks):
                # 提取特征（前feature_dim列）
                features = data[sample_idx, peak_idx, :feature_dim]
                sample_features.append(features)
                peak_indices.append(peak_idx)
            
            # 批量处理
            for batch_start in range(0, len(sample_features), batch_size):
                batch_end = min(batch_start + batch_size, len(sample_features))
                batch_features = sample_features[batch_start:batch_end]
                
                # 转换为tensor
                batch_tensor = torch.tensor(
                    np.array(batch_features), 
                    dtype=torch.float32
                ).unsqueeze(1)  # (batch_size, 1, feature_dim)
                
                # 移动到设备
                batch_tensor = batch_tensor.to(device)
                
                # 模型预测
                inputs = {'motif_features': batch_tensor}
                outputs = model(inputs)  # (batch_size, 1, 2)
                
                # 保存预测结果
                outputs_np = outputs.squeeze(1).cpu().numpy()  # (batch_size, 2)
                
                # 检查输出是否有效
                if np.any(np.isnan(outputs_np)) or np.any(np.isinf(outputs_np)):
                    logger.warning(f"⚠️ 样本[{sample_idx}] batch[{batch_start}:{batch_end}] 预测结果包含NaN或Inf")
                
                for i, peak_idx in enumerate(peak_indices[batch_start:batch_end]):
                    predictions[sample_idx, peak_idx] = outputs_np[i]
                
                total_batches += 1
                
                # 每处理100个batch打印一次进度
                if total_batches % 100 == 0:
                    logger.info(f"  已处理 {total_batches} 个batches, 当前样本: {sample_idx+1}/{num_samples}")
    
    # 验证预测结果
    logger.info("验证预测结果...")
    non_zero_preds = np.count_nonzero(predictions)
    total_preds = predictions.size
    logger.info(f"  预测结果: 总数={total_preds}, 非零值={non_zero_preds}, "
               f"非零比例={non_zero_preds/total_preds*100:.2f}%")
    logger.info(f"  预测值范围: min={np.min(predictions):.6f}, max={np.max(predictions):.6f}, "
               f"mean={np.mean(predictions):.6f}")
    
    if non_zero_preds == 0:
        logger.warning("⚠️ 警告: 所有预测结果都为0！")
    
    logger.info(f"✅ 推理完成，共处理 {total_batches} 个batches")
    return predictions, true_labels

def save_results(predictions, data_info, output_path: str, true_labels=None):
    """
    保存预测结果
    
    Args:
        predictions: 预测结果 (samples, peaks, 2)
        data_info: 数据信息字典
        output_path: 输出路径
        true_labels: 真实标签 (samples, peaks, 2) 或 None（如果数据中没有标签）
    """
    logger = logging.getLogger(__name__)
    logger.info(f"保存结果到: {output_path}")
    
    # 验证输入
    if predictions is None:
        raise ValueError("预测结果为空（None）")
    if predictions.size == 0:
        raise ValueError("预测结果数组为空（size=0）")
    
    num_samples = data_info['num_samples']
    num_peaks = data_info['num_peaks']
    peak_ids = data_info['peak_ids']
    sample_ids = data_info.get('sample_ids', [str(i) for i in range(num_samples)])
    
    logger.info(f"准备保存: {num_samples} 样本 × {num_peaks} peaks = {num_samples * num_peaks} 条记录")
    
    # 验证数据维度
    if predictions.shape[0] != num_samples or predictions.shape[1] != num_peaks:
        raise ValueError(f"预测结果维度不匹配: predictions.shape={predictions.shape}, "
                        f"期望=({num_samples}, {num_peaks}, 2)")
    
    # 验证真实标签维度（如果提供）
    has_true_labels = true_labels is not None
    if has_true_labels:
        if true_labels.shape[0] != num_samples or true_labels.shape[1] != num_peaks:
            logger.warning(f"真实标签维度不匹配，将忽略真实值: true_labels.shape={true_labels.shape}, "
                          f"期望=({num_samples}, {num_peaks}, 2)")
            has_true_labels = False
        else:
            logger.info("✅ 将保存真实标签值")
    
    # 创建结果DataFrame
    results = []
    for sample_idx in range(num_samples):
        for peak_idx in range(num_peaks):
            peak_id = peak_ids[peak_idx] if peak_idx < len(peak_ids) else f"peak_{peak_idx}"
            sample_id = sample_ids[sample_idx] if sample_idx < len(sample_ids) else str(sample_idx)
            pred_pos = float(predictions[sample_idx, peak_idx, 0])
            pred_neg = float(predictions[sample_idx, peak_idx, 1])
            
            result_dict = {
                'sample_idx': sample_idx,
                'sample_id': sample_id,
                'peak_idx': peak_idx,
                'peak_id': peak_id,
                'pred_pos': pred_pos,
                'pred_neg': pred_neg,
                'pred_sum': pred_pos + pred_neg  # 总表达量
            }
            
            # 添加真实值（如果存在）
            if has_true_labels:
                true_pos = float(true_labels[sample_idx, peak_idx, 0])
                true_neg = float(true_labels[sample_idx, peak_idx, 1])
                result_dict['true_pos'] = true_pos
                result_dict['true_neg'] = true_neg
                result_dict['true_sum'] = true_pos + true_neg
            
            results.append(result_dict)
    
    if len(results) == 0:
        raise ValueError("结果列表为空，无法保存")
    
    df = pd.DataFrame(results)
    logger.info(f"创建DataFrame: {len(df)} 行 × {len(df.columns)} 列")
    
    # 检查DataFrame是否为空
    if df.empty:
        raise ValueError("DataFrame为空，无法保存")
    
    # 保存到CSV
    try:
        df.to_csv(output_path, index=False)
        # 验证文件是否成功创建
        if not os.path.exists(output_path):
            raise IOError(f"文件保存失败: {output_path}")
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            raise IOError(f"保存的文件为空: {output_path}")
        logger.info(f"✅ 结果已保存: {len(df)} 条记录, 文件大小: {file_size / 1024:.2f} KB")
    except Exception as e:
        logger.error(f"❌ 保存文件时出错: {e}")
        raise
    
    # 打印统计信息
    logger.info("\n预测结果统计:")
    logger.info(f"  总记录数: {len(df)}")
    logger.info(f"  正链表达: 均值={df['pred_pos'].mean():.4f}, 中位数={df['pred_pos'].median():.4f}, "
               f"min={df['pred_pos'].min():.4f}, max={df['pred_pos'].max():.4f}")
    logger.info(f"  负链表达: 均值={df['pred_neg'].mean():.4f}, 中位数={df['pred_neg'].median():.4f}, "
               f"min={df['pred_neg'].min():.4f}, max={df['pred_neg'].max():.4f}")
    logger.info(f"  总表达量: 均值={df['pred_sum'].mean():.4f}, 中位数={df['pred_sum'].median():.4f}, "
               f"min={df['pred_sum'].min():.4f}, max={df['pred_sum'].max():.4f}")
    
    # 如果有真实值，打印真实值统计和对比
    if has_true_labels:
        logger.info("\n真实值统计:")
        logger.info(f"  正链表达: 均值={df['true_pos'].mean():.4f}, 中位数={df['true_pos'].median():.4f}, "
                   f"min={df['true_pos'].min():.4f}, max={df['true_pos'].max():.4f}")
        logger.info(f"  负链表达: 均值={df['true_neg'].mean():.4f}, 中位数={df['true_neg'].median():.4f}, "
                   f"min={df['true_neg'].min():.4f}, max={df['true_neg'].max():.4f}")
        logger.info(f"  总表达量: 均值={df['true_sum'].mean():.4f}, 中位数={df['true_sum'].median():.4f}, "
                   f"min={df['true_sum'].min():.4f}, max={df['true_sum'].max():.4f}")
        
        # 计算预测与真实值的相关性（如果可能）
        try:
            from scipy.stats import pearsonr
            pred_all = df['pred_sum'].values
            true_all = df['true_sum'].values
            if len(pred_all) > 1 and np.std(pred_all) > 0 and np.std(true_all) > 0:
                corr, p_value = pearsonr(pred_all, true_all)
                logger.info(f"\n预测与真实值相关性:")
                logger.info(f"  Pearson相关系数: {corr:.4f} (p-value: {p_value:.2e})")
        except ImportError:
            logger.info("  (需要scipy来计算相关性)")
        except Exception as e:
            logger.warning(f"  计算相关性时出错: {e}")
    
    # 检查是否有全0的情况
    zero_pos = (df['pred_pos'] == 0).sum()
    zero_neg = (df['pred_neg'] == 0).sum()
    zero_sum = (df['pred_sum'] == 0).sum()
    if zero_pos == len(df):
        logger.warning(f"⚠️ 警告: 所有正链预测值都为0")
    if zero_neg == len(df):
        logger.warning(f"⚠️ 警告: 所有负链预测值都为0")
    if zero_sum == len(df):
        logger.warning(f"⚠️ 警告: 所有总表达量都为0")

def load_config(config_path: str = None):
    """
    加载配置文件
    
    Args:
        config_path: yaml配置文件路径（可选，默认使用sc配置）
    
    Returns:
        config: 配置字典
    """
    if config_path is None:
        # 默认使用SC配置文件，若不存在可显式提供 --config_path
        default_sc = 'get_model/config/yeast_training_sc.yaml'
        if os.path.exists(default_sc):
            config_path = default_sc
        else:
            raise FileNotFoundError(f"未找到默认配置文件: {default_sc}，请使用 --config_path 指定")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    # 注意：此时logger可能还未初始化，使用print
    print(f"加载配置文件: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def main():
    parser = argparse.ArgumentParser(description='单Peak模型推理脚本')
    parser.add_argument('--data_path', type=str, default=None,
                        help='输入npz数据文件路径（覆盖脚本配置）')
    parser.add_argument('--config_path', type=str, default=None,
                        help='yaml配置文件路径（覆盖脚本配置）')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型checkpoint路径（覆盖脚本配置）')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='batch大小（覆盖脚本配置）')
    parser.add_argument('--device', type=str, default=None,
                        help='计算设备: auto/cpu/cuda（覆盖脚本配置）')
    parser.add_argument('--output_base_dir', type=str, default=None,
                        help='输出基础目录（覆盖脚本配置）')
    
    args = parser.parse_args()
    
    # 获取脚本所在目录，用于解析相对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ============================================================================
    # 参数优先级：命令行参数 > 脚本配置 > 配置文件（可选）> 默认值
    # ============================================================================
    # 注意：模型配置会从checkpoint中自动读取，不需要yaml文件
    # yaml文件主要用于训练配置，推理时通常不需要
    
    # 1. 确定配置文件路径（可选，通常不需要）
    config_path = args.config_path if args.config_path is not None else CONFIG_PATH
    config = None
    inference_config = {}
    
    # 2. 加载配置文件（如果指定了，作为备选配置）
    if config_path is not None:
        try:
            config = load_config(config_path)
            inference_config = config.get('inference', {})
            print(f"✅ 已加载配置文件: {config_path}（作为备选配置）")
        except Exception as e:
            print(f"⚠️ 警告: 无法加载配置文件 {config_path}: {e}")
            print("   将使用脚本中的配置参数（这是正常的，推理时不需要yaml文件）")
    
    # 3. 确定数据路径（优先级：命令行 > 脚本配置 > 配置文件）
    if args.data_path is not None:
        data_paths = [args.data_path]  # 命令行参数：单个文件转为列表
    elif DATA_PATHS is not None:
        # 脚本配置
        if isinstance(DATA_PATHS, str):
            data_paths = [DATA_PATHS]  # 单个文件转为列表
        elif isinstance(DATA_PATHS, list):
            data_paths = DATA_PATHS.copy()  # 复制列表
        else:
            raise ValueError(f"DATA_PATHS 必须是字符串或列表，当前类型: {type(DATA_PATHS)}")
    elif 'data_paths' in inference_config:
        data_paths = inference_config['data_paths']  # 配置文件：多文件列表
        if not isinstance(data_paths, list):
            data_paths = [data_paths]  # 如果不是列表，转为列表
    elif 'data_path' in inference_config:
        data_paths = [inference_config['data_path']]  # 配置文件：单个文件转为列表
    else:
        raise ValueError("未指定数据路径，请在脚本开头的DATA_PATHS中设置，或使用--data_path参数")
    
    # 将相对路径转换为绝对路径
    data_paths = [
        os.path.join(script_dir, dp) if not os.path.isabs(dp) else dp
        for dp in data_paths
    ]
    
    # 4. 确定计算设备（优先级：命令行 > 脚本配置 > 配置文件 > auto）
    if args.device is not None:
        device_str = args.device
    elif DEVICE is not None:
        device_str = DEVICE
    else:
        device_str = inference_config.get('device', 'auto')
    
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    
    # 5. 确定模型路径（优先级：命令行 > 脚本配置 > 配置文件 > 默认路径）
    if args.model_path is not None:
        model_path = args.model_path
    elif MODEL_PATH is not None:
        model_path = MODEL_PATH
    elif 'model_path' in inference_config:
        model_path = inference_config['model_path']
    else:
        # 使用默认路径
        model_path = 'output/sc_atac_single_peak_training_20251117_133145/best_model.pth'
    
    # 将相对路径转换为绝对路径
    if not os.path.isabs(model_path):
        model_path = os.path.join(script_dir, model_path)
    
    # 6. 确定batch_size（优先级：命令行 > 脚本配置 > 配置文件 > 默认值）
    if args.batch_size is not None:
        batch_size = args.batch_size
    elif BATCH_SIZE is not None:
        batch_size = BATCH_SIZE
    elif 'batch_size' in inference_config:
        batch_size = inference_config['batch_size']
    else:
        batch_size = config.get('training', {}).get('batch_size', 512) if config else 512
    
    # 7. 确定输出基础目录（优先级：命令行 > 脚本配置 > 配置文件 > 默认值）
    if args.output_base_dir is not None:
        output_base_dir = args.output_base_dir
    elif OUTPUT_BASE_DIR is not None:
        output_base_dir = OUTPUT_BASE_DIR
    elif 'output_base_dir' in inference_config:
        output_base_dir = inference_config['output_base_dir']
    else:
        output_base_dir = config.get('data', {}).get('output_base_dir', 'output') if config else 'output'
    
    # 将相对路径转换为绝对路径
    if not os.path.isabs(output_base_dir):
        output_base_dir = os.path.join(script_dir, output_base_dir)
    
    # 创建带时间戳的输出目录
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(output_base_dir) / f"inference_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    log_file = output_dir / 'inference.log'
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='w', encoding='utf-8')
        ],
        force=True
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("单Peak模型推理")
    logger.info("=" * 70)
    logger.info(f"使用设备: {device}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"数据文件数: {len(data_paths)}")
    for i, dp in enumerate(data_paths, 1):
        logger.info(f"  [{i}] {os.path.basename(dp)}")
    logger.info(f"模型文件: {model_path}")
    
    # 推理时使用较小的batch_size
    if batch_size > 512:
        batch_size = 512
        logger.warning(f"⚠️ 推理batch_size过大，调整为: {batch_size}")
    logger.info(f"Batch大小: {batch_size}")
    
    # 检查文件是否存在和有效性
    logger.info("检查文件...")
    for data_path in data_paths:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        file_size = os.path.getsize(data_path)
        if file_size == 0:
            raise ValueError(f"数据文件为空: {data_path}")
        logger.info(f"  ✅ {os.path.basename(data_path)}: {file_size / (1024*1024):.2f} MB")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    model_size = os.path.getsize(model_path)
    if model_size == 0:
        raise ValueError(f"模型文件为空: {model_path}")
    logger.info(f"  ✅ 模型文件: {os.path.basename(model_path)}: {model_size / (1024*1024):.2f} MB")
    
    # 加载模型（所有文件共享同一个模型）
    logger.info("加载模型...")
    model, model_config = load_model(model_path, device)
    
    # 对每个数据文件进行推理
    all_results = []
    for file_idx, data_path in enumerate(data_paths, 1):
        logger.info("=" * 70)
        logger.info(f"处理文件 [{file_idx}/{len(data_paths)}]: {os.path.basename(data_path)}")
        logger.info("=" * 70)
        
        # 加载数据
        logger.info("加载数据...")
        data_info = load_data(data_path)
        
        # 进行预测
        logger.info("开始推理...")
        predictions, true_labels = predict(model, data_info, device, batch_size=batch_size)
        
        # 保存结果（每个文件单独保存）
        file_basename = os.path.splitext(os.path.basename(data_path))[0]
        output_csv = output_dir / f'predictions_{file_basename}.csv'
        logger.info(f"保存结果到: {output_csv}")
        save_results(predictions, data_info, str(output_csv), true_labels=true_labels)
        
        all_results.append({
            'file': os.path.basename(data_path),
            'output': str(output_csv),
            'num_samples': data_info['num_samples'],
            'num_peaks': data_info['num_peaks']
        })
    
    logger.info("=" * 70)
    logger.info("✅ 所有文件推理完成！")
    logger.info(f"结果保存在: {output_dir}")
    logger.info("\n推理结果汇总:")
    for result in all_results:
        logger.info(f"  - {result['file']}: {result['num_samples']} 样本 × {result['num_peaks']} peaks → {result['output']}")
    logger.info("=" * 70)

if __name__ == '__main__':
    main()

