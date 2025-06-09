#!/usr/bin/env python3
import warnings
import os
import sys
import numpy as np
import pandas as pd
import pickle
import json
import logging
from datetime import datetime
from functools import partial
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
import matgl
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_pes
from sys import exit
import os
import sys
import logging
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from functools import partial
from ase.io import read
warnings.filterwarnings("ignore")
TRAIN_EXTXYZ_FILE = "combined_sampled_535_structures.extxyz"
VAL_EXTXYZ_FILE = "chno_conservative_valid.extxyz"
TEST_EXTXYZ_FILE = "chno_conservative_test.extxyz"

# 数据处理参数
CUTOFF = 5.0
THREEBODY_CUTOFF = 4.0
ELEMENT_TYPES = ["H", "C", "N", "O"]
INCLUDE_STRESS = True
BATCH_SIZE = 16

# 输出配置
RESULT_DIR = "result"
LOG_FILE = "data_processing.log"

def setup_logging():
    """设置日志记录"""
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    
    log_path = os.path.join(RESULT_DIR, LOG_FILE)
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Part 1: 数据处理和图转换开始")
    logger.info("=" * 60)
    
    return logger

def read_single_extxyz_data(file_path, dataset_name, logger):
    """
    从单个extxyz文件读取数据
    
    Args:
        file_path (str): extxyz文件路径
        dataset_name (str): 数据集名称
        logger: 日志记录器
        
    Returns:
        tuple: (structures, energies, forces, stresses)
    """
    logger.info(f"正在读取{dataset_name}: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到{dataset_name}文件: {file_path}")
    
    try:
        atoms_list = read(file_path, index=":", format="extxyz")
        logger.info(f"{dataset_name}读取 {len(atoms_list)} 个原子配置")
    except Exception as e:
        raise Exception(f"读取{dataset_name}失败: {e}")
    
    structures = []
    energies = []
    forces = []
    stresses = []
    
    adaptor = AseAtomsAdaptor()
    skipped = 0
    
    for i, atoms in enumerate(atoms_list):
        try:
            # 转换为pymatgen结构
            structure = adaptor.get_structure(atoms)
            
            # 检查元素类型
            elements = [str(site.specie) for site in structure]
            if not all(elem in ELEMENT_TYPES for elem in elements):
                logger.warning(f"{dataset_name}结构 {i} 包含不支持的元素，跳过")
                skipped += 1
                continue
            
            structures.append(structure)
            
            # 提取能量
            energy_keys = ['REF_energy', 'energy', 'Energy', 'total_energy']
            total_energy = None
            
            for key in energy_keys:
                if key in atoms.info:
                    total_energy = atoms.info[key]
                    break
            
            if total_energy is not None:
                energy_per_atom = total_energy / len(atoms)
                energies.append(energy_per_atom)
            else:
                logger.warning(f"{dataset_name}结构 {i} 缺少能量信息，跳过")
                logger.info(f"   可用的info键: {list(atoms.info.keys())}")
                skipped += 1
                continue
            
            # 提取力
            force_keys = ['REF_forces', 'forces', 'Forces', 'force']
            force_array = None
            
            for key in force_keys:
                if key in atoms.arrays:
                    force_array = atoms.arrays[key]
                    break
            
            if force_array is not None:
                forces.append(force_array.tolist())
            else:
                forces.append(np.zeros((len(atoms), 3)).tolist())
                if i == 0:
                    logger.warning(f"{dataset_name}缺少力信息，设为零")
            
            # 修复的应力处理部分
            if INCLUDE_STRESS:
                stress_keys = ['REF_stress', 'stress', 'Stress', 'virial']
                stress = None
                found_key = None
                
                for key in stress_keys:
                    if key in atoms.info:
                        stress = atoms.info[key]
                        found_key = key
                        break
                
                if stress is not None:
                    # 添加调试信息 - 只在第一个结构时打印
                    if i == 0:
                        logger.info(f"{dataset_name}找到应力数据，键名: {found_key}")
                        logger.info(f"   应力数据类型: {type(stress)}")
                        if hasattr(stress, 'shape'):
                            logger.info(f"   应力数据shape: {stress.shape}")
                        elif isinstance(stress, (list, tuple)):
                            logger.info(f"   应力数据长度: {len(stress)}")
                    
                    if hasattr(stress, 'shape'):
                        if stress.shape == (6,):
                            # Voigt notation: [xx, yy, zz, yz, xz, xy]
                            stress_matrix = np.array([
                                [stress[0], stress[5], stress[4]],
                                [stress[5], stress[1], stress[3]],
                                [stress[4], stress[3], stress[2]]
                            ])
                            if i == 0:
                                logger.info(f"   使用6元素Voigt格式转换应力")
                        elif stress.shape == (9,):
                            # 9元素格式：按行展开的3x3矩阵 [xx, xy, xz, yx, yy, yz, zx, zy, zz]
                            stress_matrix = stress.reshape(3, 3)
                            if i == 0:
                                logger.info(f"   使用9元素矩阵格式转换应力")
                        elif stress.shape == (3, 3):
                            stress_matrix = stress
                            if i == 0:
                                logger.info(f"   使用3x3矩阵格式应力")
                        else:
                            stress_matrix = np.zeros((3, 3))
                            if i == 0:
                                logger.warning(f"{dataset_name}应力shape {stress.shape} 不支持，设为零")
                    elif isinstance(stress, (list, tuple)):
                        if len(stress) == 6:
                            # Voigt notation: [xx, yy, zz, yz, xz, xy]
                            stress_array = np.array(stress, dtype=float)
                            stress_matrix = np.array([
                                [stress_array[0], stress_array[5], stress_array[4]],
                                [stress_array[5], stress_array[1], stress_array[3]],
                                [stress_array[4], stress_array[3], stress_array[2]]
                            ])
                            if i == 0:
                                logger.info(f"   使用6元素列表/元组Voigt格式转换应力")
                        elif len(stress) == 9:
                            # 9元素格式：按行展开的3x3矩阵
                            stress_array = np.array(stress, dtype=float)
                            stress_matrix = stress_array.reshape(3, 3)
                            if i == 0:
                                logger.info(f"   使用9元素列表/元组矩阵格式转换应力")
                        else:
                            if i == 0:
                                logger.warning(f"{dataset_name}应力为列表/元组但长度为{len(stress)}，期望长度为6或9，设为零")
                            stress_matrix = np.zeros((3, 3))
                    elif isinstance(stress, str):
                        # 处理字符串格式的应力（从extxyz文件中读取时可能是字符串）
                        try:
                            stress_values = [float(x) for x in stress.split()]
                            if len(stress_values) == 6:
                                # Voigt notation
                                stress_matrix = np.array([
                                    [stress_values[0], stress_values[5], stress_values[4]],
                                    [stress_values[5], stress_values[1], stress_values[3]],
                                    [stress_values[4], stress_values[3], stress_values[2]]
                                ])
                                if i == 0:
                                    logger.info(f"   使用6元素字符串Voigt格式转换应力")
                            elif len(stress_values) == 9:
                                # 9元素矩阵格式
                                stress_matrix = np.array(stress_values).reshape(3, 3)
                                if i == 0:
                                    logger.info(f"   使用9元素字符串矩阵格式转换应力")
                            else:
                                if i == 0:
                                    logger.warning(f"{dataset_name}字符串应力元素数量为{len(stress_values)}，期望6或9，设为零")
                                stress_matrix = np.zeros((3, 3))
                        except ValueError as e:
                            if i == 0:
                                logger.warning(f"{dataset_name}无法解析字符串应力'{stress}'，设为零: {e}")
                            stress_matrix = np.zeros((3, 3))
                    else:
                        if i == 0:
                            logger.warning(f"{dataset_name}应力格式不支持: 类型={type(stress)}, 值={stress}，设为零")
                        stress_matrix = np.zeros((3, 3))
                    
                    stresses.append(stress_matrix.tolist())
                else:
                    if i == 0:
                        logger.info(f"{dataset_name}未找到应力数据，设为零")
                        logger.info(f"   可用的info键: {list(atoms.info.keys())}")
                    stresses.append(np.zeros((3, 3)).tolist())
            else:
                stresses.append(np.zeros((3, 3)).tolist())
                
        except Exception as e:
            logger.warning(f"处理{dataset_name}结构 {i} 时出错，跳过: {e}")
            skipped += 1
            continue
    
    logger.info(f"{dataset_name}成功处理 {len(structures)} 个结构")
    if skipped > 0:
        logger.warning(f"{dataset_name}跳过了 {skipped} 个有问题的结构")
    
    if len(structures) == 0:
        raise ValueError(f"{dataset_name}没有有效的结构数据！")
    
    # 显示数据统计
    logger.info(f"{dataset_name}统计:")
    logger.info(f"   能量范围: {min(energies):.4f} 到 {max(energies):.4f} eV/atom")
    logger.info(f"   平均原子数: {np.mean([len(s) for s in structures]):.1f}")
    
    return structures, energies, forces, stresses

def read_all_datasets(logger):
    """读取所有三个数据集"""
    logger.info("开始读取所有数据集")
    
    # 读取训练集
    train_structures, train_energies, train_forces, train_stresses = read_single_extxyz_data(
        TRAIN_EXTXYZ_FILE, "训练集", logger
    )
    
    # 读取验证集
    val_structures, val_energies, val_forces, val_stresses = read_single_extxyz_data(
        VAL_EXTXYZ_FILE, "验证集", logger
    )
    
    # 读取测试集
    test_structures, test_energies, test_forces, test_stresses = read_single_extxyz_data(
        TEST_EXTXYZ_FILE, "测试集", logger
    )
    
    # 保存数据集统计信息
    dataset_stats = {
        '数据集': ['训练集', '验证集', '测试集'],
        '结构数量': [len(train_structures), len(val_structures), len(test_structures)],
        '能量均值(eV/atom)': [np.mean(train_energies), np.mean(val_energies), np.mean(test_energies)],
        '能量标准差(eV/atom)': [np.std(train_energies), np.std(val_energies), np.std(test_energies)]
    }
    
    df_stats = pd.DataFrame(dataset_stats)
    stats_path = os.path.join(RESULT_DIR, "dataset_statistics.csv")
    df_stats.to_csv(stats_path, index=False)
    logger.info(f"数据集统计已保存到: {stats_path}")
    
    return (train_structures, train_energies, train_forces, train_stresses,
            val_structures, val_energies, val_forces, val_stresses,
            test_structures, test_energies, test_forces, test_stresses)

def create_datasets_and_loaders(train_data, val_data, test_data, logger):
    """创建数据集和数据加载器"""
    logger.info("创建数据集和图转换...")
    
    # 创建结构到图的转换器
    converter = Structure2Graph(element_types=ELEMENT_TYPES, cutoff=CUTOFF)
    logger.info(f"图转换器配置: cutoff={CUTOFF}, elements={ELEMENT_TYPES}")
    
    # 解包数据
    train_structures, train_energies, train_forces, train_stresses = train_data
    val_structures, val_energies, val_forces, val_stresses = val_data
    test_structures, test_energies, test_forces, test_stresses = test_data
    
    # 创建训练数据集
    logger.info("创建训练数据集...")
    train_labels = {
        "energies": train_energies,
        "forces": train_forces,
        "stresses": train_stresses,
    }
    train_dataset = MGLDataset(
        threebody_cutoff=THREEBODY_CUTOFF,
        structures=train_structures,
        converter=converter,
        labels=train_labels,
        include_line_graph=True,
    )
    
    # 创建验证集
    logger.info("创建验证数据集...")
    val_labels = {
        "energies": val_energies,
        "forces": val_forces,
        "stresses": val_stresses,
    }
    val_dataset = MGLDataset(
        threebody_cutoff=THREEBODY_CUTOFF,
        structures=val_structures,
        converter=converter,
        labels=val_labels,
        include_line_graph=True,
    )
    
    # 创建测试数据集
    logger.info("创建测试数据集...")
    test_labels = {
        "energies": test_energies,
        "forces": test_forces,
        "stresses": test_stresses,
    }
    test_dataset = MGLDataset(
        threebody_cutoff=THREEBODY_CUTOFF,
        structures=test_structures,
        converter=converter,
        labels=test_labels,
        include_line_graph=True,
    )
    
    logger.info(f"数据集信息:")
    logger.info(f"   训练集: {len(train_dataset)} 个样本")
    logger.info(f"   验证集: {len(val_dataset)} 个样本")
    logger.info(f"   测试集: {len(test_dataset)} 个样本")
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    collate_fn = partial(collate_fn_pes, include_line_graph=True, include_stress=INCLUDE_STRESS)
    
    train_loader, val_loader, test_loader = MGLDataLoader(
        train_data=train_dataset,
        val_data=val_dataset,
        test_data=test_dataset,
        collate_fn=collate_fn,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )
    
    logger.info("数据加载器创建完成")
    
    return train_loader, val_loader, test_loader, (train_energies, val_energies, test_energies)

def save_processed_data(train_loader, val_loader, test_loader, energies_data, logger):
    """保存处理后的数据"""
    logger.info("保存处理后的数据...")
    
    # 保存数据加载器（序列化）
    data_dict = {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'energies_data': energies_data,
        'config': {
            'CUTOFF': CUTOFF,
            'THREEBODY_CUTOFF': THREEBODY_CUTOFF,
            'ELEMENT_TYPES': ELEMENT_TYPES,
            'INCLUDE_STRESS': INCLUDE_STRESS,
            'BATCH_SIZE': BATCH_SIZE
        }
    }
    
    data_path = os.path.join(RESULT_DIR, 'processed_data.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(data_dict, f)
    
    logger.info(f"处理后的数据已保存到: {data_path}")
    
    # 保存处理状态
    status = {
        'part1_completed': True,
        'completion_time': datetime.now().isoformat(),
        'data_path': data_path
    }
    
    status_path = os.path.join(RESULT_DIR, 'processing_status.json')
    with open(status_path, 'w') as f:
        json.dump(status, f, indent=2)
    
    logger.info(f"处理状态已保存到: {status_path}")

def check_files():
    """检查必要的文件是否存在"""
    missing_files = []
    
    for file_path, name in [
        (TRAIN_EXTXYZ_FILE, "训练集"),
        (VAL_EXTXYZ_FILE, "验证集"),
        (TEST_EXTXYZ_FILE, "测试集")
    ]:
        if not os.path.exists(file_path):
            missing_files.append(f"{name}: {file_path}")
    
    return missing_files

def main():
    """主函数"""
    logger = setup_logging()
    
    try:
        # 检查文件
        missing_files = check_files()
        if missing_files:
            logger.error("以下文件缺失:")
            for file in missing_files:
                logger.error(f"   {file}")
            logger.error("请修改文件路径后重新运行")
            sys.exit(1)
        
        logger.info("配置信息:")
        logger.info(f"   训练集: {TRAIN_EXTXYZ_FILE}")
        logger.info(f"   验证集: {VAL_EXTXYZ_FILE}")
        logger.info(f"   测试集: {TEST_EXTXYZ_FILE}")
        logger.info(f"   批次大小: {BATCH_SIZE}")
        logger.info(f"   元素类型: {ELEMENT_TYPES}")
        logger.info(f"   截断半径: {CUTOFF}")
        
        # 1. 读取所有数据集
        all_data = read_all_datasets(logger)
        
        # 2. 创建数据集和加载器
        train_loader, val_loader, test_loader, energies_data = create_datasets_and_loaders(
            all_data[:4], all_data[4:8], all_data[8:], logger
        )
        
        # 3. 保存处理后的数据
        save_processed_data(train_loader, val_loader, test_loader, energies_data, logger)
        
        logger.info("=" * 60)
        logger.info("Part 1: 数据处理和图转换完成!")
        logger.info("现在可以运行 Part 2: 模型训练")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"数据处理失败: {e}")
        logger.error("请检查错误信息并修复后重新运行")
        sys.exit(1)

if __name__ == "__main__":
    main()






# ---------------------------------------------------------------------------
import warnings
import os
import sys
import pickle
import json
import logging
from datetime import datetime
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

try:
    import torch
except ImportError:
    print("错误: 未找到PyTorch!")
    print("请安装PyTorch:")
    print("# CPU版本")
    sys.exit(1)

import matgl
from matgl.models import M3GNet
from matgl.utils.training import PotentialLightningModule

# 忽略警告信息
warnings.filterwarnings("ignore")

# 基础训练控制
MAX_EPOCHS = 200                    # 最大训练轮数
LEARNING_RATE = 5e-4               # 初始学习率
PATIENCE = 30                      # 早停耐心值（多少轮不改善后停止）
MIN_LR = 5e-7                      # 最小学习率
WARMUP_EPOCHS = 5

# 批次和数据加载
BATCH_SIZE = 32                    # 批次大小
ACCUMULATE_GRAD_BATCHES = 2        # 梯度累积批次数（模拟更大batch）
NUM_WORKERS = 4                    # 数据加载工作进程数

# 优化器参数
OPTIMIZER = "Adam"                 # 优化器类型 ("Adam", "AdamW", "SGD", "RMSprop")
WEIGHT_DECAY = 1e-4                # 权重衰减（L2正则化）
BETAS = (0.9, 0.999)              # Adam的beta参数
EPS = 1e-8                        # Adam的epsilon参数
MOMENTUM = 0.9                    # SGD动量（仅当OPTIMIZER="SGD"时使用）
AMSGRAD = True                  # 是否使用AMSGrad（Adam变种）

# 学习率调度器
LR_SCHEDULER = "CosineAnnealingWarmRestarts" # 学习率调度器类型
LR_FACTOR = 0.7                   # 学习率衰减因子
LR_PATIENCE = 15                  # 学习率调度耐心值
LR_STEP_SIZE = 20                 # StepLR的步长
LR_GAMMA = 0.1                   # StepLR和ExponentialLR的衰减率
LR_MILESTONES = [30, 60, 90]     # MultiStepLR的里程碑
T_MAX = 100                      # CosineAnnealingLR的周期
T_0 = 20                         # 余弦退火初始周期
T_MULT = 2                       # 周期倍增因子
ETA_MIN = 1e-6  
# 损失函数权重
ENERGY_WEIGHT = 10.0              # 能量损失权重
FORCE_WEIGHT = 100.0               # 力损失权重
STRESS_WEIGHT = 10.0             # 应力损失权重

# 梯度控制
GRADIENT_CLIP_VAL = 0.5          # 梯度裁剪值（0表示不裁剪）
GRADIENT_CLIP_ALGORITHM = "norm"  # 梯度裁剪算法 ("norm", "value")

# 验证和监控
VAL_CHECK_INTERVAL = 0.5        # 验证检查间隔（每个epoch的比例）
CHECK_VAL_EVERY_N_EPOCH = 1      # 每N个epoch验证一次
MONITOR_METRIC = "val_Total_Loss" # 监控的指标
MONITOR_MODE = "min"             # 监控模式 ("min", "max")
MIN_DELTA = 0.0001              # 最小改善量

# 模型保存
SAVE_TOP_K = 5                  # 保存最好的K个模型
SAVE_LAST = True                # 是否保存最后一个模型
SAVE_WEIGHTS_ONLY = False       # 是否只保存权重

# 训练稳定性
DETERMINISTIC = False           # 是否使用确定性算法（可重现但较慢）
BENCHMARK = True               # 是否启用cudnn benchmark（加速但不确定性）

# 调试和开发
FAST_DEV_RUN = False           # 快速开发运行（只训练少量batch）
OVERFIT_BATCHES = 0            # 过拟合少量批次用于调试
LIMIT_TRAIN_BATCHES = 1.0      # 限制训练批次数量（1.0=全部）
LIMIT_VAL_BATCHES = 1.0        # 限制验证批次数量
LIMIT_TEST_BATCHES = 1.0       # 限制测试批次数量

# 日志记录
LOG_EVERY_N_STEPS = 10         # 每N步记录一次日志
FLUSH_LOGS_EVERY_N_STEPS = 100 # 每N步刷新日志

# 自动混合精度（加速训练）
USE_AMP = False                # 是否使用自动混合精度
AMP_BACKEND = "native"         # AMP后端 ("native", "apex")

# 随机种子
RANDOM_SEED = 42              # 随机种子（None表示不设置）

# 模型配置
USE_PRETRAINED = False        # 关掉预训练
PRETRAINED_MODEL = "M3GNet-MP-2021.2.8-PES"

# 输出配置
RESULT_DIR = "result"
MODEL_SAVE_PATH = "trained_m3gnet_model"
LOG_DIR = "training_logs"
LOG_FILE = "model_training.log"

def setup_logging():
    """设置日志记录"""
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    
    log_path = os.path.join(RESULT_DIR, LOG_FILE)
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Part 2: 模型训练开始")
    logger.info("=" * 60)
    
    return logger

def load_processed_data(logger):
    """加载处理好的数据"""
    logger.info("加载处理好的数据...")
    
    # 检查处理状态
    status_path = os.path.join(RESULT_DIR, 'processing_status.json')
    if not os.path.exists(status_path):
        logger.error("找不到数据处理状态文件")
        logger.error("请先运行 Part 1: 数据处理")
        sys.exit(1)
    
    with open(status_path, 'r') as f:
        status = json.load(f)
    
    if not status.get('part1_completed', False):
        logger.error("Part 1 数据处理未完成")
        logger.error("请先运行 Part 1: 数据处理")
        sys.exit(1)
    
    # 加载数据
    data_path = status['data_path']
    if not os.path.exists(data_path):
        logger.error(f"找不到处理后的数据文件: {data_path}")
        sys.exit(1)
    
    logger.info(f"从 {data_path} 加载数据...")
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    train_loader = data_dict['train_loader']
    val_loader = data_dict['val_loader']
    test_loader = data_dict['test_loader']
    energies_data = data_dict['energies_data']
    config = data_dict['config']
    
    logger.info("数据加载完成:")
    logger.info(f"   训练集批次数: {len(train_loader)}")
    logger.info(f"   验证集批次数: {len(val_loader)}")
    logger.info(f"   测试集批次数: {len(test_loader)}")
    logger.info(f"   配置: {config}")
    
    return train_loader, val_loader, test_loader, energies_data, config

def create_model_and_trainer(config, logger):
    """创建模型和训练器"""
    logger.info("创建模型...")
    
    # 从配置中获取参数
    ELEMENT_TYPES = config['ELEMENT_TYPES']
    INCLUDE_STRESS = config['INCLUDE_STRESS']
    
    # 设置随机种子
    if RANDOM_SEED is not None:
        L.seed_everything(RANDOM_SEED, workers=True)
        logger.info(f"设置随机种子: {RANDOM_SEED}")
    
    if USE_PRETRAINED:
        try:
            logger.info(f"加载预训练模型: {PRETRAINED_MODEL}")
            pretrained = matgl.load_model(PRETRAINED_MODEL)
            model = pretrained.model
            
            # 获取元素能量偏移
            element_refs = getattr(pretrained, 'element_refs', None)
            property_offset = element_refs.property_offset if element_refs else None
            
            logger.info("预训练模型加载成功")
            
        except Exception as e:
            logger.warning(f"预训练模型加载失败: {e}")
            logger.info("改为从头训练...")
            model = M3GNet(element_types=ELEMENT_TYPES, is_intensive=False)
            property_offset = None
    else:
        logger.info("从头创建M3GNet模型...")
        model = M3GNet(element_types=ELEMENT_TYPES, is_intensive=False)
        property_offset = None
    
    # 创建Lightning模块
    stress_weight = STRESS_WEIGHT if INCLUDE_STRESS else 0.0
    
    lit_module = PotentialLightningModule(
        model=model,
        element_refs=property_offset,
        lr=LEARNING_RATE,
        include_line_graph=True,
        stress_weight=stress_weight,
    )
    
    logger.info(f"模型配置:")
    logger.info(f"   学习率: {LEARNING_RATE}")
    logger.info(f"   应力权重: {stress_weight}")
    logger.info(f"   元素类型: {ELEMENT_TYPES}")
    logger.info(f"   优化器: {OPTIMIZER}")
    logger.info(f"   权重衰减: {WEIGHT_DECAY}")
    
    # 设置回调函数
    callbacks = []
    
    # 创建检查点目录
    checkpoint_dir = "./checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # 模型检查点
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="m3gnet-{epoch:02d}-{val_Total_Loss:.4f}",
        monitor=MONITOR_METRIC,
        mode=MONITOR_MODE,
        save_top_k=SAVE_TOP_K,
        save_last=SAVE_LAST,
        save_weights_only=SAVE_WEIGHTS_ONLY,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    logger.info(f"模型检查点将保存到: {checkpoint_dir}")
    
    # 早停
    early_stopping = EarlyStopping(
        monitor=MONITOR_METRIC,
        patience=PATIENCE,
        mode=MONITOR_MODE,
        min_delta=MIN_DELTA,
        verbose=True,
    )
    callbacks.append(early_stopping)
    logger.info(f"早停配置: 耐心值={PATIENCE}, 监控指标={MONITOR_METRIC}")
    
    # 创建日志目录
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    # 日志记录器
    csv_logger = CSVLogger(LOG_DIR, name="m3gnet_training")
    logger.info(f"训练日志将保存到: {LOG_DIR}")
    
    # 创建训练器
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices="auto",
        logger=csv_logger,
        callbacks=callbacks,
        inference_mode=False,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        val_check_interval=VAL_CHECK_INTERVAL,
        check_val_every_n_epoch=CHECK_VAL_EVERY_N_EPOCH,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        gradient_clip_algorithm=GRADIENT_CLIP_ALGORITHM,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        deterministic=DETERMINISTIC,
        benchmark=BENCHMARK,
        fast_dev_run=FAST_DEV_RUN,
        overfit_batches=OVERFIT_BATCHES,
        limit_train_batches=LIMIT_TRAIN_BATCHES,
        limit_val_batches=LIMIT_VAL_BATCHES,
        limit_test_batches=LIMIT_TEST_BATCHES,
        precision=32,  # 明确设置精度
    )
    
    logger.info(f"训练器配置:")
    logger.info(f"   最大轮数: {MAX_EPOCHS}")
    logger.info(f"   加速器: auto")
    logger.info(f"   设备: auto")
    logger.info(f"   梯度裁剪: {GRADIENT_CLIP_VAL}")
    logger.info(f"   验证间隔: {VAL_CHECK_INTERVAL}")
    
    return lit_module, trainer


def train_model(lit_module, trainer, train_loader, val_loader, logger):
    """训练模型"""
    logger.info("开始模型训练...")
    
    try:
        # 开始训练
        start_time = datetime.now()
        logger.info(f"训练开始时间: {start_time}")
        
        trainer.fit(
            model=lit_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        logger.info(f"训练结束时间: {end_time}")
        logger.info(f"训练总耗时: {training_duration}")
        
        # 保存模型
        logger.info(f"保存模型到: {MODEL_SAVE_PATH}")
        lit_module.model.save(MODEL_SAVE_PATH)
        
        # 更新状态
        status_path = os.path.join(RESULT_DIR, 'processing_status.json')
        with open(status_path, 'r') as f:
            status = json.load(f)
        
        status.update({
            'part2_completed': True,
            'training_completion_time': end_time.isoformat(),
            'training_duration': str(training_duration),
            'model_path': MODEL_SAVE_PATH,
            'training_config': {
                'max_epochs': MAX_EPOCHS,
                'learning_rate': LEARNING_RATE,
                'patience': PATIENCE,
                'use_pretrained': USE_PRETRAINED,
                'optimizer': OPTIMIZER,
                'weight_decay': WEIGHT_DECAY,
                'batch_size': BATCH_SIZE,
                'gradient_clip_val': GRADIENT_CLIP_VAL
            }
        })
        
        with open(status_path, 'w') as f:
            json.dump(status, f, indent=2)
        
        logger.info("训练状态已更新")
        return True
        
    except Exception as e:
        logger.error(f"训练过程出错: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return False

def test_model(trainer, test_loader, logger):
    """测试模型"""
    logger.info("开始模型测试...")
    
    try:
        test_results = trainer.test(dataloaders=test_loader)
        logger.info("模型测试完成")
        logger.info(f"测试结果: {test_results}")
        return test_results
        
    except Exception as e:
        logger.error(f"模型测试出错: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return None


def validate_training_params(logger):
    """验证训练参数"""
    logger.info("验证训练参数...")
    
    errors = []
    warnings = []
    
    # 基本检查
    if MAX_EPOCHS <= 0:
        errors.append("MAX_EPOCHS必须大于0")
    
    if LEARNING_RATE <= 0:
        errors.append("LEARNING_RATE必须大于0")
    
    if BATCH_SIZE <= 0:
        errors.append("BATCH_SIZE必须大于0")
    
    if PATIENCE < 0:
        errors.append("PATIENCE不能为负数")
    
    # 权重检查
    if ENERGY_WEIGHT < 0 or FORCE_WEIGHT < 0 or STRESS_WEIGHT < 0:
        errors.append("损失权重不能为负数")
    
    # 警告
    if LEARNING_RATE > 0.01:
        warnings.append("学习率较大，可能导致训练不稳定")
    
    if BATCH_SIZE > 64:
        warnings.append("批次大小较大，可能导致内存不足")
    
    # 记录结果
    if errors:
        logger.error("参数验证发现错误:")
        for error in errors:
            logger.error(f"  - {error}")
    
    if warnings:
        logger.warning("参数验证发现警告:")
        for warning in warnings:
            logger.warning(f"  - {warning}")
    
    if not errors:
        logger.info("参数验证通过")
    
    return len(errors) == 0


def main():
    """主函数"""
    logger = setup_logging()
    
    try:
        # 验证参数
        if not validate_training_params(logger):
            logger.error("参数验证失败，请修复后重试")
            sys.exit(1)
        
        logger.info("训练配置:")
        logger.info(f"   最大训练轮数: {MAX_EPOCHS}")
        logger.info(f"   学习率: {LEARNING_RATE}")
        logger.info(f"   早停耐心值: {PATIENCE}")
        logger.info(f"   批次大小: {BATCH_SIZE}")
        logger.info(f"   优化器: {OPTIMIZER}")
        logger.info(f"   权重衰减: {WEIGHT_DECAY}")
        logger.info(f"   使用预训练模型: {USE_PRETRAINED}")
        if USE_PRETRAINED:
            logger.info(f"   预训练模型: {PRETRAINED_MODEL}")
        
        # 1. 加载处理好的数据
        train_loader, val_loader, test_loader, energies_data, config = load_processed_data(logger)
        
        # 2. 创建模型和训练器
        lit_module, trainer = create_model_and_trainer(config, logger)
        
        # 3. 训练模型
        training_success = train_model(lit_module, trainer, train_loader, val_loader, logger)
        
        if training_success:
            # 4. 测试模型
            test_results = test_model(trainer, test_loader, logger)
            
            logger.info("=" * 60)
            logger.info("Part 2: 模型训练完成!")
            logger.info(f"模型已保存到: {MODEL_SAVE_PATH}")
            logger.info("现在可以运行 Part 3: 结果可视化")
            logger.info("=" * 60)
        else:
            logger.error("训练失败，请检查错误信息")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"训练过程失败: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        logger.error("请检查错误信息并修复后重新运行")
        sys.exit(1)

if __name__ == "__main__":
    main()


# ---------------------------------------------------------------
#!/usr/bin/env python3
import warnings
import os
import sys
import pickle
import json
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

try:
    import torch
except ImportError:
    print("错误: 未找到PyTorch!")
    print("请先安装PyTorch")
    sys.exit(1)

import matgl
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor

# 忽略警告信息
warnings.filterwarnings("ignore")

# 设置matplotlib样式
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150
plt.style.use('seaborn-v0_8')

# 输出配置
RESULT_DIR = "result"
LOG_DIR = "training_logs"
MODEL_SAVE_PATH = "trained_m3gnet_model"
LOG_FILE = "visualization.log"

# 测试文件（用于快速测试）
TEST_EXTXYZ_FILE = "chno_conservative_test.extxyz"

def setup_logging():
    """设置日志记录"""
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    
    log_path = os.path.join(RESULT_DIR, LOG_FILE)
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Part 3: 结果可视化开始")
    logger.info("=" * 60)
    
    return logger

def check_previous_parts(logger):
    """检查前面的部分是否完成"""
    status_path = os.path.join(RESULT_DIR, 'processing_status.json')
    if not os.path.exists(status_path):
        logger.error("找不到处理状态文件")
        logger.error("请先运行 Part 1 和 Part 2")
        sys.exit(1)
    
    with open(status_path, 'r') as f:
        status = json.load(f)
    
    if not status.get('part1_completed', False):
        logger.error("Part 1 数据处理未完成")
        sys.exit(1)
    
    if not status.get('part2_completed', False):
        logger.error("Part 2 模型训练未完成")
        sys.exit(1)
    
    logger.info("前面的部分都已完成，可以进行可视化")
    return status

def load_data_and_model(logger):
    """加载数据和模型"""
    logger.info("加载数据和训练好的模型...")
    
    # 加载处理后的数据
    data_path = os.path.join(RESULT_DIR, 'processed_data.pkl')
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    train_loader = data_dict['train_loader']
    val_loader = data_dict['val_loader']
    test_loader = data_dict['test_loader']
    energies_data = data_dict['energies_data']
    
    # 加载训练好的模型
    if not os.path.exists(MODEL_SAVE_PATH):
        logger.error(f"找不到训练好的模型: {MODEL_SAVE_PATH}")
        sys.exit(1)
    
    model = matgl.load_model(MODEL_SAVE_PATH)
    logger.info("模型加载成功")
    
    return train_loader, val_loader, test_loader, energies_data, model

def plot_training_curves(logger):
    """绘制训练曲线"""
    logger.info("绘制训练曲线...")
    
    # 读取训练日志
    log_file = None
    for root, dirs, files in os.walk(LOG_DIR):
        for file in files:
            if file.endswith('metrics.csv'):
                log_file = os.path.join(root, file)
                break
        if log_file:
            break
    
    if not log_file or not os.path.exists(log_file):
        logger.warning("找不到训练日志文件，跳过训练曲线绘制")
        return
    
    df = pd.read_csv(log_file)
    logger.info(f"读取训练日志: {log_file}")
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('M3GNet Training Progress', fontsize=16, fontweight='bold')
    
    # 1. 总损失
    if 'train_Total_Loss' in df.columns and 'val_Total_Loss' in df.columns:
        axes[0, 0].plot(df['epoch'], df['train_Total_Loss'].dropna(), 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(df['epoch'], df['val_Total_Loss'].dropna(), 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
    
    # 2. 能量MAE
    if 'train_Energy_MAE' in df.columns and 'val_Energy_MAE' in df.columns:
        axes[0, 1].plot(df['epoch'], df['train_Energy_MAE'].dropna(), 'b-', label='Train Energy MAE', linewidth=2)
        axes[0, 1].plot(df['epoch'], df['val_Energy_MAE'].dropna(), 'r-', label='Val Energy MAE', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Energy MAE (eV/atom)')
        axes[0, 1].set_title('Energy Mean Absolute Error')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 力MAE
    if 'train_Force_MAE' in df.columns and 'val_Force_MAE' in df.columns:
        axes[1, 0].plot(df['epoch'], df['train_Force_MAE'].dropna(), 'b-', label='Train Force MAE', linewidth=2)
        axes[1, 0].plot(df['epoch'], df['val_Force_MAE'].dropna(), 'r-', label='Val Force MAE', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Force MAE (eV/Å)')
        axes[1, 0].set_title('Force Mean Absolute Error')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 学习率
    if 'lr-Adam' in df.columns:
        axes[1, 1].plot(df['epoch'], df['lr-Adam'].dropna(), 'g-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    # 保存图片
    curve_path = os.path.join(RESULT_DIR, "training_curves.png")
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(RESULT_DIR, "training_curves.pdf"), bbox_inches='tight')
    plt.close()
    
    logger.info(f"训练曲线已保存到: {curve_path}")

def evaluate_and_plot_predictions(model, data_loader, actual_energies, dataset_name, logger):
    """评估模型并绘制预测结果"""
    logger.info(f"评估{dataset_name}集性能...")
    
    model.eval()
    predictions = []
    
    # 获取预测结果
    for batch in data_loader:
        with torch.no_grad():
            pred = model(batch)
            predictions.extend(pred.detach().cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actual_energies)
    
    # 计算评估指标
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    
    logger.info(f"{dataset_name}集性能:")
    logger.info(f"   MAE: {mae:.4f} eV/atom")
    logger.info(f"   RMSE: {rmse:.4f} eV/atom")
    logger.info(f"   R²: {r2:.4f}")
    
    # 保存评估结果
    eval_results = {
        'Dataset': [dataset_name],
        'MAE (eV/atom)': [mae],
        'RMSE (eV/atom)': [rmse],
        'R²': [r2],
        'Sample_Count': [len(actuals)]
    }
    
    df_eval = pd.DataFrame(eval_results)
    eval_path = os.path.join(RESULT_DIR, f"{dataset_name.lower()}_evaluation.csv")
    df_eval.to_csv(eval_path, index=False)
    
    # 绘制预测vs实际图
    plt.figure(figsize=(10, 8))
    
    # 散点图
    plt.scatter(actuals, predictions, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # 理想线
    min_val = min(min(actuals), min(predictions))
    max_val = max(max(actuals), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # 添加统计信息
    plt.text(0.05, 0.95, f'MAE = {mae:.4f} eV/atom\nRMSE = {rmse:.4f} eV/atom\nR² = {r2:.4f}\nn = {len(actuals)}',
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel('Actual Energy (eV/atom)', fontsize=14)
    plt.ylabel('Predicted Energy (eV/atom)', fontsize=14)
    plt.title(f'M3GNet {dataset_name} Set: Predicted vs Actual Energy', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 设置坐标轴相等
    plt.axis('equal')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    plt.tight_layout()
    
    # 保存图片
    pred_path = os.path.join(RESULT_DIR, f"{dataset_name.lower()}_predictions.png")
    plt.savefig(pred_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(RESULT_DIR, f"{dataset_name.lower()}_predictions.pdf"), bbox_inches='tight')
    plt.close()
    
    logger.info(f"{dataset_name}集预测图已保存到: {pred_path}")
    
    return mae, rmse, r2

def plot_error_distribution(model, test_loader, test_energies, logger):
    """绘制误差分布图"""
    logger.info("绘制误差分布...")
    
    model.eval()
    predictions = []
    
    # 获取预测结果
    for batch in test_loader:
        with torch.no_grad():
            pred = model(batch)
            predictions.extend(pred.detach().cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(test_energies)
    errors = predictions - actuals
    
    # 创建误差分布图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. 误差直方图
    axes[0].hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].set_xlabel('Prediction Error (eV/atom)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Error Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. 误差vs实际值
    axes[1].scatter(actuals, errors, alpha=0.6, s=30)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Actual Energy (eV/atom)')
    axes[1].set_ylabel('Prediction Error (eV/atom)')
    axes[1].set_title('Error vs Actual Energy')
    axes[1].grid(True, alpha=0.3)
    
    # 3. 误差的Q-Q图
    stats.probplot(errors, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot (Normality Check)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    error_path = os.path.join(RESULT_DIR, "error_analysis.png")
    plt.savefig(error_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(RESULT_DIR, "error_analysis.pdf"), bbox_inches='tight')
    plt.close()
    
    logger.info(f"误差分析已保存到: {error_path}")

def create_summary_report(train_mae, val_mae, test_mae, train_r2, val_r2, test_r2, logger):
    """创建训练总结报告"""
    logger.info("创建总结报告...")
    
    # 创建总结表格
    summary_data = {
        'Dataset': ['Training', 'Validation', 'Test'],
        'MAE (eV/atom)': [train_mae, val_mae, test_mae],
        'R²': [train_r2, val_r2, test_r2]
    }
    
    df_summary = pd.DataFrame(summary_data)
    summary_path = os.path.join(RESULT_DIR, "training_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    
    # 创建性能对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    datasets = ['Training', 'Validation', 'Test']
    mae_values = [train_mae, val_mae, test_mae]
    r2_values = [train_r2, val_r2, test_r2]
    
    # MAE对比
    bars1 = ax1.bar(datasets, mae_values, color=['skyblue', 'lightgreen', 'lightcoral'], 
                    edgecolor='black', linewidth=1)
    ax1.set_ylabel('MAE (eV/atom)')
    ax1.set_title('Mean Absolute Error Comparison')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, val in zip(bars1, mae_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_values)*0.01,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # R²对比
    bars2 = ax2.bar(datasets, r2_values, color=['skyblue', 'lightgreen', 'lightcoral'],
                    edgecolor='black', linewidth=1)
    ax2.set_ylabel('R²')
    ax2.set_title('R² Score Comparison')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, val in zip(bars2, r2_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图片
    comparison_path = os.path.join(RESULT_DIR, "performance_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(RESULT_DIR, "performance_comparison.pdf"), bbox_inches='tight')
    plt.close()
    
    logger.info(f"总结报告已保存到: {summary_path}")
    logger.info(f"性能对比图已保存到: {comparison_path}")

def quick_model_test(model, logger):
    """快速测试训练好的模型"""
    logger.info("进行快速模型测试...")
    
    if not os.path.exists(TEST_EXTXYZ_FILE):
        logger.warning(f"找不到测试文件: {TEST_EXTXYZ_FILE}，跳过快速测试")
        return
    
    try:
        # 读取测试结构
        atoms = read(TEST_EXTXYZ_FILE, index=0)
        adaptor = AseAtomsAdaptor()
        test_structure = adaptor.get_structure(atoms)
        
        # 预测
        prediction = model.predict_structure(test_structure)
        
        # 尝试多种能量键名
        energy_keys = ['REF_energy', 'energy', 'Energy', 'total_energy']
        actual_energy = 0
        
        for key in energy_keys:
            if key in atoms.info:
                actual_energy = atoms.info[key] / len(atoms)
                break
        
        logger.info(f"快速测试结果:")
        logger.info(f"   预测能量: {float(prediction):.6f} eV/atom")
        logger.info(f"   实际能量: {actual_energy:.6f} eV/atom")
        logger.info(f"   绝对误差: {abs(float(prediction) - actual_energy):.6f} eV/atom")
        
        # 保存测试结果
        quick_test_results = {
            'Predicted_Energy_eV_per_atom': [float(prediction)],
            'Actual_Energy_eV_per_atom': [actual_energy],
            'Absolute_Error_eV_per_atom': [abs(float(prediction) - actual_energy)]
        }
        
        df_quick = pd.DataFrame(quick_test_results)
        quick_path = os.path.join(RESULT_DIR, "quick_test_result.csv")
        df_quick.to_csv(quick_path, index=False)
        logger.info(f"快速测试结果已保存到: {quick_path}")
        
    except Exception as e:
        logger.error(f"快速测试失败: {e}")

def main():
    """主函数"""
    logger = setup_logging()
    
    try:
        # 1. 检查前面的部分是否完成
        status = check_previous_parts(logger)
        
        # 2. 加载数据和模型
        train_loader, val_loader, test_loader, energies_data, model = load_data_and_model(logger)
        train_energies, val_energies, test_energies = energies_data
        
        # 3. 绘制训练曲线
        plot_training_curves(logger)
        
        # 4. 评估各个数据集
        train_mae, train_rmse, train_r2 = evaluate_and_plot_predictions(
            model.model, train_loader, train_energies, "Train", logger
        )
        val_mae, val_rmse, val_r2 = evaluate_and_plot_predictions(
            model.model, val_loader, val_energies, "Validation", logger
        )
        test_mae, test_rmse, test_r2 = evaluate_and_plot_predictions(
            model.model, test_loader, test_energies, "Test", logger
        )
        
        # 5. 绘制误差分析
        plot_error_distribution(model.model, test_loader, test_energies, logger)
        
        # 6. 创建总结报告
        create_summary_report(train_mae, val_mae, test_mae, train_r2, val_r2, test_r2, logger)
        
        # 7. 快速模型测试
        quick_model_test(model, logger)
        
        # 8. 更新完成状态
        status_path = os.path.join(RESULT_DIR, 'processing_status.json')
        status.update({
            'part3_completed': True,
            'visualization_completion_time': datetime.now().isoformat(),
            'final_results': {
                'train_mae': train_mae,
                'val_mae': val_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'test_r2': test_r2
            }
        })
        
        with open(status_path, 'w') as f:
            json.dump(status, f, indent=2)
        
        logger.info("=" * 60)
        logger.info("Part 3: 结果可视化完成!")
        logger.info("=" * 60)
        logger.info("所有结果文件:")
        logger.info(f"   训练曲线: {RESULT_DIR}/training_curves.png")
        logger.info(f"   预测结果: {RESULT_DIR}/test_predictions.png")
        logger.info(f"   误差分析: {RESULT_DIR}/error_analysis.png")
        logger.info(f"   性能对比: {RESULT_DIR}/performance_comparison.png")
        logger.info(f"   数据统计: {RESULT_DIR}/dataset_statistics.csv")
        logger.info(f"   训练总结: {RESULT_DIR}/training_summary.csv")
        logger.info("=" * 60)
        logger.info("最终性能:")
        logger.info(f"   训练集 MAE: {train_mae:.4f} eV/atom, R²: {train_r2:.4f}")
        logger.info(f"   验证集 MAE: {val_mae:.4f} eV/atom, R²: {val_r2:.4f}")
        logger.info(f"   测试集 MAE: {test_mae:.4f} eV/atom, R²: {test_r2:.4f}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"可视化过程失败: {e}")
        logger.error("请检查错误信息并修复后重新运行")
        sys.exit(1)

if __name__ == "__main__":
    main()