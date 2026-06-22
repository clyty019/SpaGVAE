#!/usr/bin/env python
# ===============================
# SpaceFlow - Multi-Dataset & 1000 Seeds Analysis
# ===============================

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from sklearn.metrics.cluster import adjusted_rand_score
from SpaceFlow import SpaceFlow
import sys
from io import StringIO
import traceback

rpy2.robjects.numpy2ri.activate()

# ========== 配置区 ==========
DATASETS = ['151507', '151508', '151509', '151510', '151669', '151670', 
            '151671', '151672', '151673', '151674', '151675', '151676']
N_SEEDS = 1000
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 路径配置
BASE_DATA_PATH = '/extra/zhanglab0/SpatialTranscriptomicsData/10XVisium/DLPFC'
BASE_SAVE_PATH = '/extra/zhanglab0/INDV/zihend1/StaR'

# ========== 工具函数 ==========

def flatten_model_parameters(state_dict):
    """展平模型参数，用于后续稳定性分析"""
    flat_params = []
    for k in sorted(state_dict.keys()):
        if 'bias' in k or 'bn' in k.lower():
            continue
        flat_params.append(state_dict[k].flatten().cpu().numpy())
    return np.concatenate(flat_params)

def mclust_R(adata, num_cluster, used_obsm='sf_emb', random_seed=2020):
    """调用 R 语言的 mclust 进行聚类"""
    robjects.r.library("mclust")
    robjects.r['set.seed'](random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, 'EEE')
    mclust_res = np.array(res[-2])
    adata.obs['mclust'] = pd.Categorical(mclust_res.astype(str))
    return adata

def load_and_preprocess(dataset):
    """
    加载数据并进行严格的清洗与标准化预处理
    1. 过滤掉无标注的 Spot (NaN/NA)
    2. 自动检测准确的簇数 (n_clusters)
    3. 执行 Scanpy 标准预处理流水线
    """
    file_fold = f'{BASE_DATA_PATH}/{dataset}'
    
    # 加载 Visium 数据
    adata = sc.read_visium(file_fold, count_file=f'{dataset}_filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()
    
    # 1. 读取并映射 Ground Truth
    truth_df = pd.read_csv(f'{file_fold}_truth.txt', sep='\t', header=None, index_col=0)
    truth_df.columns = ['ground_truth']
    adata.obs['ground_truth'] = truth_df.loc[adata.obs_names, 'ground_truth']
    
    # 2. 核心清洗：去除无标注的 Spot
    # 过滤掉物理意义上的空值 (NaN)
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    
    # 过滤掉字符串形式的 "nan", "NA", "None" (不区分大小写)
    invalid_labels = ['nan', 'na', 'none', '']
    adata = adata[~adata.obs['ground_truth'].astype(str).str.lower().isin(invalid_labels)]

    # 3. 自动计算准确的簇数
    n_clusters = adata.obs['ground_truth'].nunique()
    print(f"Dataset {dataset}: Final n_clusters = {n_clusters}")

    # 4. 预处理流水线
    # 高变基因选择 (通常选 3000)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    
    # 归一化
    sc.pp.normalize_total(adata, target_sum=1e4)
    
    # 对数化
    sc.pp.log1p(adata)
    
    # 缩放 (不中心化以保持稀疏性，设置最大值为 10 增强鲁棒性)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    
    # 只保留高变基因以加速后续 SpaceFlow/stCluster 训练
    adata = adata[:, adata.var['highly_variable']]
    
    return adata, n_clusters

# ========== 核心运行逻辑 ==========

def main():
    print(f"开始 SpaceFlow 大规模分析。保存路径: {BASE_SAVE_PATH}")
    
    for dataset in DATASETS:
        dataset_dir = f"{BASE_SAVE_PATH}/{dataset}/SpaceFlow"
        os.makedirs(dataset_dir, exist_ok=True)
        
        # 加载数据
        try:
            adata_raw, n_clusters = load_and_preprocess(dataset)
            print(f"\n>>> 处理数据集: {dataset} | 目标簇数: {n_clusters}")
        except Exception as e:
            print(f"跳过数据集 {dataset}: 加载失败 - {e}")
            continue

        results = []
        
        for seed in range(N_SEEDS):
            # 检查是否已存在结果（断点续传）
            model_save_path = f"{dataset_dir}/model_seed{seed}.pth"
            # if os.path.exists(model_save_path):
            #     # 如果模型已存在，我们可以尝试读取之前的 ARI（如果已记录在 summary 中则跳过）
            #     continue
                
            if seed % 10 == 0:
                print(f"  进度: Dataset {dataset} | Seed {seed}/{N_SEEDS}")
            
            try:
                # 实例化 SpaceFlow
                sf = SpaceFlow.SpaceFlow(adata=adata_raw.copy())
                sf.preprocessing_data(n_top_genes=3000)
                
                # 训练模型 (此时已支持 sf.model)
                # 使用 StringIO 抑制冗长的训练输出
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                
                sf.train(
                    spatial_regularization_strength=0.1, 
                    z_dim=50, 
                    lr=1e-3, 
                    epochs=1000,
                    max_patience=50, 
                    min_stop=100, 
                    random_seed=seed,
                    gpu=0 if torch.cuda.is_available() else -1,
                    regularization_acceleration=True
                )
                
                sys.stdout = old_stdout
                
                # 1. 保存模型权重
                torch.save(sf.model.state_dict(), model_save_path)
                
                # 2. 获取嵌入并聚类
                adata = adata_raw.copy()
                adata.obsm['sf_emb'] = sf.embedding
                adata = mclust_R(adata, num_cluster=n_clusters, random_seed=seed)
                
                # 3. 计算并存储 ARI
                ari = adjusted_rand_score(adata.obs['mclust'], adata.obs['ground_truth'])
                results.append({'seed': seed, 'ARI': ari})
                
                # 每 50 个 seed 保存一次临时 CSV 以防崩溃
                if seed % 10 == 0:
                    pd.DataFrame(results).to_csv(f"{dataset_dir}/ari_checkpoint.csv", index=False)
                    
            except Exception as e:
                print(f"\n[ERROR] 数据集 {dataset} Seed {seed} 失败:")
                traceback.print_exc()
                results.append({'seed': seed, 'ARI': np.nan})

        # 保存该数据集的最终结果汇总
        final_df = pd.DataFrame(results)
        final_df.to_csv(f"{dataset_dir}/seed_ari_summary.csv", index=False)
        print(f"数据集 {dataset} 完成。平均 ARI: {final_df['ARI'].mean():.4f}")

if __name__ == '__main__':
    main()