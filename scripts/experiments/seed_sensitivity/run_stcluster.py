#!/usr/bin/env python
# =================================================================
# stCluster - 12 Datasets x 1000 Seeds Stability Analysis
# =================================================================

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from sklearn.metrics.cluster import adjusted_rand_score

# 导入 stCluster 核心库
from stCluster.stCluster.train import train as stCluster_train
from stCluster.stCluster.run import evaluate_embedding

# ========== 配置区 ==========
DATASETS = ['151507', '151508', '151509', '151510', '151669', '151670', 
            '151671', '151672', '151673', '151674', '151675', '151676']
N_SEEDS = 1000
BASE_DATA_PATH = '/extra/zhanglab0/SpatialTranscriptomicsData/10XVisium/DLPFC'
BASE_SAVE_PATH = '/extra/zhanglab0/INDV/zihend1/StaR'

# ========== 工具函数 ==========

def load_and_preprocess(dataset):
    """
    加载数据、清洗标签(排除NaN)、预处理基因
    """
    file_fold = f'{BASE_DATA_PATH}/{dataset}'
    adata = sc.read_visium(file_fold, count_file=f'{dataset}_filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()
    
    # 读取标签并清洗
    truth_df = pd.read_csv(f'{file_fold}_truth.txt', sep='\t', header=None, index_col=0)
    truth_df.columns = ['ground_truth']
    adata.obs['ground_truth'] = truth_df.loc[adata.obs_names, 'ground_truth']
    
    # 过滤掉真实的 NaN 和 字符串形式的 'nan', 'na'
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    invalid_labels = ['nan', 'na', 'none', '']
    adata = adata[~adata.obs['ground_truth'].astype(str).str.lower().isin(invalid_labels)]
    
    n_clusters = adata.obs['ground_truth'].nunique()
    
    # 基础预处理加速训练
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    adata = adata[:, adata.var['highly_variable']]
    
    return adata, n_clusters

def run_single_seed(adata, seed, n_clusters, dataset_save_dir):
    """
    执行单次训练并保存模型
    """
    model_path = f"{dataset_save_dir}/model_seed{seed}.pth"
    
    try:
        # 抑制 train 函数内部大量的 print 和 tqdm 输出
        # with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        # 直接利用 stCluster 原生的 save_model 参数
        adata_train, _ = stCluster_train(
            adata, 
            radius=150, 
            ae_rate=0.8, 
            adj_rate=0.2, 
            pred_rate=0.3, 
            seed=seed,
            save_model=model_path, # 关键：原生保存
            show=False
        )
        
        # 使用 R mclust 进行聚类评估
        adata_train, score = evaluate_embedding(
            adata=adata_train, 
            n_cluster=n_clusters,
            cluster_method=['mclust'], 
            cluster_score_method='ARI'
        )
    
        ari = score.get('mclust', np.nan)
        print(f"  - Seed {seed} | ARI: {ari:.4f}")
        return ari
    except Exception as e:
        return np.nan

# ========== 主程序 ==========

def main():
    print(f">>> 启动 stCluster 稳定性分析任务")
    print(f">>> 目标路径: {BASE_SAVE_PATH}")

    for dataset in DATASETS:
        dataset_dir = f"{BASE_SAVE_PATH}/{dataset}/stCluster"
        os.makedirs(dataset_dir, exist_ok=True)
        
        print(f"\n[Dataset: {dataset}] 正在加载...")
        try:
            adata_raw, n_clusters = load_and_preprocess(dataset)
            print(f"  - 数据载入成功，有效簇数: {n_clusters}")
        except Exception as e:
            print(f"  - 加载失败: {e}")
            continue

        results = []
        for seed in range(N_SEEDS):
            # 断点续传：检查模型文件是否已存在
            checkpoint_model = f"{dataset_dir}/model_seed{seed}.pth"
            if os.path.exists(checkpoint_model):
                continue

            if seed % 10 == 0:
                print(f"  - Progress: {dataset} | Seed {seed}/{N_SEEDS}", flush=True)

            ari = run_single_seed(adata_raw.copy(), seed, n_clusters, dataset_dir)
            results.append({'seed': seed, 'ARI': ari})

            # 每 50 次迭代保存一次 CSV，防止中途断电
            if seed % 1 == 0:
                pd.DataFrame(results).to_csv(f"{dataset_dir}/ari_checkpoint.csv", index=False)

        # 最终保存
        final_df = pd.DataFrame(results)
        final_df.to_csv(f"{dataset_dir}/seed_ari_summary.csv", index=False)
        print(f"  - {dataset} 完成。平均 ARI: {final_df['ARI'].mean():.4f}")

if __name__ == '__main__':
    main()