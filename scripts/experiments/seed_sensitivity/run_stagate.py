#!/usr/bin/env python
# =================================================================
# STAGATE - 12 Datasets x 1000 Seeds Analysis (Unified & Corrected)
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
import STAGATE_pyG as STAGATE

# ========== Configuration ==========
DATASETS = ['151507', '151508', '151509', '151510', '151669', '151670', 
            '151671', '151672', '151673', '151674', '151675', '151676']
N_SEEDS = 1000
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BASE_DATA_PATH = '/extra/zhanglab0/SpatialTranscriptomicsData/10XVisium/DLPFC'
BASE_SAVE_PATH = '/extra/zhanglab0/INDV/zihend1/StaR'

# ========== R Clustering Helper ==========
def mclust_R(adata, num_cluster, used_obsm='STAGATE', random_seed=2020):
    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    robjects.r.library("mclust")
    robjects.r['set.seed'](random_seed)
    rmclust = robjects.r['Mclust']
    # 提取 latent embedding
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, 'EEE')
    mclust_res = np.array(res[-2])
    adata.obs['mclust'] = pd.Categorical(mclust_res.astype(str))
    return adata

# ========== Data Loader ==========
def load_and_preprocess(dataset):
    file_fold = f'{BASE_DATA_PATH}/{dataset}'
    adata = sc.read_visium(file_fold, count_file=f'{dataset}_filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()
    
    # 读取 Ground Truth
    truth_df = pd.read_csv(f'{file_fold}_truth.txt', sep='\t', header=None, index_col=0)
    truth_df.columns = ['ground_truth']
    adata.obs['ground_truth'] = truth_df.loc[adata.obs_names, 'ground_truth']
    
    # 核心清洗：彻底解决 NaN 和 8 簇问题
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    invalid_labels = ['nan', 'na', 'none', '']
    adata = adata[~adata.obs['ground_truth'].astype(str).str.lower().isin(invalid_labels)]
    
    n_clusters = adata.obs['ground_truth'].nunique()
    
    # 预处理
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    adata = adata[:, adata.var['highly_variable']]
    
    # 构建 STAGATE 特有的空间网络
    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=150)
    
    return adata, n_clusters

# ========== Main Running Logic ==========
def main():
    print(f">>> STAGATE Stability Task Started. Root: {BASE_SAVE_PATH}")
    
    for dataset in DATASETS:
        dataset_dir = f"{BASE_SAVE_PATH}/{dataset}/STAGATE"
        os.makedirs(dataset_dir, exist_ok=True)
        
        print(f"\n[Processing Dataset: {dataset}]")
        try:
            adata_raw, n_clusters = load_and_preprocess(dataset)
            print(f"  - Labels cleaned. n_clusters: {n_clusters}")
        except Exception as e:
            print(f"  - Error loading {dataset}: {e}")
            continue

        results = []
        
        for seed in range(N_SEEDS):
            model_save_path = f"{dataset_dir}/model_seed{seed}.pth"
            
            # 断点续传逻辑
            if os.path.exists(model_save_path):
                continue
            
            if seed % 10 == 0:
                print(f"  - Seed {seed}/{N_SEEDS}", flush=True)
                
            try:
                adata = adata_raw.copy()
                
                # 运行训练 (注意：需要你之前在 STAGATE.train_STAGATE 源码末尾加上 return adata, model)
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    adata, model = STAGATE.train_STAGATE(
                        adata, 
                        random_seed=seed, 
                        device=DEVICE, 
                        return_model=True # 依赖源码修改
                    )
                    
                    # 聚类
                    adata = mclust_R(adata, num_cluster=n_clusters, random_seed=seed)
                
                # 计算 ARI
                ari = adjusted_rand_score(adata.obs['mclust'], adata.obs['ground_truth'])
                results.append({'seed': seed, 'ARI': ari})
                
                # 保存模型参数
                torch.save(model.state_dict(), model_save_path)
                
                # 可选：保存嵌入以备后续分析
                # np.save(f"{dataset_dir}/latent_seed{seed}.npy", adata.obsm['STAGATE'])

            except Exception as e:
                print(f"  - Seed {seed} failed: {e}")
                results.append({'seed': seed, 'ARI': np.nan})

            # 每 100 个 seed 备份一次汇总表
            if seed % 100 == 99:
                pd.DataFrame(results).to_csv(f"{dataset_dir}/ari_checkpoint.csv", index=False)

        # 最终保存该数据集的汇总表
        final_df = pd.DataFrame(results)
        final_df.to_csv(f"{dataset_dir}/seed_ari_summary.csv", index=False)
        print(f"  - Dataset {dataset} finished. Mean ARI: {final_df['ARI'].mean():.4f}")

if __name__ == '__main__':
    main()