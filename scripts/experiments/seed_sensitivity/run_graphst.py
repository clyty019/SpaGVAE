#!/usr/bin/env python
# =================================================================
# GraphST - 12 Datasets x 1000 Seeds Analysis (Unified & Corrected)
# =================================================================

import os
import warnings
import numpy as np
import pandas as pd
import torch
import scanpy as sc
from sklearn import metrics
from sklearn.decomposition import PCA
from GraphST import GraphST
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

rpy2.robjects.numpy2ri.activate()
warnings.filterwarnings("ignore")

# ========== 配置 ==========
DATASETS = ['151507', '151508', '151509', '151510', '151669', '151670', 
            '151671', '151672', '151673', '151674', '151675', '151676']
N_SEEDS = 1000
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BASE_DATA_PATH = '/extra/zhanglab0/SpatialTranscriptomicsData/10XVisium/DLPFC'
BASE_SAVE_PATH = '/extra/zhanglab0/INDV/zihend1/StaR'

# ========== mclust 聚类辅助函数 ==========
def mclust_R(adata, num_cluster, used_obsm='emb_pca', random_seed=2020):
    robjects.r.library("mclust")
    robjects.r['set.seed'](random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, 'EEE')
    mclust_res = np.array(res[-2])
    adata.obs['mclust'] = pd.Categorical(mclust_res.astype(str))
    return adata

# ========== 数据载入与清洗 ==========
def load_and_preprocess(dataset):
    file_fold = f'{BASE_DATA_PATH}/{dataset}'
    adata = sc.read_visium(file_fold, count_file=f'{dataset}_filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()
    
    # 读取 Ground Truth
    truth_df = pd.read_csv(f'{file_fold}_truth.txt', sep='\t', header=None, index_col=0)
    truth_df.columns = ['ground_truth']
    adata.obs['ground_truth'] = truth_df.loc[adata.obs_names, 'ground_truth']
    
    # 彻底过滤无效标签（解决 8 簇问题）
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
    
    return adata, n_clusters

# ========== 主运行逻辑 ==========
def main():
    print(f">>> GraphST Stability Analysis Task Started. Root: {BASE_SAVE_PATH}")
    
    for dataset in DATASETS:
        dataset_save_dir = f"{BASE_SAVE_PATH}/{dataset}/GraphST"
        os.makedirs(dataset_save_dir, exist_ok=True)
        
        print(f"\n[Dataset: {dataset}] Loading...")
        try:
            adata_raw, n_clusters = load_and_preprocess(dataset)
            print(f"  - Cleaned n_clusters: {n_clusters}")
        except Exception as e:
            print(f"  - Failed to load {dataset}: {e}")
            continue

        results = []
        for seed in range(N_SEEDS):
            model_save_path = f"{dataset_save_dir}/model_seed{seed}.pth"
            
            # 断点续传
            if os.path.exists(model_save_path):
                continue
            
            if seed % 10 == 0:
                print(f"  - Seed {seed}/{N_SEEDS}", flush=True)
                
            try:
                adata = adata_raw.copy()
                
                # 运行 GraphST
                # 使用 redirect_stdout 抑制训练日志
                with redirect_stdout(StringIO()):
                    st_model = GraphST.GraphST(adata, device=DEVICE, random_seed=seed)
                    adata = st_model.train()
                    
                    # 提取嵌入 + PCA
                    pca = PCA(n_components=20, random_state=seed)
                    adata.obsm['emb_pca'] = pca.fit_transform(adata.obsm['emb'])
                    
                    # 聚类
                    adata = mclust_R(adata, num_cluster=n_clusters, used_obsm='emb_pca', random_seed=seed)
                
                ARI = metrics.adjusted_rand_score(adata.obs['mclust'], adata.obs['ground_truth'])
                results.append({'seed': seed, 'ARI': ARI})
                
                # 保存模型参数 (注意 GraphST 内部模型是 st_model.model)
                torch.save(st_model.model.state_dict(), model_save_path)
                
            except Exception as e:
                print(f"  - Seed {seed} error: {e}")
                results.append({'seed': seed, 'ARI': np.nan})

            # 阶段性保存汇总表
            if seed % 100 == 99:
                pd.DataFrame(results).to_csv(f"{dataset_save_dir}/ari_checkpoint.csv", index=False)

        # 最终保存
        final_df = pd.DataFrame(results)
        final_df.to_csv(f"{dataset_save_dir}/seed_ari_summary.csv", index=False)
        print(f"  - {dataset} finished. Mean ARI: {final_df['ARI'].mean():.4f}")

if __name__ == '__main__':
    main()