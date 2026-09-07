#!/usr/bin/env python3
"""
集成两个模型预测结果的脚本
例如：
模型1: dataset2_5091.csv (分数: 0.5091)
模型2: dataset2_5055.csv (分数: 0.5055)
"""

import fire
import numpy as np
import pandas as pd
import os

def load_predictions(file_path):
    """加载预测结果"""
    print(f"加载预测文件: {file_path}")
    df = pd.read_csv(file_path, header=None)
    preds = df.values.astype(np.float32)
    print(f"  形状: {preds.shape}, 范围: [{preds.min():.8f}, {preds.max():.8f}]")
    return preds

def ensemble_predictions(pred1, pred2, method='weighted', weights=None):
    """集成两个预测结果"""
    
    if method == 'average':
        # 简单平均
        print("使用简单平均集成")
        return (pred1 + pred2) / 2.0
    
    elif method == 'weighted':
        # 加权平均（基于测试集分数）
        if weights is None:
            # 使用测试集分数作为权重
            score1, score2 = 0.5, 0.5
            total = score1 + score2
            w1 = score1 / total
            w2 = score2 / total
        else:
            w1, w2 = weights
        
        print(f"使用加权平均集成: w1={w1:.4f}, w2={w2:.4f}")
        return w1 * pred1 + w2 * pred2
    
    elif method == 'geometric':
        # 几何平均
        print("使用几何平均集成")
        return np.sqrt(pred1 * pred2 + 1e-8)
    
    elif method == 'rank':
        # 基于排名的集成
        print("使用排名集成")
        N, M = pred1.shape
        ens_ranks = np.zeros((N, M))
        
        for i in range(N):
            # 获取每个预测的排名（分数越高排名越靠前）
            rank1 = np.argsort(np.argsort(-pred1[i]))
            rank2 = np.argsort(np.argsort(-pred2[i]))
            
            # 平均排名
            avg_rank = (rank1 + rank2) / 2.0
            
            # 将平均排名转换回分数（排名越小分数越高）
            ens_ranks[i] = 1.0 / (avg_rank + 1.0)
        
        return ens_ranks
    
    else:
        raise ValueError(f"未知的集成方法: {method}")

def save_predictions(preds, output_path):
    """保存预测结果"""
    print(f"保存集成结果到: {output_path}")
    print(f"  形状: {preds.shape}, 范围: [{preds.min():.6f}, {preds.max():.6f}]")
    
    # 使用pandas保存，保持与原始文件一致的格式
    df = pd.DataFrame(preds)
    df.to_csv(output_path, header=False, index=False)
    

def evaluate_on_validation(pred1, pred2, val_labels_path=None):
    """在验证集上评估（可选）"""
    if val_labels_path is None:
        print("未提供验证集标签，跳过验证集评估")
        return None, None
    
    try:
        val_labels = np.load(val_labels_path)
        print(f"验证集标签形状: {val_labels.shape}")
        
        # 简单评估
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        def eval_pred(pred, labels):
            aucs, aps = [], []
            for i in range(len(pred)):
                pos_scores = pred[i][labels[i] == 1]
                neg_scores = pred[i][labels[i] == 0]
                if len(pos_scores) > 0 and len(neg_scores) > 0:
                    y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
                    y_score = np.concatenate([pos_scores, neg_scores])
                    aucs.append(roc_auc_score(y_true, y_score))
                    aps.append(average_precision_score(y_true, y_score))
            return np.mean(aucs), np.mean(aps)
        
        auc1, ap1 = eval_pred(pred1, val_labels)
        auc2, ap2 = eval_pred(pred2, val_labels)
        
        print(f"模型1 (0.5091): AUC={auc1:.4f}, AP={ap1:.4f}")
        print(f"模型2 (0.5055): AUC={auc2:.4f}, AP={ap2:.4f}")
        
        return (auc1, ap1), (auc2, ap2)
    except Exception as e:
        print(f"验证集评估失败: {e}")
        return None, None

def main(dataset='dataset1', score1='8435', score2='8455', base_url='./saved_result'):
    """
    排名集成两个模型预测结果的入口。

    自动从 saved_result/{dataset}/ 下查找第 1 与第 3 个 epoch 的预测文件：
      file1 = {base_url}/{dataset}/0.{score1}_epoch1/{dataset}_result.csv
      file2 = {base_url}/{dataset}/0.{score2}_epoch3/{dataset}_result.csv
    无需手动重命名文件。

    命令行示例:
      python code/ensemble.py --dataset dataset1 --score1 9401 --score2 9436
      python code/ensemble.py --dataset dataset2 --score1 5924 --score2 6103
    """
    # 自动查找结果文件
    file1 = f"{base_url}/{dataset}/0.{score1}_epoch1/{dataset}_result.csv"
    file2 = f"{base_url}/{dataset}/0.{score2}_epoch3/{dataset}_result.csv"
    output_dir = f"{dataset}_ensemble_results"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载预测结果
    pred1 = load_predictions(file1)
    pred2 = load_predictions(file2)
    
    # 检查形状是否一致
    if pred1.shape != pred2.shape:
        raise ValueError(f"预测结果形状不一致: {pred1.shape} vs {pred2.shape}")
    
    # 可选：在验证集上评估
    val_labels_path = None  # 如果有验证集标签，设置为路径
    eval_on_val = False  # 是否进行验证集评估
    
    if eval_on_val:
        eval_pred1, eval_pred2 = evaluate_on_validation(pred1, pred2, val_labels_path)
    
    # 尝试不同的集成方法
    #methods = ['weighted', 'average', 'geometric', 'rank']
    methods = ['rank']
    for method in methods:
        print("\n" + "="*50)
        print(f"使用集成方法: {method}")
        print("="*50)
        
        # 集成
        if method == 'weighted':
            # 使用测试集分数作为权重
            ensemble_pred = ensemble_predictions(pred1, pred2, method, weights=(float(score1)/(float(score1)+float(score2)), float(score2)/(float(score1)+float(score2))))
        else:
            ensemble_pred = ensemble_predictions(pred1, pred2, method)
        
        # 保存结果
        output_file = f"{output_dir}/{dataset}_result.csv"
        save_predictions(ensemble_pred, output_file)

if __name__ == "__main__":
    fire.Fire(main)