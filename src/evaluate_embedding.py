import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.embedding import EmbeddingExtractor
from data.TreeSAT import TreeSAT
from data.datamodule import DataModule
from data.transforms.transform import TransformMAE


def extract_embeddings(model_path, data_path, split='test', batch_size=32):
    """提取embedding"""
    print(f"从{split}集提取embeddings...")
    
    # 动态从TreeSAT标签文件中读取所有类别
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from data.utils import get_treesat_classes
    
    classes = get_treesat_classes(data_path, verbose=False)
    
    # 数据变换
    transform = TransformMAE(p=0.0, size=224)  # 评估时不使用增强
    
    # 创建数据集
    dataset = TreeSAT(
        path=data_path,
        modalities=['aerial', 's1', 's2'],
        transform=transform,
        split=split,
        classes=classes,
        partition=1.0
    )
    
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=dataset.collate_fn
    )
    
    # 创建embedding提取器
    extractor = EmbeddingExtractor(model_path)
    
    # 提取embeddings
    embeddings, labels, names = extractor.extract_embeddings(dataloader)
    
    return embeddings, labels, names, classes


def evaluate_retrieval(embeddings, labels, top_k=[1, 5, 10]):
    """评估跨模态检索性能"""
    print("评估跨模态检索性能...")
    
    results = {}
    modalities = list(embeddings.keys())
    
    for query_mod in modalities:
        for target_mod in modalities:
            if query_mod == target_mod:
                continue
                
            query_emb = embeddings[query_mod]
            target_emb = embeddings[target_mod]
            
            # 计算相似度矩阵
            similarity = torch.mm(query_emb, target_emb.t())
            
            # 计算Top-K准确率
            retrieval_results = {}
            for k in top_k:
                _, top_indices = torch.topk(similarity, k, dim=1)
                
                correct = 0
                total = len(query_emb)
                
                for i in range(total):
                    if i in top_indices[i]:  # 正确的匹配应该是同一个样本
                        correct += 1
                
                accuracy = correct / total
                retrieval_results[f'top_{k}'] = accuracy
            
            results[f'{query_mod}_to_{target_mod}'] = retrieval_results
    
    return results


def evaluate_classification(embeddings, labels, classes):
    """评估分类性能"""
    print("评估分类性能...")
    
    results = {}
    
    # 将多标签转换为单标签（选择第一个标签）
    single_labels = []
    for label in labels:
        class_indices = torch.where(label == 1)[0]
        if len(class_indices) > 0:
            single_labels.append(class_indices[0].item())
        else:
            single_labels.append(0)  # 默认类别
    
    single_labels = np.array(single_labels)
    
    # 为每个模态评估分类性能
    for modality, emb in embeddings.items():
        if emb.numel() == 0:
            continue
            
        X = emb.numpy()
        y = single_labels
        
        # 简单的train/test分割
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # KNN分类器
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        knn_score = knn.score(X_test, y_test)
        
        # 逻辑回归分类器
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        lr_score = lr.score(X_test, y_test)
        
        results[modality] = {
            'knn_accuracy': knn_score,
            'lr_accuracy': lr_score
        }
    
    return results


def visualize_embeddings(embeddings, labels, classes, output_dir):
    """可视化embeddings"""
    print("可视化embeddings...")
    
    # 将多标签转换为单标签用于可视化
    single_labels = []
    for label in labels:
        class_indices = torch.where(label == 1)[0]
        if len(class_indices) > 0:
            single_labels.append(class_indices[0].item())
        else:
            single_labels.append(0)
    
    single_labels = np.array(single_labels)
    
    # 为每个模态创建t-SNE可视化
    fig, axes = plt.subplots(1, len(embeddings), figsize=(6*len(embeddings), 6))
    if len(embeddings) == 1:
        axes = [axes]
    
    for idx, (modality, emb) in enumerate(embeddings.items()):
        if emb.numel() == 0:
            continue
            
        # 运行t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        emb_2d = tsne.fit_transform(emb.numpy())
        
        # 绘制散点图
        scatter = axes[idx].scatter(
            emb_2d[:, 0], emb_2d[:, 1], 
            c=single_labels, 
            cmap='tab20', 
            alpha=0.7,
            s=20
        )
        axes[idx].set_title(f'{modality.upper()} Embeddings (t-SNE)')
        axes[idx].set_xlabel('t-SNE 1')
        axes[idx].set_ylabel('t-SNE 2')
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=axes[idx])
        cbar.set_label('Tree Species')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'embedding_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()


def compute_modality_alignment(embeddings):
    """计算模态对齐质量"""
    print("计算模态对齐质量...")
    
    alignment_scores = {}
    modalities = list(embeddings.keys())
    
    for i in range(len(modalities)):
        for j in range(i+1, len(modalities)):
            mod1, mod2 = modalities[i], modalities[j]
            emb1, emb2 = embeddings[mod1], embeddings[mod2]
            
            if emb1.numel() == 0 or emb2.numel() == 0:
                continue
            
            # 计算余弦相似度
            similarities = torch.nn.functional.cosine_similarity(emb1, emb2, dim=1)
            
            alignment_scores[f'{mod1}_{mod2}'] = {
                'mean_similarity': similarities.mean().item(),
                'std_similarity': similarities.std().item(),
                'min_similarity': similarities.min().item(),
                'max_similarity': similarities.max().item()
            }
    
    return alignment_scores


def save_results(results, output_dir):
    """保存评估结果"""
    results_file = os.path.join(output_dir, 'evaluation_results.txt')
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("多模态Embedding评估结果\n")
        f.write("=" * 50 + "\n\n")
        
        # 跨模态检索结果
        if 'retrieval' in results:
            f.write("跨模态检索性能:\n")
            f.write("-" * 30 + "\n")
            for pair, metrics in results['retrieval'].items():
                f.write(f"{pair}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
                f.write("\n")
        
        # 分类性能结果
        if 'classification' in results:
            f.write("分类性能:\n")
            f.write("-" * 30 + "\n")
            for modality, metrics in results['classification'].items():
                f.write(f"{modality}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
                f.write("\n")
        
        # 模态对齐结果
        if 'alignment' in results:
            f.write("模态对齐质量:\n")
            f.write("-" * 30 + "\n")
            for pair, metrics in results['alignment'].items():
                f.write(f"{pair}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
                f.write("\n")
    
    print(f"评估结果已保存到: {results_file}")


def main():
    """主评估函数"""
    parser = argparse.ArgumentParser(description='多模态遥感Embedding评估')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型检查点路径')
    parser.add_argument('--data_path', type=str, required=True,
                       help='TreeSAT数据集路径')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='评估结果输出目录')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='使用的数据分割')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 提取embeddings
    embeddings, labels, names, classes = extract_embeddings(
        args.model_path, args.data_path, args.split, args.batch_size
    )
    
    print(f"提取的embedding数量:")
    for modality, emb in embeddings.items():
        print(f"  {modality}: {emb.shape}")
    
    # 评估结果容器
    results = {}
    
    # 评估跨模态检索
    try:
        retrieval_results = evaluate_retrieval(embeddings, labels)
        results['retrieval'] = retrieval_results
        print("跨模态检索评估完成")
    except Exception as e:
        print(f"跨模态检索评估失败: {e}")
    
    # 评估分类性能
    try:
        classification_results = evaluate_classification(embeddings, labels, classes)
        results['classification'] = classification_results
        print("分类性能评估完成")
    except Exception as e:
        print(f"分类性能评估失败: {e}")
    
    # 计算模态对齐质量
    try:
        alignment_results = compute_modality_alignment(embeddings)
        results['alignment'] = alignment_results
        print("模态对齐质量计算完成")
    except Exception as e:
        print(f"模态对齐质量计算失败: {e}")
    
    # 可视化embeddings
    try:
        visualize_embeddings(embeddings, labels, classes, args.output_dir)
        print("Embedding可视化完成")
    except Exception as e:
        print(f"Embedding可视化失败: {e}")
    
    # 保存结果
    save_results(results, args.output_dir)
    
    # 打印总结
    print("\n" + "=" * 50)
    print("评估总结:")
    print("=" * 50)
    
    if 'classification' in results:
        print("分类准确率:")
        for modality, metrics in results['classification'].items():
            print(f"  {modality}: KNN={metrics['knn_accuracy']:.4f}, LR={metrics['lr_accuracy']:.4f}")
    
    if 'alignment' in results:
        print("\n模态对齐相似度:")
        for pair, metrics in results['alignment'].items():
            print(f"  {pair}: {metrics['mean_similarity']:.4f} ± {metrics['std_similarity']:.4f}")
    
    print(f"\n详细结果已保存到: {args.output_dir}")


if __name__ == '__main__':
    main() 