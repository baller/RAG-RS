#!/usr/bin/env python3
"""
多模态遥感Embedding使用示例

这个脚本演示了如何：
1. 加载训练好的embedding模型
2. 加载向量数据库
3. 进行跨模态检索
4. 可视化检索结果
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.embedding import EmbeddingExtractor
from data.TreeSAT import TreeSAT
from data.transforms.transform import TransformMAE
from build_vector_db import SimpleVectorDB, FAISSVectorDB


class MultiModalRetrieval:
    """多模态检索系统"""
    
    def __init__(self, model_path, db_path, db_type='simple'):
        """
        初始化检索系统
        
        Args:
            model_path: 训练好的模型路径
            db_path: 向量数据库路径
            db_type: 数据库类型 ('simple' 或 'faiss')
        """
        self.model_path = model_path
        self.db_path = db_path
        self.db_type = db_type
        
        print("加载embedding模型...")
        self.extractor = EmbeddingExtractor(model_path)
        
        print("加载向量数据库...")
        if db_type == 'simple':
            self.db = SimpleVectorDB.load(db_path)
        elif db_type == 'faiss':
            self.db = FAISSVectorDB.load(db_path)
        else:
            raise ValueError(f"不支持的数据库类型: {db_type}")
        
        print("系统初始化完成!")
    
    def encode_image(self, image_path, modality):
        """编码单张图像"""
        # 这里简化处理，实际使用时需要根据具体数据格式调整
        # 创建虚拟数据进行演示
        if modality == 'aerial':
            dummy_data = torch.randn(1, 4, 224, 224)
        elif modality == 's1':
            dummy_data = torch.randn(1, 2, 224, 224)
        elif modality == 's2':
            dummy_data = torch.randn(1, 10, 224, 224)
        else:
            raise ValueError(f"不支持的模态: {modality}")
        
        # 构造batch格式
        batch = {modality: dummy_data}
        
        # 提取embedding
        with torch.no_grad():
            embeddings = self.extractor.model(batch)
            return embeddings[modality][0]  # 返回第一个样本的embedding
    
    def cross_modal_search(self, query_embedding, query_modality, target_modality, top_k=5):
        """跨模态搜索"""
        print(f"执行跨模态搜索: {query_modality} -> {target_modality}")
        
        # 在目标模态中搜索
        results, similarities = self.db.search(
            query_embedding.numpy(), 
            target_modality, 
            top_k=top_k
        )
        
        return results, similarities
    
    def visualize_results(self, results, similarities, save_path=None):
        """可视化检索结果"""
        if not results:
            print("没有检索结果")
            return
        
        print("\n检索结果:")
        print("-" * 50)
        for i, (result, sim) in enumerate(zip(results, similarities)):
            print(f"#{i+1} (相似度: {sim:.4f})")
            print(f"  文件名: {result['name']}")
            print(f"  模态: {result['modality']}")
            print(f"  数据分割: {result.get('split', 'unknown')}")
            print()
    
    def demo_retrieval_workflow(self):
        """演示检索工作流程"""
        print("=" * 60)
        print("多模态遥感检索演示")
        print("=" * 60)
        
        # 检查数据库中的模态
        available_modalities = list(self.db.vectors.keys()) if hasattr(self.db, 'vectors') else list(self.db.metadata.keys())
        print(f"可用模态: {available_modalities}")
        
        if len(available_modalities) < 2:
            print("数据库中模态数量不足，无法进行跨模态检索")
            return
        
        # 演示场景1: aerial -> s1 检索
        if 'aerial' in available_modalities and 's1' in available_modalities:
            print("\n场景1: 航空影像 -> SAR数据检索")
            print("-" * 30)
            
            # 随机选择一个aerial样本作为查询
            aerial_vectors = self.db.vectors['aerial'] if hasattr(self.db, 'vectors') else None
            if aerial_vectors is not None and len(aerial_vectors) > 0:
                query_idx = np.random.randint(0, len(aerial_vectors))
                query_embedding = torch.tensor(aerial_vectors[query_idx])
                
                print(f"查询样本: {self.db.metadata['aerial'][query_idx]['name']}")
                
                # 执行检索
                results, similarities = self.cross_modal_search(
                    query_embedding, 'aerial', 's1', top_k=3
                )
                
                # 显示结果
                self.visualize_results(results, similarities)
        
        # 演示场景2: s2 -> aerial 检索
        if 's2' in available_modalities and 'aerial' in available_modalities:
            print("\n场景2: 多光谱数据 -> 航空影像检索")
            print("-" * 30)
            
            s2_vectors = self.db.vectors['s2'] if hasattr(self.db, 'vectors') else None
            if s2_vectors is not None and len(s2_vectors) > 0:
                query_idx = np.random.randint(0, len(s2_vectors))
                query_embedding = torch.tensor(s2_vectors[query_idx])
                
                print(f"查询样本: {self.db.metadata['s2'][query_idx]['name']}")
                
                # 执行检索
                results, similarities = self.cross_modal_search(
                    query_embedding, 's2', 'aerial', top_k=3
                )
                
                # 显示结果
                self.visualize_results(results, similarities)
    
    def compute_statistics(self):
        """计算数据库统计信息"""
        print("\n数据库统计信息:")
        print("-" * 30)
        
        if hasattr(self.db, 'vectors'):
            for modality, vectors in self.db.vectors.items():
                print(f"{modality}:")
                print(f"  向量数量: {len(vectors)}")
                print(f"  向量维度: {vectors.shape[1] if len(vectors) > 0 else 0}")
                if len(vectors) > 0:
                    similarities = np.dot(vectors, vectors.T)
                    mean_sim = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
                    print(f"  平均内部相似度: {mean_sim:.4f}")
                print()


def main():
    """主函数 - 运行演示"""
    # 示例路径（需要根据实际情况修改）
    model_path = "./outputs/checkpoints/embedding-epoch=99-val_total_loss=0.1234.ckpt"
    db_path = "./vector_database/vector_db.pkl"  # 或者 faiss_db 目录
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先训练模型，或修改model_path为正确的路径")
        return
    
    if not os.path.exists(db_path):
        print(f"向量数据库不存在: {db_path}")
        print("请先构建向量数据库，或修改db_path为正确的路径")
        return
    
    try:
        # 初始化检索系统
        retrieval_system = MultiModalRetrieval(
            model_path=model_path,
            db_path=db_path,
            db_type='simple'  # 或 'faiss'
        )
        
        # 计算统计信息
        retrieval_system.compute_statistics()
        
        # 运行演示
        retrieval_system.demo_retrieval_workflow()
        
        print("\n演示完成!")
        
    except Exception as e:
        print(f"运行演示时出错: {e}")
        print("请检查模型路径和数据库路径是否正确")


if __name__ == '__main__':
    # 一些基本的使用示例
    print("多模态遥感Embedding使用示例")
    print("=" * 40)
    
    print("\n1. 基本使用流程:")
    print("   - 训练embedding模型: python src/train_embedding.py --data_path /path/to/TreeSAT")
    print("   - 构建向量数据库: python src/build_vector_db.py --model_path /path/to/model.ckpt --data_path /path/to/TreeSAT")
    print("   - 评估模型性能: python src/evaluate_embedding.py --model_path /path/to/model.ckpt --data_path /path/to/TreeSAT")
    print("   - 运行检索演示: python src/example_usage.py")
    
    print("\n2. 高级功能:")
    print("   - 使用FAISS加速检索: --db_type faiss")
    print("   - 使用Milvus向量数据库: --db_type milvus")
    print("   - 调整embedding维度: --embed_dim 768")
    print("   - 多GPU训练: --num_devices 2")
    
    print("\n3. 开始演示...")
    main() 