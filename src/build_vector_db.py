import os
import sys
import argparse
import torch
import numpy as np
import json
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.embedding import EmbeddingExtractor
from data.TreeSAT import TreeSAT
from data.transforms.transform import TransformMAE

# 向量数据库相关导入（需要安装相应的包）
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("警告: FAISS未安装，将使用简单的向量存储")

try:
    from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    print("警告: Milvus客户端未安装，Milvus功能不可用")


class SimpleVectorDB:
    """简单的向量数据库实现"""
    
    def __init__(self, embed_dim=512):
        self.embed_dim = embed_dim
        self.vectors = {}  # {modality: numpy_array}
        self.metadata = {}  # {modality: list_of_dicts}
        
    def add_vectors(self, modality, embeddings, metadata):
        """添加向量和元数据"""
        self.vectors[modality] = embeddings.numpy() if torch.is_tensor(embeddings) else embeddings
        self.metadata[modality] = metadata
        
    def search(self, query_vector, modality, top_k=10):
        """搜索最相似的向量"""
        if modality not in self.vectors:
            return [], []
        
        # 计算余弦相似度
        vectors = self.vectors[modality]
        query_norm = np.linalg.norm(query_vector)
        vector_norms = np.linalg.norm(vectors, axis=1)
        
        similarities = np.dot(vectors, query_vector) / (vector_norms * query_norm)
        
        # 获取top-k结果
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_similarities = similarities[top_indices]
        
        # 返回元数据和相似度
        results = []
        for idx, sim in zip(top_indices, top_similarities):
            result = self.metadata[modality][idx].copy()
            result['similarity'] = float(sim)
            results.append(result)
            
        return results, top_similarities.tolist()
    
    def save(self, filepath):
        """保存数据库到文件"""
        data = {
            'vectors': self.vectors,
            'metadata': self.metadata,
            'embed_dim': self.embed_dim
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath):
        """从文件加载数据库"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        db = cls(data['embed_dim'])
        db.vectors = data['vectors']
        db.metadata = data['metadata']
        return db


class FAISSVectorDB:
    """基于FAISS的向量数据库"""
    
    def __init__(self, embed_dim=512):
        self.embed_dim = embed_dim
        self.indices = {}  # {modality: faiss_index}
        self.metadata = {}  # {modality: list_of_dicts}
        
    def add_vectors(self, modality, embeddings, metadata):
        """添加向量和元数据"""
        if torch.is_tensor(embeddings):
            embeddings = embeddings.numpy()
        
        # 创建FAISS索引
        index = faiss.IndexFlatIP(self.embed_dim)  # 内积索引（适用于归一化向量）
        
        # 确保向量是归一化的
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # 添加向量到索引
        index.add(embeddings.astype(np.float32))
        
        self.indices[modality] = index
        self.metadata[modality] = metadata
        
    def search(self, query_vector, modality, top_k=10):
        """搜索最相似的向量"""
        if modality not in self.indices:
            return [], []
        
        # 归一化查询向量
        query_vector = query_vector / np.linalg.norm(query_vector)
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        
        # 搜索
        similarities, indices = self.indices[modality].search(query_vector, top_k)
        similarities = similarities[0]
        indices = indices[0]
        
        # 返回元数据和相似度
        results = []
        for idx, sim in zip(indices, similarities):
            if idx < len(self.metadata[modality]):
                result = self.metadata[modality][idx].copy()
                result['similarity'] = float(sim)
                results.append(result)
            
        return results, similarities.tolist()
    
    def save(self, dirpath):
        """保存数据库到目录"""
        os.makedirs(dirpath, exist_ok=True)
        
        # 保存FAISS索引
        for modality, index in self.indices.items():
            faiss.write_index(index, os.path.join(dirpath, f'{modality}.index'))
        
        # 保存元数据
        metadata_file = os.path.join(dirpath, 'metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        # 保存配置
        config = {'embed_dim': self.embed_dim}
        config_file = os.path.join(dirpath, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load(cls, dirpath):
        """从目录加载数据库"""
        # 加载配置
        config_file = os.path.join(dirpath, 'config.json')
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        db = cls(config['embed_dim'])
        
        # 加载元数据
        metadata_file = os.path.join(dirpath, 'metadata.json')
        with open(metadata_file, 'r', encoding='utf-8') as f:
            db.metadata = json.load(f)
        
        # 加载FAISS索引
        for modality in db.metadata.keys():
            index_file = os.path.join(dirpath, f'{modality}.index')
            if os.path.exists(index_file):
                db.indices[modality] = faiss.read_index(index_file)
        
        return db


class MilvusVectorDB:
    """基于Milvus的向量数据库"""
    
    def __init__(self, embed_dim=512, host='localhost', port='19530'):
        self.embed_dim = embed_dim
        self.host = host
        self.port = port
        self.collections = {}
        
        # 连接到Milvus
        connections.connect("default", host=host, port=port)
        
    def _create_collection(self, modality):
        """创建集合"""
        collection_name = f"multimodal_rs_{modality}"
        
        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embed_dim),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="modality", dtype=DataType.VARCHAR, max_length=50)
        ]
        
        # 创建集合架构
        schema = CollectionSchema(fields, f"Multi-modal RS embeddings for {modality}")
        
        # 如果集合已存在，先删除
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
        
        # 创建集合
        collection = Collection(collection_name, schema)
        
        # 创建索引
        index_params = {
            "metric_type": "IP",  # 内积
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index("embedding", index_params)
        
        self.collections[modality] = collection
        return collection
    
    def add_vectors(self, modality, embeddings, metadata):
        """添加向量和元数据"""
        collection = self._create_collection(modality)
        
        if torch.is_tensor(embeddings):
            embeddings = embeddings.numpy()
        
        # 归一化向量
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # 准备数据
        data = []
        for i, meta in enumerate(metadata):
            data.append({
                "embedding": embeddings[i].tolist(),
                "name": meta['name'],
                "label": json.dumps(meta['label']),
                "modality": modality
            })
        
        # 批量插入
        batch_size = 1000
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            collection.insert(batch)
        
        # 刷新数据
        collection.flush()
        
        # 加载集合
        collection.load()
    
    def search(self, query_vector, modality, top_k=10):
        """搜索最相似的向量"""
        collection_name = f"multimodal_rs_{modality}"
        if not utility.has_collection(collection_name):
            return [], []
        
        collection = Collection(collection_name)
        collection.load()
        
        # 归一化查询向量
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        # 搜索参数
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        
        # 执行搜索
        results = collection.search(
            data=[query_vector.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["name", "label", "modality"]
        )
        
        # 解析结果
        search_results = []
        similarities = []
        
        for hits in results:
            for hit in hits:
                result = {
                    'name': hit.entity.get('name'),
                    'label': json.loads(hit.entity.get('label')),
                    'modality': hit.entity.get('modality'),
                    'similarity': float(hit.score)
                }
                search_results.append(result)
                similarities.append(float(hit.score))
        
        return search_results, similarities


def extract_and_build_database(model_path, data_path, output_dir, db_type='simple', **kwargs):
    """提取embedding并构建向量数据库"""
    
    # 动态从TreeSAT标签文件中读取所有类别
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from data.utils import get_treesat_classes
    
    classes = get_treesat_classes(data_path, verbose=False)
    
    # 数据变换
    transform = TransformMAE(p=0.0, size=224)  # 不使用增强
    
    # 模态设置
    modalities = ['aerial', 's1', 's2']
    
    print("提取embeddings...")
    all_embeddings = {}
    all_metadata = {}
    
    # 为每个split提取embeddings
    for split in ['train', 'val', 'test']:
        print(f"处理{split}集...")
        
        # 创建数据集
        dataset = TreeSAT(
            path=data_path,
            modalities=modalities,
            transform=transform,
            split=split,
            classes=classes,
            partition=1.0
        )
        
        # 创建数据加载器
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=kwargs.get('batch_size', 32),
            shuffle=False,
            num_workers=4,
            collate_fn=dataset.collate_fn
        )
        
        # 创建embedding提取器
        extractor = EmbeddingExtractor(model_path)
        
        # 提取embeddings
        embeddings, labels, names = extractor.extract_embeddings(dataloader, modalities)
        
        # 整理元数据
        for modality in modalities:
            if modality not in all_embeddings:
                all_embeddings[modality] = []
                all_metadata[modality] = []
            
            if modality in embeddings and embeddings[modality].numel() > 0:
                all_embeddings[modality].append(embeddings[modality])
                
                # 构建元数据
                for i, (name, label) in enumerate(zip(names, labels)):
                    metadata = {
                        'name': name,
                        'label': label.tolist(),
                        'split': split,
                        'modality': modality
                    }
                    all_metadata[modality].append(metadata)
    
    # 合并所有split的数据
    for modality in modalities:
        if all_embeddings[modality]:
            all_embeddings[modality] = torch.cat(all_embeddings[modality], dim=0)
        else:
            all_embeddings[modality] = torch.empty(0, kwargs.get('embed_dim', 512))
    
    print("构建向量数据库...")
    
    # 根据类型创建数据库
    if db_type == 'simple':
        db = SimpleVectorDB(kwargs.get('embed_dim', 512))
        save_path = os.path.join(output_dir, 'vector_db.pkl')
    elif db_type == 'faiss' and FAISS_AVAILABLE:
        db = FAISSVectorDB(kwargs.get('embed_dim', 512))
        save_path = os.path.join(output_dir, 'faiss_db')
    elif db_type == 'milvus' and MILVUS_AVAILABLE:
        db = MilvusVectorDB(
            kwargs.get('embed_dim', 512),
            kwargs.get('milvus_host', 'localhost'),
            kwargs.get('milvus_port', '19530')
        )
        save_path = None  # Milvus不需要本地保存
    else:
        print(f"数据库类型 {db_type} 不可用，使用简单数据库")
        db = SimpleVectorDB(kwargs.get('embed_dim', 512))
        save_path = os.path.join(output_dir, 'vector_db.pkl')
    
    # 添加向量到数据库
    for modality in modalities:
        if all_embeddings[modality].numel() > 0:
            print(f"添加{modality}模态的{len(all_embeddings[modality])}个向量...")
            db.add_vectors(modality, all_embeddings[modality], all_metadata[modality])
    
    # 保存数据库
    if save_path:
        print(f"保存数据库到: {save_path}")
        db.save(save_path)
    
    # 保存统计信息
    stats = {
        'total_vectors': {mod: len(emb) for mod, emb in all_embeddings.items()},
        'embed_dim': kwargs.get('embed_dim', 512),
        'modalities': modalities,
        'classes': classes,
        'db_type': db_type
    }
    
    stats_file = os.path.join(output_dir, 'database_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print("数据库构建完成!")
    print("统计信息:")
    for mod, count in stats['total_vectors'].items():
        print(f"  {mod}: {count} 个向量")
    
    return db, stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='构建多模态遥感向量数据库')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型检查点路径')
    parser.add_argument('--data_path', type=str, required=True,
                       help='TreeSAT数据集路径')
    parser.add_argument('--output_dir', type=str, default='./vector_database',
                       help='向量数据库输出目录')
    parser.add_argument('--db_type', type=str, default='simple',
                       choices=['simple', 'faiss', 'milvus'],
                       help='向量数据库类型')
    parser.add_argument('--embed_dim', type=int, default=512,
                       help='Embedding维度')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    
    # Milvus相关参数
    parser.add_argument('--milvus_host', type=str, default='localhost',
                       help='Milvus服务器地址')
    parser.add_argument('--milvus_port', type=str, default='19530',
                       help='Milvus服务器端口')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查依赖
    if args.db_type == 'faiss' and not FAISS_AVAILABLE:
        print("错误: FAISS未安装，请运行: pip install faiss-cpu 或 pip install faiss-gpu")
        return
    
    if args.db_type == 'milvus' and not MILVUS_AVAILABLE:
        print("错误: Milvus客户端未安装，请运行: pip install pymilvus")
        return
    
    # 提取embeddings并构建数据库
    db, stats = extract_and_build_database(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        db_type=args.db_type,
        embed_dim=args.embed_dim,
        batch_size=args.batch_size,
        milvus_host=args.milvus_host,
        milvus_port=args.milvus_port
    )
    
    print(f"\n向量数据库已成功构建并保存到: {args.output_dir}")


if __name__ == '__main__':
    main() 