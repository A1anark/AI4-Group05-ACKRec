"""
数据集准备脚本
用于处理和准备新数据集
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle as pkl
import os
import sys
from collections import defaultdict

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class NewDatasetPreparer:
    def __init__(self, raw_data_dir='./raw_data', 
                 output_dir='./processed_data'):
        """
        准备新数据集
        
        Args:
            raw_data_dir: 原始数据目录（包含CSV文件）
            output_dir: 处理后的数据输出目录
        """
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        self.user_map = {}
        self.item_map = {}
        
    def load_and_process(self):
        """加载并处理数据"""
        print("="*60)
        print("ACKRec 数据集准备工具")
        print("="*60)
        
        # 1. 确保目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 2. 加载交互数据
        interactions_path = os.path.join(self.raw_data_dir, 'interactions.csv')
        if not os.path.exists(interactions_path):
            print(f"⚠️ 找不到交互文件: {interactions_path}")
            print("请确保有以下文件在 raw_data/ 目录:")
            print("  - interactions.csv (必需)")
            print("  - users.csv (可选)")
            print("  - items.csv (可选)")
            return None
        
        print(f"📊 加载交互数据: {interactions_path}")
        try:
            interactions = pd.read_csv(interactions_path)
        except Exception as e:
            print(f"❌ 加载CSV文件失败: {e}")
            return None
        
        # 3. 创建ID映射（将原始ID映射到连续索引）
        unique_users = sorted(interactions['user_id'].unique())
        unique_items = sorted(interactions['item_id'].unique())
        
        self.user_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_map = {iid: idx for idx, iid in enumerate(unique_items)}
        
        self.num_users = len(unique_users)
        self.num_items = len(unique_items)
        
        print(f"✅ 数据加载成功")
        print(f"   用户数量: {self.num_users}")
        print(f"   物品数量: {self.num_items}")
        print(f"   交互数量: {len(interactions)}")
        
        # 4. 创建评分矩阵
        print("\n📈 创建评分矩阵...")
        rows = interactions['user_id'].map(self.user_map)
        cols = interactions['item_id'].map(self.item_map)
        
        # 如果有评分列就使用，否则用1表示交互
        if 'rating' in interactions.columns:
            values = interactions['rating'].values
            print(f"   使用评分列: rating")
        else:
            values = np.ones(len(interactions))
            print(f"   无评分列，使用默认值: 1")
        
        rating_matrix = sp.csr_matrix(
            (values, (rows, cols)), 
            shape=(self.num_users, self.num_items)
        )
        
        density = rating_matrix.nnz / (self.num_users * self.num_items) * 100
        print(f"   评分矩阵密度: {density:.4f}%")
        
        # 5. 生成负样本
        print("\n🎯 生成负样本...")
        negative_samples = self.generate_negative_samples(interactions)
        
        # 6. 创建特征
        print("\n🔧 创建特征...")
        user_features = self.create_user_features()
        item_features = self.create_item_features()
        
        # 7. 创建邻接矩阵
        print("\n🔗 创建邻接矩阵...")
        adjacency_matrices = self.create_adjacency_matrices(rating_matrix)
        
        # 8. 保存所有数据
        print("\n💾 保存处理后的数据...")
        self.save_all_data(
            rating_matrix=rating_matrix,
            negative_samples=negative_samples,
            user_features=user_features,
            item_features=item_features,
            adjacency_matrices=adjacency_matrices
        )
        
        # 9. 打印统计信息
        print("\n" + "="*60)
        print("📊 数据集统计信息")
        print("="*60)
        print(f"用户数量: {self.num_users}")
        print(f"物品数量: {self.num_items}")
        print(f"交互数量: {len(interactions)}")
        print(f"评分矩阵形状: {rating_matrix.shape}")
        print(f"评分密度: {density:.4f}%")
        print(f"用户特征形状: {user_features.shape}")
        print(f"物品特征形状: {item_features.shape}")
        print(f"负样本形状: {negative_samples.shape}")
        print(f"邻接矩阵: {list(adjacency_matrices.keys())}")
        print("="*60)
        print(f"\n✅ 数据处理完成!")
        print(f"数据已保存到: {self.output_dir}")
        
        return {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'rating_shape': rating_matrix.shape,
            'density': density
        }
    
    def generate_negative_samples(self, interactions, num_negatives=99):
        """
        生成负样本
        格式：每个用户100个样本（99负 + 1正）
        """
        print(f"   每个用户生成 {num_negatives} 个负样本 + 1 个正样本")
        
        # 获取每个用户的交互物品
        user_interactions = defaultdict(set)
        for _, row in interactions.iterrows():
            user_idx = self.user_map[row['user_id']]
            item_idx = self.item_map[row['item_id']]
            user_interactions[user_idx].add(item_idx)
        
        all_items = list(range(self.num_items))
        negative_samples = []
        
        valid_users = 0
        for user_idx in range(self.num_users):
            # 获取该用户交互过的物品
            positive_items = user_interactions.get(user_idx, set())
            
            if not positive_items:
                # 如果用户没有交互，跳过
                continue
            
            valid_users += 1
            
            # 生成负样本
            negative_candidates = []
            attempts = 0
            max_attempts = num_negatives * 10
            
            while len(negative_candidates) < num_negatives and attempts < max_attempts:
                candidate = np.random.choice(all_items)
                if candidate not in positive_items and candidate not in negative_candidates:
                    negative_candidates.append(candidate)
                attempts += 1
            
            # 如果负样本不够，用随机物品填充（允许重复）
            while len(negative_candidates) < num_negatives:
                candidate = np.random.choice(all_items)
                negative_candidates.append(candidate)
            
            # 选择一个正样本
            positive_sample = np.random.choice(list(positive_items))
            
            # 构建样本：[用户ID, 物品ID]
            samples = [[user_idx, item] for item in negative_candidates]
            samples.append([user_idx, positive_sample])  # 正样本在最后
            
            negative_samples.append(samples)
        
        print(f"   有效用户数: {valid_users}/{self.num_users}")
        
        return np.array(negative_samples, dtype=np.int32)
    
    def create_user_features(self):
        """创建用户特征"""
        # 如果有用户特征文件就加载
        user_features_path = os.path.join(self.raw_data_dir, 'users.csv')
        
        if os.path.exists(user_features_path):
            print(f"   加载用户特征: {user_features_path}")
            try:
                users_df = pd.read_csv(user_features_path)
                
                # 过滤并排序
                users_df = users_df[users_df['user_id'].isin(self.user_map.keys())]
                users_df['mapped_id'] = users_df['user_id'].map(self.user_map)
                users_df = users_df.sort_values('mapped_id')
                
                # 提取特征列（排除ID列）
                feature_cols = [col for col in users_df.columns 
                              if col not in ['user_id', 'mapped_id']]
                
                if len(feature_cols) > 0:
                    features = users_df[feature_cols].values
                    
                    # 归一化
                    features = features.astype(np.float32)
                    row_sums = features.sum(axis=1)
                    row_sums[row_sums == 0] = 1  # 避免除以0
                    features = features / row_sums[:, np.newaxis]
                    
                    print(f"   用户特征维度: {features.shape}")
                    return features
                else:
                    print("   ⚠️ 用户特征文件没有有效的特征列")
            except Exception as e:
                print(f"   ⚠️ 加载用户特征失败: {e}")
        
        # 如果没有特征文件，使用one-hot编码
        print("   使用单位矩阵作为用户特征")
        features = np.eye(self.num_users, dtype=np.float32)
        print(f"   用户特征维度: {features.shape}")
        return features
    
    def create_item_features(self):
        """创建物品特征"""
        # 类似用户特征的处理
        item_features_path = os.path.join(self.raw_data_dir, 'items.csv')
        
        if os.path.exists(item_features_path):
            print(f"   加载物品特征: {item_features_path}")
            try:
                items_df = pd.read_csv(item_features_path)
                
                # 过滤并排序
                items_df = items_df[items_df['item_id'].isin(self.item_map.keys())]
                items_df['mapped_id'] = items_df['item_id'].map(self.item_map)
                items_df = items_df.sort_values('mapped_id')
                
                # 提取特征列
                feature_cols = [col for col in items_df.columns 
                              if col not in ['item_id', 'mapped_id']]
                
                if len(feature_cols) > 0:
                    features = items_df[feature_cols].values
                    
                    # 归一化
                    features = features.astype(np.float32)
                    row_sums = features.sum(axis=1)
                    row_sums[row_sums == 0] = 1
                    features = features / row_sums[:, np.newaxis]
                    
                    print(f"   物品特征维度: {features.shape}")
                    return features
                else:
                    print("   ⚠️ 物品特征文件没有有效的特征列")
            except Exception as e:
                print(f"   ⚠️ 加载物品特征失败: {e}")
        
        # 如果没有特征文件，使用one-hot编码
        print("   使用单位矩阵作为物品特征")
        features = np.eye(self.num_items, dtype=np.float32)
        print(f"   物品特征维度: {features.shape}")
        return features
    
    def create_adjacency_matrices(self, rating_matrix):
        """创建各种邻接矩阵"""
        print("   创建邻接矩阵...")
        adjacency_matrices = {}
        
        try:
            # 1. UK矩阵 (用户-物品)
            uk_matrix = rating_matrix.copy()
            
            # 2. UKU矩阵 (用户-物品-用户)
            print("     - UKU矩阵 (用户-物品-用户)")
            uku_matrix = uk_matrix.dot(uk_matrix.T)
            # 添加自连接并归一化
            uku_matrix = self.normalize_adjacency(uku_matrix)
            adjacency_matrices['uku'] = uku_matrix
            
            # 3. KUK矩阵 (物品-用户-物品)
            print("     - KUK矩阵 (物品-用户-物品)")
            kuk_matrix = uk_matrix.T.dot(uk_matrix)
            kuk_matrix = self.normalize_adjacency(kuk_matrix)
            adjacency_matrices['kuk'] = kuk_matrix
            
            # 4. UCU矩阵 (用户特征相似度)
            print("     - UCU矩阵 (用户特征相似度)")
            try:
                with open(os.path.join(self.output_dir, 'UC.p'), 'rb') as f:
                    uc = pkl.load(f)
                    if hasattr(uc, 'todense'):
                        uc = uc.todense()
                uc = uc.dot(uc.T) + np.eye(uc.shape[0])
                ucu = self.normalize_adjacency(uc)
                adjacency_matrices['ucu'] = ucu
            except:
                print("       ⚠️ 无法创建UCU矩阵，使用UKU替代")
                adjacency_matrices['ucu'] = uku_matrix
            
            print(f"   成功创建 {len(adjacency_matrices)} 个邻接矩阵")
            
        except Exception as e:
            print(f"   ⚠️ 创建邻接矩阵时出错: {e}")
            # 创建简单的单位矩阵作为备用
            identity_user = np.eye(self.num_users)
            identity_item = np.eye(self.num_items)
            adjacency_matrices['uku'] = identity_user
            adjacency_matrices['kuk'] = identity_item
        
        return adjacency_matrices
    
    def normalize_adjacency(self, adjacency):
        """归一化邻接矩阵"""
        if sp.issparse(adjacency):
            adjacency = adjacency.toarray()
        
        # 添加自连接
        adjacency = adjacency + np.eye(adjacency.shape[0])
        
        # 对称归一化: D^(-1/2) * A * D^(-1/2)
        rowsum = np.array(adjacency.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        
        normalized = d_mat_inv_sqrt.dot(adjacency).dot(d_mat_inv_sqrt)
        
        # 缩放
        normalized = normalized * 100
        
        return normalized
    
    def save_all_data(self, rating_matrix, negative_samples, 
                     user_features, item_features, adjacency_matrices):
        """保存所有数据"""
        try:
            # 1. 保存评分矩阵
            pkl.dump(
                rating_matrix,
                open(os.path.join(self.output_dir, 'rate_matrix.p'), 'wb')
            )
            print("   ✅ 保存评分矩阵")
            
            # 2. 保存负样本
            pkl.dump(
                negative_samples,
                open(os.path.join(self.output_dir, 'negative.p'), 'wb')
            )
            print("   ✅ 保存负样本")
            
            # 3. 保存用户特征
            pkl.dump(
                user_features,
                open(os.path.join(self.output_dir, 'UC.p'), 'wb')
            )
            print("   ✅ 保存用户特征")
            
            # 4. 保存物品特征
            pkl.dump(
                item_features,
                open(os.path.join(self.output_dir, 'concept_feature_bow.p'), 'wb')
            )
            print("   ✅ 保存物品特征")
            
            # 5. 保存嵌入特征（可以用随机初始化）
            embedding_dim = min(50, item_features.shape[1])
            if item_features.shape[1] > embedding_dim:
                # 使用PCA降维
                try:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=embedding_dim)
                    concept_embedding = pca.fit_transform(item_features)
                    print(f"   ✅ 使用PCA降维到 {embedding_dim} 维")
                except:
                    concept_embedding = item_features[:, :embedding_dim]
                    print(f"   ✅ 截取前 {embedding_dim} 维特征")
            else:
                concept_embedding = item_features
            
            pkl.dump(
                concept_embedding,
                open(os.path.join(self.output_dir, 'concept_embedding.p'), 'wb')
            )
            print("   ✅ 保存嵌入特征")
            
            # 6. 保存邻接矩阵
            for name, matrix in adjacency_matrices.items():
                pkl.dump(
                    matrix,
                    open(os.path.join(self.output_dir, f'{name}_matrix.p'), 'wb')
                )
                print(f"   ✅ 保存 {name} 矩阵")
            
            # 7. 保存主要邻接矩阵（兼容旧代码）
            if 'uku' in adjacency_matrices:
                pkl.dump(
                    adjacency_matrices['uku'],
                    open(os.path.join(self.output_dir, 'adjacency_matrix.p'), 'wb')
                )
                print("   ✅ 保存主要邻接矩阵")
            
            # 8. 保存元数据
            metadata = {
                'num_users': self.num_users,
                'num_items': self.num_items,
                'user_map': self.user_map,
                'item_map': self.item_map,
                'created_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            pkl.dump(
                metadata,
                open(os.path.join(self.output_dir, 'metadata.p'), 'wb')
            )
            print("   ✅ 保存元数据")
            
            # 9. 保存文本格式的元数据（便于查看）
            with open(os.path.join(self.output_dir, 'dataset_info.txt'), 'w') as f:
                f.write("="*60 + "\n")
                f.write("ACKRec 数据集信息\n")
                f.write("="*60 + "\n\n")
                f.write(f"用户数量: {self.num_users}\n")
                f.write(f"物品数量: {self.num_items}\n")
                f.write(f"评分矩阵形状: {rating_matrix.shape}\n")
                f.write(f"评分密度: {rating_matrix.nnz / (self.num_users * self.num_items):.4f}\n")
                f.write(f"负样本形状: {negative_samples.shape}\n")
                f.write(f"用户特征形状: {user_features.shape}\n")
                f.write(f"物品特征形状: {item_features.shape}\n")
                f.write(f"邻接矩阵: {list(adjacency_matrices.keys())}\n")
                f.write(f"\n生成时间: {metadata['created_at']}\n")
            
            print("   ✅ 保存数据集信息")
            
        except Exception as e:
            print(f"   ❌ 保存数据时出错: {e}")
            raise

def main():
    """主函数"""
    print("ACKRec 数据集准备工具")
    print("-" * 40)
    
    # 检查原始数据目录
    raw_dir = './raw_data'
    if not os.path.exists(raw_dir):
        print(f"原始数据目录 '{raw_dir}' 不存在")
        print("创建示例目录结构...")
        os.makedirs(raw_dir, exist_ok=True)
        
        # 创建示例文件
        example_data = {
            'user_id': [1, 1, 2, 2, 3],
            'item_id': [101, 102, 101, 103, 102],
            'rating': [5, 3, 4, 2, 5]
        }
        example_df = pd.DataFrame(example_data)
        example_df.to_csv(os.path.join(raw_dir, 'interactions.csv'), index=False)
        
        print(f"已在 '{raw_dir}' 创建示例 interactions.csv 文件")
        print("请将您的数据文件放入该目录后重新运行")
        return
    
    # 创建准备器
    preparer = NewDatasetPreparer(
        raw_data_dir=raw_dir,
        output_dir='./processed_data'
    )
    
    # 处理数据
    try:
        stats = preparer.load_and_process()
        if stats:
            print(f"\n🎉 数据集准备完成!")
            print(f"处理后的数据保存在: ./processed_data/")
    except Exception as e:
        print(f"\n❌ 数据处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())