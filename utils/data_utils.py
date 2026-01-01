import numpy as np
import pickle as pkl
import random
import scipy.sparse as sp
import torch
import os
import warnings

def load_data(user=['uku'], item=['kuk'], data_dir='./data'):
    """
    加载数据并预处理
    
    Args:
        user: 用户支持矩阵类型列表
        item: 物品支持矩阵类型列表
        data_dir: 数据目录路径
        
    Returns:
        rating: 评分矩阵
        features_item: 物品特征
        features_user: 用户特征
        support_user: 用户支持矩阵列表
        support_item: 物品支持矩阵列表
        negative: 负样本
    """
    support_user = []
    support_item = []
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    
    try:
        # rating matrix
        rating_path = os.path.join(data_dir, 'rate_matrix.p')
        if not os.path.exists(rating_path):
            # 尝试加载样本数据
            rating_path = os.path.join(data_dir, 'sample_rate_matrix.p')
            if not os.path.exists(rating_path):
                raise FileNotFoundError(f"找不到评分矩阵文件: {rating_path}")
        
        with open(rating_path, 'rb') as source:
            rating = pkl.load(source)
            if hasattr(rating, 'todense'):
                rating = rating.todense()
            rating = np.array(rating, dtype=np.float32)
        
        # concept w2v features
        w2v_path = os.path.join(data_dir, 'concept_embedding.p')
        if os.path.exists(w2v_path):
            with open(w2v_path, 'rb') as source:
                concept_w2v = np.array(pkl.load(source))
        else:
            # 使用随机特征
            concept_w2v = np.random.randn(rating.shape[1], 50).astype(np.float32)
            warnings.warn(f"找不到词向量特征文件: {w2v_path}，使用随机特征代替")
        
        # concept bow features
        bow_path = os.path.join(data_dir, 'concept_feature_bow.p')
        if os.path.exists(bow_path):
            with open(bow_path, 'rb') as source:
                concept_bow = pkl.load(source)
                if hasattr(concept_bow, 'todense'):
                    concept_bow = concept_bow.todense()
        else:
            # 使用随机特征
            concept_bow = np.random.randn(rating.shape[1], 100).astype(np.float32)
            warnings.warn(f"找不到BOW特征文件: {bow_path}，使用随机特征代替")
        
        # 合并特征
        if concept_w2v.shape[0] == concept_bow.shape[0]:
            concept = np.hstack((concept_w2v, concept_bow))
        else:
            # 如果维度不匹配，只使用其中一个
            if concept_w2v.shape[0] == rating.shape[1]:
                concept = concept_w2v
            else:
                concept = concept_bow
            warnings.warn("特征维度不匹配，使用单个特征集")
        
        features_item = preprocess_features(concept.astype(np.float32))
        
        # user features
        uc_path = os.path.join(data_dir, 'UC.p')
        if os.path.exists(uc_path):
            with open(uc_path, 'rb') as source:
                features = pkl.load(source)
                if hasattr(features, 'todense'):
                    features = features.todense()
                features_user = preprocess_features(features.astype(np.float32))
        else:
            # 使用单位矩阵作为用户特征
            features_user = np.eye(rating.shape[0], dtype=np.float32)
            warnings.warn(f"找不到用户特征文件: {uc_path}，使用单位矩阵代替")
        
        # uku/kuk adjacency matrix
        if 'uku' in user or 'kuk' in item:
            adj_path = os.path.join(data_dir, 'adjacency_matrix.p')
            if os.path.exists(adj_path):
                with open(adj_path, 'rb') as source:
                    uk = pkl.load(source)
                    if hasattr(uk, 'todense'):
                        uk = uk.todense()
                
                if 'uku' in user:
                    uk_user = uk.dot(uk.T) + np.eye(uk.shape[0])
                    uku = preprocess_adj(uk_user)
                    support_user.append(uku)
                
                if 'kuk' in item:
                    ku_item = uk.T.dot(uk) + np.eye(uk.T.shape[0])
                    kuk = preprocess_adj(ku_item)
                    support_item.append(kuk)
            else:
                warnings.warn(f"找不到邻接矩阵文件: {adj_path}")
        
        # ucu matrix
        if 'ucu' in user:
            uc_path = os.path.join(data_dir, 'UC.p')
            if os.path.exists(uc_path):
                with open(uc_path, 'rb') as source:
                    uc = pkl.load(source)
                    if hasattr(uc, 'todense'):
                        uc = uc.todense()
                uc = uc.dot(uc.T) + np.eye(uc.shape[0])
                ucu = preprocess_adj(uc)
                support_user.append(ucu)
        
        # uctcu matrix
        if 'uctcu' in user:
            uct_path = os.path.join(data_dir, 'UCT.p')
            if os.path.exists(uct_path):
                with open(uct_path, 'rb') as source:
                    uct = pkl.load(source)
                    if hasattr(uct, 'todense'):
                        uct = uct.todense()
                uct = uct.dot(uct.T) + np.eye(uct.shape[0])
                uctcu = preprocess_adj(uct)
                support_user.append(uctcu)
            else:
                warnings.warn(f"找不到UCT矩阵文件: {uct_path}")
        
        # uvu matrix
        if 'uvu' in user:
            uv_path = os.path.join(data_dir, 'UV.p')
            if os.path.exists(uv_path):
                with open(uv_path, 'rb') as source:
                    uv = pkl.load(source)
                    if hasattr(uv, 'todense'):
                        uv = uv.todense()
                uv = uv.dot(uv.T) + np.eye(uv.shape[0])
                uvu = preprocess_adj(uv)
                support_user.append(uvu)
            else:
                warnings.warn(f"找不到UV矩阵文件: {uv_path}")
        
        # negative sample
        negative_path = os.path.join(data_dir, 'negative.p')
        if os.path.exists(negative_path):
            with open(negative_path, 'rb') as source:
                negative = np.array(pkl.load(source), dtype=np.int32)
        else:
            # 使用样本负样本
            negative_path = os.path.join(data_dir, 'sample_negative.p')
            if os.path.exists(negative_path):
                with open(negative_path, 'rb') as source:
                    negative = np.array(pkl.load(source), dtype=np.int32)
            else:
                # 创建虚拟负样本
                negative = np.zeros((rating.shape[0], 100, 2), dtype=np.int32)
                warnings.warn(f"找不到负样本文件，使用虚拟负样本")
        
        # 转换为PyTorch张量
        rating = torch.FloatTensor(rating)
        features_item = torch.FloatTensor(features_item)
        features_user = torch.FloatTensor(features_user)
        
        # 处理支持矩阵
        support_user_tensors = []
        for sup in support_user:
            if sup is not None:
                support_user_tensors.append(torch.FloatTensor(sup))
        
        support_item_tensors = []
        for sup in support_item:
            if sup is not None:
                support_item_tensors.append(torch.FloatTensor(sup))
        
        negative = torch.LongTensor(negative)
        
        print(f"✅ 数据加载成功:")
        print(f"   评分矩阵: {rating.shape}")
        print(f"   用户特征: {features_user.shape}")
        print(f"   物品特征: {features_item.shape}")
        print(f"   负样本: {negative.shape}")
        print(f"   用户支持矩阵: {len(support_user_tensors)} 个")
        print(f"   物品支持矩阵: {len(support_item_tensors)} 个")
        
        return rating, features_item, features_user, support_user_tensors, support_item_tensors, negative
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        raise

def load_new_dataset(data_dir='./processed_data'):
    """
    加载新数据集
    
    Args:
        data_dir: 处理后的数据目录
        
    Returns:
        与load_data相同的返回值
    """
    return load_data(user=['uku'], item=['kuk'], data_dir=data_dir)

def preprocess_features(features):
    """
    特征归一化
    
    Args:
        features: 输入特征矩阵
        
    Returns:
        归一化后的特征
    """
    if isinstance(features, torch.Tensor):
        features = features.numpy()
    
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    features = r_mat_inv.dot(features)
    return features

def preprocess_adj(adjacency):
    """
    邻接矩阵归一化
    
    Args:
        adjacency: 输入邻接矩阵
        
    Returns:
        归一化后的邻接矩阵
    """
    if isinstance(adjacency, torch.Tensor):
        adjacency = adjacency.numpy()
    
    rowsum = np.array(adjacency.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adjacency.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt) * 1e2

def construct_batch_data(features_user, features_item, rating, supports_user, supports_item, negative):
    """
    构造批处理数据字典
    
    Args:
        features_user: 用户特征
        features_item: 物品特征
        rating: 评分矩阵
        supports_user: 用户支持矩阵
        supports_item: 物品支持矩阵
        negative: 负样本
        
    Returns:
        批处理数据字典
    """
    return {
        'features_user': features_user,
        'features_item': features_item,
        'rating': rating,
        'supports_user': supports_user,
        'supports_item': supports_item,
        'negative': negative
    }

def create_dummy_data(num_users=100, num_items=50):
    """
    创建虚拟数据用于测试
    
    Args:
        num_users: 用户数量
        num_items: 物品数量
        
    Returns:
        虚拟数据
    """
    print(f"创建虚拟数据: {num_users} 用户, {num_items} 物品")
    
    # 创建评分矩阵（稀疏）
    rating = np.zeros((num_users, num_items), dtype=np.float32)
    for i in range(num_users):
        # 每个用户随机交互5-10个物品
        num_interactions = np.random.randint(5, 11)
        items = np.random.choice(num_items, num_interactions, replace=False)
        ratings = np.random.uniform(3, 5, num_interactions)
        rating[i, items] = ratings
    
    # 创建特征
    features_user = np.eye(num_users, dtype=np.float32)
    features_item = np.random.randn(num_items, 100).astype(np.float32)
    
    # 创建邻接矩阵（简单版本）
    adjacency = rating > 0
    adjacency = adjacency.astype(np.float32)
    
    # 创建支持矩阵
    uk_user = adjacency.dot(adjacency.T) + np.eye(num_users)
    uku = preprocess_adj(uk_user)
    
    ku_item = adjacency.T.dot(adjacency) + np.eye(num_items)
    kuk = preprocess_adj(ku_item)
    
    # 创建负样本
    negative = np.zeros((num_users, 100, 2), dtype=np.int32)
    for i in range(num_users):
        # 为每个用户创建99个负样本和1个正样本
        positive_items = np.where(rating[i] > 0)[0]
        if len(positive_items) > 0:
            # 选择一个正样本
            positive_idx = np.random.choice(positive_items)
            # 创建99个负样本（确保不是正样本）
            all_items = np.arange(num_items)
            negative_items = np.setdiff1d(all_items, positive_items)
            if len(negative_items) >= 99:
                selected_negatives = np.random.choice(negative_items, 99, replace=False)
            else:
                # 如果负样本不够，允许重复
                selected_negatives = np.random.choice(negative_items, 99, replace=True)
            
            # 组合：前99个负样本，最后一个正样本
            for j in range(99):
                negative[i, j] = [i, selected_negatives[j]]
            negative[i, 99] = [i, positive_idx]
    
    # 转换为张量
    rating = torch.FloatTensor(rating)
    features_item = torch.FloatTensor(features_item)
    features_user = torch.FloatTensor(features_user)
    support_user = [torch.FloatTensor(uku)]
    support_item = [torch.FloatTensor(kuk)]
    negative = torch.LongTensor(negative)
    
    return rating, features_item, features_user, support_user, support_item, negative

def save_sample_data(data_dir='./data'):
    """
    保存样本数据
    
    Args:
        data_dir: 数据目录
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # 创建虚拟数据
    rating, features_item, features_user, support_user, support_item, negative = create_dummy_data(10, 5)
    
    # 保存为样本文件
    sample_files = {
        'sample_rate_matrix.p': rating.numpy(),
        'sample_negative.p': negative.numpy(),
        'sample_UC.p': features_user.numpy(),
        'sample_concept_feature_bow.p': features_item.numpy(),
        'sample_adjacency_matrix.p': support_user[0].numpy() if support_user else np.eye(10)
    }
    
    for filename, data in sample_files.items():
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'wb') as f:
            pkl.dump(data, f)
        print(f"✅ 保存样本文件: {filepath}")
    
    print(f"🎉 样本数据已保存到 {data_dir}")