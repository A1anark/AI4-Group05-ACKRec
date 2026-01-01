import numpy as np
import warnings

def get_top_k_scores(rate, negative, k=100):
    """
    提取负样本评分并获取top-k
    
    Args:
        rate: 评分矩阵 (user_dim, item_dim)
        negative: 负样本矩阵 (length, 100)
        k: top-k值
        
    Returns:
        test_scores: 测试分数
        topk_indices: top-k索引
    """
    length = negative.shape[0]
    test_scores = []
    
    for i in range(length):
        user_indices = negative[i, :, 0]
        item_indices = negative[i, :, 1]
        
        # 确保索引在有效范围内
        valid_users = (user_indices >= 0) & (user_indices < rate.shape[0])
        valid_items = (item_indices >= 0) & (item_indices < rate.shape[1])
        valid_mask = valid_users & valid_items
        
        if np.any(~valid_mask):
            warnings.warn(f"样本 {i} 中有无效索引，跳过无效索引")
            user_indices = user_indices[valid_mask]
            item_indices = item_indices[valid_mask]
        
        scores = rate[user_indices, item_indices]
        
        # 如果样本不足，填充为0
        if len(scores) < 100:
            scores = np.pad(scores, (0, 100 - len(scores)), 'constant')
        
        test_scores.append(scores)
    
    test_scores = np.array(test_scores)  # (length, 100)
    
    # 获取top-k索引
    if len(test_scores) > 0:
        topk_indices = np.argsort(-test_scores, axis=1)[:, :k]  # 获取top-k索引
    else:
        topk_indices = np.array([])
    
    return test_scores, topk_indices

def hr(rate, negative, length, k=5):
    """
    Hit Ratio @ K
    
    Args:
        rate: 评分矩阵 (user_dim, item_dim)
        negative: 负样本矩阵 (length, 100)
        length: 样本长度
        k: top-k值
        
    Returns:
        hit ratio值
    """
    if length == 0:
        return 0.0
    
    try:
        test_scores, topk_indices = get_top_k_scores(rate, negative, k)
        
        if len(topk_indices) == 0:
            return 0.0
        
        # 检查正样本是否在top-k中（正样本在最后一个位置）
        is_in = (topk_indices == 99)  # 正样本在索引99
        
        row_hits = np.sum(is_in, axis=1)
        total_hits = np.sum(row_hits)
        
        return total_hits / length if length > 0 else 0.0
    except Exception as e:
        warnings.warn(f"计算HR@{k}时出错: {e}")
        return 0.0

def ndcg(rate, negative, length, k=5):
    """
    Normalized Discounted Cumulative Gain @ K
    
    Args:
        rate: 评分矩阵 (user_dim, item_dim)
        negative: 负样本矩阵 (length, 100)
        length: 样本长度
        k: top-k值
        
    Returns:
        NDCG值
    """
    if length == 0:
        return 0.0
    
    try:
        test_scores, topk_indices = get_top_k_scores(rate, negative, k)
        
        if len(topk_indices) == 0:
            return 0.0
        
        # 计算DCG
        dcg_sum = 0
        for i in range(length):
            # 找到正样本的位置
            pos = np.where(topk_indices[i] == 99)[0]
            if len(pos) > 0:
                rank = pos[0] + 1  # 位置从1开始计数
                dcg_sum += np.log(2) / np.log(rank + 1)
        
        # 计算IDCG（假设正样本在第一位）
        idcg_sum = 0
        for r in range(1, min(k, 1) + 1):
            idcg_sum += np.log(2) / np.log(r + 1)
        
        return dcg_sum / length if idcg_sum > 0 and length > 0 else 0.0
    except Exception as e:
        warnings.warn(f"计算NDCG@{k}时出错: {e}")
        return 0.0

def mrr(rate, negative, length):
    """
    Mean Reciprocal Rank
    
    Args:
        rate: 评分矩阵 (user_dim, item_dim)
        negative: 负样本矩阵 (length, 100)
        length: 样本长度
        
    Returns:
        MRR值
    """
    if length == 0:
        return 0.0
    
    try:
        test_scores, topk_indices = get_top_k_scores(rate, negative, 100)
        
        if len(topk_indices) == 0:
            return 0.0
        
        mrr_sum = 0
        for i in range(length):
            pos = np.where(topk_indices[i] == 99)[0]
            if len(pos) > 0:
                rank = pos[0] + 1
                mrr_sum += 1.0 / rank
        
        return mrr_sum / length if length > 0 else 0.0
    except Exception as e:
        warnings.warn(f"计算MRR时出错: {e}")
        return 0.0

def auc(rate, negative, length):
    """
    Area Under Curve
    
    Args:
        rate: 评分矩阵 (user_dim, item_dim)
        negative: 负样本矩阵 (length, 100)
        length: 样本长度
        
    Returns:
        AUC值
    """
    if length == 0:
        return 0.0
    
    try:
        test_scores, topk_indices = get_top_k_scores(rate, negative, 100)
        
        if len(topk_indices) == 0:
            return 0.5  # 随机猜测的AUC
        
        auc_sum = 0
        for i in range(length):
            pos = np.where(topk_indices[i] == 99)[0]
            if len(pos) > 0:
                rank = pos[0]
                auc_sum += (100 - rank) / 100.0
            else:
                # 如果不在top-100中，随机猜测
                auc_sum += 0.5
        
        return auc_sum / length if length > 0 else 0.5
    except Exception as e:
        warnings.warn(f"计算AUC时出错: {e}")
        return 0.5

def evaluate_all_metrics(rate, negative, length, k_list=[1, 5, 10, 20]):
    """
    计算所有评估指标
    
    Args:
        rate: 评分矩阵
        negative: 负样本矩阵
        length: 样本长度
        k_list: 要计算的k值列表
        
    Returns:
        包含所有指标的字典
    """
    metrics = {}
    
    # 计算HR@K
    for k in k_list:
        metrics[f'hr@{k}'] = hr(rate, negative, length, k=k)
    
    # 计算NDCG@K
    for k in k_list:
        if k in [5, 10, 20]:  # 通常计算这些k值的NDCG
            metrics[f'ndcg@{k}'] = ndcg(rate, negative, length, k=k)
    
    # 计算MRR和AUC
    metrics['mrr'] = mrr(rate, negative, length)
    metrics['auc'] = auc(rate, negative, length)
    
    return metrics

def print_metrics(metrics, prefix=""):
    """
    打印评估指标
    
    Args:
        metrics: 指标字典
        prefix: 前缀字符串
    """
    if not metrics:
        print(f"{prefix}无评估指标")
        return
    
    print(f"{prefix}评估结果:")
    print(f"{prefix}{'-'*30}")
    
    # 按类型分组打印
    hr_metrics = {k: v for k, v in metrics.items() if k.startswith('hr@')}
    ndcg_metrics = {k: v for k, v in metrics.items() if k.startswith('ndcg@')}
    other_metrics = {k: v for k, v in metrics.items() if not k.startswith('hr@') and not k.startswith('ndcg@')}
    
    if hr_metrics:
        for k in sorted(hr_metrics.keys(), key=lambda x: int(x.split('@')[1])):
            print(f"{prefix}{k:10}: {hr_metrics[k]:.4f}")
    
    if ndcg_metrics:
        for k in sorted(ndcg_metrics.keys(), key=lambda x: int(x.split('@')[1])):
            print(f"{prefix}{k:10}: {ndcg_metrics[k]:.4f}")
    
    if other_metrics:
        for k, v in other_metrics.items():
            print(f"{prefix}{k:10}: {v:.4f}")
    
    print(f"{prefix}{'-'*30}")

def compute_ranking_metrics(predictions, ground_truth, k=10):
    """
    计算排名指标
    
    Args:
        predictions: 预测评分
        ground_truth: 真实评分
        k: top-k值
        
    Returns:
        排名指标字典
    """
    # 确保输入是numpy数组
    if hasattr(predictions, 'cpu'):
        predictions = predictions.cpu().numpy()
    if hasattr(ground_truth, 'cpu'):
        ground_truth = ground_truth.cpu().numpy()
    
    # 获取top-k预测
    top_k_indices = np.argsort(-predictions, axis=1)[:, :k]
    
    metrics = {}
    
    # 计算Precision@K
    precision_sum = 0
    for i in range(len(ground_truth)):
        relevant_items = np.where(ground_truth[i] > 0)[0]
        recommended_items = top_k_indices[i]
        intersection = len(np.intersect1d(relevant_items, recommended_items))
        precision_sum += intersection / k if k > 0 else 0
    
    metrics[f'precision@{k}'] = precision_sum / len(ground_truth) if len(ground_truth) > 0 else 0
    
    # 计算Recall@K
    recall_sum = 0
    for i in range(len(ground_truth)):
        relevant_items = np.where(ground_truth[i] > 0)[0]
        recommended_items = top_k_indices[i]
        intersection = len(np.intersect1d(relevant_items, recommended_items))
        recall_sum += intersection / len(relevant_items) if len(relevant_items) > 0 else 0
    
    metrics[f'recall@{k}'] = recall_sum / len(ground_truth) if len(ground_truth) > 0 else 0
    
    return metrics