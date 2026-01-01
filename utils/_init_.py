"""
工具函数模块
提供数据加载、预处理和评估功能
"""

from .data_utils import (
    load_data, 
    preprocess_features, 
    preprocess_adj, 
    construct_batch_data,
    load_new_dataset
)

from .metrics import (
    hr, 
    ndcg, 
    mrr, 
    auc, 
    get_top_k_scores,
    evaluate_all_metrics
)

__version__ = "1.0.0"
__all__ = [
    # 数据工具
    'load_data',
    'preprocess_features',
    'preprocess_adj',
    'construct_batch_data',
    'load_new_dataset',
    
    # 评估指标
    'hr',
    'ndcg',
    'mrr',
    'auc',
    'get_top_k_scores',
    'evaluate_all_metrics'
]

def get_available_metrics():
    """获取可用的评估指标"""
    return ['hr', 'ndcg', 'mrr', 'auc']

def print_version():
    """打印版本信息"""
    print(f"ACKRec Utils Version: {__version__}")