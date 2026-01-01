import torch
import numpy as np

def glorot(shape, name=None):
    """
    Glorot初始化（Xavier初始化）
    
    Args:
        shape: 张量形状
        name: 参数名称（可选）
        
    Returns:
        初始化后的张量
    """
    # 计算初始化范围
    if len(shape) == 2:
        fan_in, fan_out = shape
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    else:
        fan_in = fan_out = np.prod(shape) // 2
    
    init_range = np.sqrt(6.0 / (fan_in + fan_out))
    
    # 生成均匀分布的随机数
    tensor = torch.empty(*shape)
    torch.nn.init.uniform_(tensor, -init_range, init_range)
    
    return tensor

def zeros(shape):
    """
    全零初始化
    
    Args:
        shape: 张量形状
        
    Returns:
        全零张量
    """
    return torch.zeros(shape, dtype=torch.float32)

def ones(shape):
    """
    全一初始化
    
    Args:
        shape: 张量形状
        
    Returns:
        全一张量
    """
    return torch.ones(shape, dtype=torch.float32)

def normal(shape, mean=0.0, std=0.01):
    """
    正态分布初始化
    
    Args:
        shape: 张量形状
        mean: 均值
        std: 标准差
        
    Returns:
        正态分布初始化的张量
    """
    tensor = torch.empty(*shape)
    torch.nn.init.normal_(tensor, mean=mean, std=std)
    return tensor

def xavier_uniform(shape):
    """
    Xavier均匀分布初始化
    
    Args:
        shape: 张量形状
        
    Returns:
        初始化的张量
    """
    tensor = torch.empty(*shape)
    torch.nn.init.xavier_uniform_(tensor)
    return tensor

def xavier_normal(shape):
    """
    Xavier正态分布初始化
    
    Args:
        shape: 张量形状
        
    Returns:
        初始化的张量
    """
    tensor = torch.empty(*shape)
    torch.nn.init.xavier_normal_(tensor)
    return tensor

def kaiming_uniform(shape):
    """
    Kaiming均匀分布初始化
    
    Args:
        shape: 张量形状
        
    Returns:
        初始化的张量
    """
    tensor = torch.empty(*shape)
    torch.nn.init.kaiming_uniform_(tensor, nonlinearity='relu')
    return tensor

def kaiming_normal(shape):
    """
    Kaiming正态分布初始化
    
    Args:
        shape: 张量形状
        
    Returns:
        初始化的张量
    """
    tensor = torch.empty(*shape)
    torch.nn.init.kaiming_normal_(tensor, nonlinearity='relu')
    return tensor

# 导出所有初始化函数
__all__ = [
    'glorot', 'zeros', 'ones', 'normal',
    'xavier_uniform', 'xavier_normal',
    'kaiming_uniform', 'kaiming_normal'
]