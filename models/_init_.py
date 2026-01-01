"""
ACKRec 模型包
包含所有模型定义和层定义
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "ACKRec Team"
__email__ = "ackrec@example.com"

# 导入初始化函数
from .inits import (
    glorot,
    zeros,
    ones,
    normal,
    xavier_uniform,
    xavier_normal,
    kaiming_uniform,
    kaiming_normal
)

# 导入层定义
from .layers import (
    GraphConvolution,
    SimpleAttLayer,
    RateLayer,
    GraphAttentionLayer
)

# 导入模型定义
from .models import (
    Model,
    GCN,
    AGCNrec
)

# 导出所有公共API
__all__ = [
    # 初始化函数
    'glorot',
    'zeros',
    'ones',
    'normal',
    'xavier_uniform',
    'xavier_normal',
    'kaiming_uniform',
    'kaiming_normal',
    
    # 层定义
    'GraphConvolution',
    'SimpleAttLayer',
    'RateLayer',
    'GraphAttentionLayer',
    
    # 模型定义
    'Model',
    'GCN',
    'AGCNrec',
    
    # 元数据
    '__version__',
    '__author__',
    '__email__'
]

# 版本信息
def get_version():
    """获取当前版本号"""
    return __version__

def get_author():
    """获取作者信息"""
    return __author__

def get_email():
    """获取联系邮箱"""
    return __email__

def print_info():
    """打印包信息"""
    info = f"""
    ACKRec Models Package
    Version: {__version__}
    Author: {__author__}
    Email: {__email__}
    
    Available Components:
      - Initialization functions: glorot, zeros, ones, normal, xavier_*, kaiming_*
      - Layers: GraphConvolution, SimpleAttLayer, RateLayer, GraphAttentionLayer
      - Models: Model, GCN, AGCNrec
    
    Usage:
      from models import AGCNrec, GraphConvolution
    """
    print(info)

# 在导入时打印信息（可选）
if __name__ != "__main__":
    import sys
    if "streamlit" not in sys.modules and "ipykernel" not in sys.modules:
        # 在非交互式环境中不打印
        pass