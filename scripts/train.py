"""
基础训练脚本
用于训练ACKRec模型
"""

import numpy as np
import torch
import random
import warnings
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_utils import load_data, construct_batch_data
from models.models import AGCNrec
from utils.metrics import print_metrics

# 设置随机种子
def set_seed(seed=123):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"随机种子设置为: {seed}")

# 超参数
def get_config():
    """获取训练配置"""
    return {
        'learning_rate': 0.001,
        'global_steps': 1000,
        'eval_frequency': 50,
        'seed': 123,
        'user_supports': ['uku'],
        'item_supports': ['kuk'],
        'data_dir': './data'
    }

def train_model(config):
    """
    训练模型
    
    Args:
        config: 训练配置字典
    """
    print("="*60)
    print("ACKRec 模型训练")
    print("="*60)
    
    # 设置随机种子
    set_seed(config['seed'])
    
    # 加载数据
    print("\n1. 加载数据...")
    try:
        rating, features_item, features_user, support_user, support_item, negative = load_data(
            user=config['user_supports'],
            item=config['item_supports'],
            data_dir=config['data_dir']
        )
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        print("尝试使用虚拟数据...")
        from utils.data_utils import create_dummy_data
        rating, features_item, features_user, support_user, support_item, negative = create_dummy_data()
    
    print(f"✅ 数据加载成功")
    print(f"   用户特征形状: {features_user.shape}")
    print(f"   物品特征形状: {features_item.shape}")
    print(f"   评分矩阵形状: {rating.shape}")
    print(f"   用户支持矩阵数量: {len(support_user)}")
    print(f"   物品支持矩阵数量: {len(support_item)}")
    
    user_dim = rating.shape[0]
    item_dim = rating.shape[1]
    
    # 创建placeholders字典
    placeholders = {
        'rating': rating,
        'features_user': features_user,
        'features_item': features_item,
        'negative': negative
    }
    
    # 创建模型
    print("\n2. 创建模型...")
    model = AGCNrec(
        placeholders=placeholders,
        input_dim_user=features_user.shape[1],
        input_dim_item=features_item.shape[1],
        user_dim=user_dim,
        item_dim=item_dim,
        learning_rate=config['learning_rate']
    )
    
    # 打印模型摘要
    model.summary()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"\n3. 使用设备: {device}")
    
    # 将数据移到设备上
    features_user = features_user.to(device)
    features_item = features_item.to(device)
    rating = rating.to(device)
    negative = negative.to(device)
    support_user = [sup.to(device) for sup in support_user]
    support_item = [sup.to(device) for sup in support_item]
    
    # 构造批处理数据
    batch_data = {
        'features_user': features_user,
        'features_item': features_item,
        'rating': rating,
        'supports_user': support_user,
        'supports_item': support_item,
        'negative': negative
    }
    
    # 训练循环
    print(f"\n4. 开始训练 ({config['global_steps']} 轮)...")
    print("-" * 80)
    print(f"{'轮次':^6} | {'损失':^10} | {'HR@10':^8} | {'NDCG@10':^8} | {'MRR':^8} | {'AUC':^8}")
    print("-" * 80)
    
    best_hr10 = 0.0
    best_epoch = 0
    
    for epoch in range(config['global_steps']):
        # 训练步骤
        loss_value = model.train_step(batch_data)
        
        # 评估
        if epoch % config['eval_frequency'] == 0 or epoch == config['global_steps'] - 1:
            # 评估指标
            metrics = model.evaluate(batch_data)
            
            # 保存最佳模型
            if metrics['hr@10'] > best_hr10:
                best_hr10 = metrics['hr@10']
                best_epoch = epoch
                # 保存最佳模型
                os.makedirs('./saved_models', exist_ok=True)
                model.save('./saved_models/best_model.pth')
            
            # 打印结果
            print(f"{epoch:6d} | {loss_value:10.4f} | {metrics['hr@10']:8.4f} | "
                  f"{metrics['ndcg@10']:8.4f} | {metrics['mrr']:8.4f} | {metrics['auc']:8.4f}")
    
    print("-" * 80)
    print(f"✅ 训练完成!")
    print(f"   最佳HR@10: {best_hr10:.4f} (第 {best_epoch} 轮)")
    
    # 最终评估
    print("\n5. 最终评估...")
    final_metrics = model.evaluate(batch_data)
    print_metrics(final_metrics, prefix="    ")
    
    # 保存最终模型
    model.save('./saved_models/final_model.pth')
    
    return model, final_metrics

def main():
    """主函数"""
    # 获取配置
    config = get_config()
    
    # 训练模型
    try:
        model, metrics = train_model(config)
        
        # 测试推荐功能
        print("\n6. 测试推荐功能...")
        try:
            recommendations = model.predict(user_id=0, top_k=5)
            print(f"   用户 0 的Top-5推荐:")
            for i, rec in enumerate(recommendations):
                print(f"     {i+1}. 物品 {rec['item_id']} (评分: {rec['score']:.4f})")
        except Exception as e:
            print(f"   ⚠️ 推荐测试失败: {e}")
        
        print("\n" + "="*60)
        print("🎉 训练流程完成!")
        print(f"   模型已保存到: ./saved_models/")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())