"""
新数据集训练脚本
使用自动配置训练ACKRec模型
"""

import numpy as np
import torch
import random
import time
import os
import sys
import pickle as pkl
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 导入必要的模块
from models.models import AGCNrec
from utils.data_utils import load_new_dataset
from config import ExperimentConfig, DatasetConfig, TrainingConfig
from utils.metrics import print_metrics

class NewDatasetTrainer:
    """新数据集的训练器"""
    
    def __init__(self, config):
        """
        初始化训练器
        
        Args:
            config: 完整的配置字典
        """
        self.config = config
        self.model = None
        self.batch_data = None
        
        # 设置随机种子
        self.set_random_seed()
        
        # 创建输出目录
        self.create_output_dir()
        
    def set_random_seed(self):
        """设置随机种子"""
        seed = self.config['training']['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        print(f"✅ 随机种子设置为: {seed}")
    
    def create_output_dir(self):
        """创建输出目录"""
        output_dir = self.config['training']['output_dir']
        
        # 添加时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(output_dir, f'run_{timestamp}')
        
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"✅ 输出目录: {self.output_dir}")
        
        # 保存配置
        config_file = os.path.join(self.output_dir, 'config.pkl')
        with open(config_file, 'wb') as f:
            pkl.dump(self.config, f)
        
        # 保存文本格式的配置
        config_text = os.path.join(self.output_dir, 'config.txt')
        with open(config_text, 'w') as f:
            f.write("="*60 + "\n")
            f.write("ACKRec 实验配置\n")
            f.write("="*60 + "\n\n")
            
            f.write("📊 数据集统计:\n")
            stats = self.config['stats']
            for key, value in stats.items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\n🤖 模型配置:\n")
            model_config = self.config['model']
            for key, value in model_config.items():
                if key != 'description':
                    f.write(f"  {key}: {value}\n")
            f.write(f"  描述: {model_config['description']}\n")
            
            f.write("\n⚙️ 训练配置:\n")
            train_config = self.config['training']
            for key, value in train_config.items():
                f.write(f"  {key}: {value}\n")
        
        print(f"✅ 配置已保存到: {config_text}")
    
    def load_data(self):
        """加载数据"""
        print("\n" + "="*50)
        print("📂 加载数据")
        print("="*50)
        
        dataset_config = self.config['dataset']
        
        try:
            rating, features_item, features_user, support_user, support_item, negative = load_new_dataset(
                data_dir=dataset_config['data_dir']
            )
            
            print(f"✅ 数据加载完成:")
            print(f"   评分矩阵形状: {rating.shape}")
            print(f"   用户特征形状: {features_user.shape}")
            print(f"   物品特征形状: {features_item.shape}")
            print(f"   负样本形状: {negative.shape}")
            print(f"   用户支持矩阵数量: {len(support_user)}")
            print(f"   物品支持矩阵数量: {len(support_item)}")
            
            # 保存数据信息
            self.dataset_info = {
                'user_dim': rating.shape[0],
                'item_dim': rating.shape[1],
                'input_dim_user': features_user.shape[1],
                'input_dim_item': features_item.shape[1]
            }
            
            return rating, features_item, features_user, support_user, support_item, negative
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            print("尝试使用虚拟数据...")
            from utils.data_utils import create_dummy_data
            return create_dummy_data(
                num_users=self.config['stats'].get('num_users', 100),
                num_items=self.config['stats'].get('num_items', 50)
            )
    
    def create_model(self):
        """创建模型"""
        print("\n" + "="*50)
        print("🤖 创建模型")
        print("="*50)
        
        # 从配置获取参数
        model_config = self.config['model']
        
        # 创建placeholders
        placeholders = {
            'rating': self.rating,
            'features_user': self.features_user,
            'features_item': self.features_item,
            'negative': self.negative
        }
        
        # 创建模型
        print(f"使用模型配置:")
        print(f"  隐藏层维度: {model_config['hidden_dims']}")
        print(f"  输出维度: {model_config['output_dim']}")
        print(f"  潜在维度: {model_config['latent_dim']}")
        print(f"  注意力大小: {model_config['attention_size']}")
        print(f"  Dropout率: {model_config['dropout_rate']}")
        print(f"  学习率: {model_config['learning_rate']}")
        print(f"  训练轮数: {model_config['epochs']}")
        
        self.model = AGCNrec(
            placeholders=placeholders,
            input_dim_user=self.dataset_info['input_dim_user'],
            input_dim_item=self.dataset_info['input_dim_item'],
            user_dim=self.dataset_info['user_dim'],
            item_dim=self.dataset_info['item_dim'],
            learning_rate=model_config['learning_rate']
        )
        
        # 设置设备
        device = self.config['training']['device']
        self.model = self.model.to(device)
        print(f"✅ 模型已创建并移动到: {device}")
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"✅ 模型参数总数: {total_params:,}")
        
        # 保存模型结构
        model_structure_file = os.path.join(self.output_dir, 'model_structure.txt')
        with open(model_structure_file, 'w') as f:
            f.write(str(self.model))
        print(f"✅ 模型结构已保存到: {model_structure_file}")
    
    def prepare_batch_data(self):
        """准备批处理数据"""
        device = self.config['training']['device']
        
        self.batch_data = {
            'features_user': self.features_user.to(device),
            'features_item': self.features_item.to(device),
            'rating': self.rating.to(device),
            'supports_user': [sup.to(device) for sup in self.support_user],
            'supports_item': [sup.to(device) for sup in self.support_item],
            'negative': self.negative.to(device)
        }
        
        print(f"✅ 批处理数据已准备")
    
    def train(self):
        """训练模型"""
        print("\n" + "="*50)
        print("🚀 开始训练")
        print("="*50)
        
        # 获取训练参数
        epochs = self.config['model']['epochs']
        eval_frequency = self.config['training']['eval_frequency']
        output_dir = self.output_dir
        
        # 创建日志文件
        log_file = os.path.join(output_dir, 'training_log.csv')
        with open(log_file, 'w') as f:
            f.write('epoch,loss,hr1,hr5,hr10,hr20,ndcg5,ndcg10,ndcg20,mrr,auc,time\n')
        
        print(f"训练参数:")
        print(f"  总轮数: {epochs}")
        print(f"  评估频率: 每 {eval_frequency} 轮")
        print(f"  日志文件: {log_file}")
        print("-" * 80)
        print(f"{'轮次':^6} | {'损失':^10} | {'HR@10':^8} | {'NDCG@10':^8} | {'MRR':^8} | {'AUC':^8} | {'时间':^6}")
        print("-" * 80)
        
        # 训练循环
        best_hr10 = 0.0
        best_epoch = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # 训练步骤
            loss = self.model.train_step(self.batch_data)
            epoch_time = time.time() - epoch_start
            
            # 评估
            if epoch % eval_frequency == 0 or epoch == epochs - 1:
                metrics = self.model.evaluate(self.batch_data)
                
                # 保存最佳模型
                if metrics['hr@10'] > best_hr10:
                    best_hr10 = metrics['hr@10']
                    best_epoch = epoch
                    model_path = os.path.join(output_dir, f'best_model_epoch{epoch}.pth')
                    self.model.save(model_path)
                
                # 打印进度
                print(f"{epoch:6d} | {loss:10.4f} | {metrics['hr@10']:8.4f} | "
                      f"{metrics['ndcg@10']:8.4f} | {metrics['mrr']:8.4f} | "
                      f"{metrics['auc']:8.4f} | {epoch_time:6.1f}s")
                
                # 记录日志
                with open(log_file, 'a') as f:
                    f.write(f"{epoch},{loss:.4f},"
                           f"{metrics['hr@1']:.4f},{metrics['hr@5']:.4f},"
                           f"{metrics['hr@10']:.4f},{metrics['hr@20']:.4f},"
                           f"{metrics['ndcg@5']:.4f},{metrics['ndcg@10']:.4f},"
                           f"{metrics['ndcg@20']:.4f},{metrics['mrr']:.4f},"
                           f"{metrics['auc']:.4f},{epoch_time:.1f}\n")
        
        # 训练完成
        total_time = time.time() - start_time
        print("-" * 80)
        print(f"✅ 训练完成! 总时间: {total_time:.1f}秒")
        print(f"🎯 最佳HR@10: {best_hr10:.4f} (epoch {best_epoch})")
        
        # 保存最终模型
        final_model_path = os.path.join(output_dir, 'final_model.pth')
        self.model.save(final_model_path)
        print(f"✅ 最终模型已保存到: {final_model_path}")
        
        # 加载最佳模型进行最终评估
        print("\n📊 加载最佳模型进行最终评估...")
        best_model_path = os.path.join(output_dir, f'best_model_epoch{best_epoch}.pth')
        self.model.load(best_model_path)
        final_metrics = self.model.evaluate(self.batch_data)
        
        print("\n📈 最终评估结果:")
        print_metrics(final_metrics, prefix="  ")
        
        # 保存最终结果
        self.save_final_results(best_epoch, best_hr10, total_time, final_metrics)
        
        return final_metrics
    
    def save_final_results(self, best_epoch, best_hr10, total_time, metrics):
        """保存最终结果"""
        result_file = os.path.join(self.output_dir, 'final_results.txt')
        
        with open(result_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("🎉 ACKRec 训练最终结果\n")
            f.write("="*60 + "\n\n")
            
            f.write("📋 训练摘要:\n")
            f.write(f"  最佳轮次: {best_epoch}\n")
            f.write(f"  最佳HR@10: {best_hr10:.4f}\n")
            f.write(f"  总训练时间: {total_time:.1f}秒\n\n")
            
            f.write("📊 最终评估指标:\n")
            f.write("-" * 40 + "\n")
            
            # 分组显示指标
            hr_metrics = {k: v for k, v in metrics.items() if k.startswith('hr@')}
            ndcg_metrics = {k: v for k, v in metrics.items() if k.startswith('ndcg@')}
            other_metrics = {k: v for k, v in metrics.items() if not k.startswith('hr@') and not k.startswith('ndcg@')}
            
            if hr_metrics:
                f.write("Hit Rate (HR):\n")
                for k in sorted(hr_metrics.keys(), key=lambda x: int(x.split('@')[1])):
                    f.write(f"  {k:8}: {hr_metrics[k]:.4f}\n")
                f.write("\n")
            
            if ndcg_metrics:
                f.write("Normalized DCG:\n")
                for k in sorted(ndcg_metrics.keys(), key=lambda x: int(x.split('@')[1])):
                    f.write(f"  {k:8}: {ndcg_metrics[k]:.4f}\n")
                f.write("\n")
            
            if other_metrics:
                f.write("其他指标:\n")
                for k, v in other_metrics.items():
                    f.write(f"  {k:8}: {v:.4f}\n")
            
            f.write("-" * 40 + "\n\n")
            
            f.write("⚙️ 配置摘要:\n")
            f.write("-" * 40 + "\n")
            
            f.write("数据集:\n")
            stats = self.config['stats']
            for key, value in stats.items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\n模型架构:\n")
            model_config = self.config['model']
            for key, value in model_config.items():
                if key != 'description':
                    f.write(f"  {key}: {value}\n")
            f.write(f"  描述: {model_config['description']}\n")
        
        print(f"✅ 结果已保存到: {result_file}")
    
    def run(self):
        """运行完整的训练流程"""
        print("\n" + "="*60)
        print("🚀 ACKRec 新数据集训练流程")
        print("="*60)
        
        try:
            # 1. 加载数据
            print("\n1. 加载数据...")
            (self.rating, self.features_item, self.features_user, 
             self.support_user, self.support_item, self.negative) = self.load_data()
            
            # 2. 创建模型
            print("\n2. 创建模型...")
            self.create_model()
            
            # 3. 准备数据
            print("\n3. 准备数据...")
            self.prepare_batch_data()
            
            # 4. 训练模型
            print("\n4. 训练模型...")
            final_metrics = self.train()
            
            # 5. 显示结果
            print("\n" + "="*60)
            print("🎉 训练完成!")
            print("="*60)
            print(f"📁 输出目录: {self.output_dir}")
            print(f"📊 最终指标:")
            print_metrics(final_metrics, prefix="  ")
            
            return final_metrics
            
        except Exception as e:
            print(f"\n❌ 训练流程失败: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """主函数"""
    print("ACKRec 新数据集训练器")
    print("="*60)
    
    # 第一步：检查数据目录
    data_dir = './processed_data'
    if not os.path.exists(data_dir):
        print(f"⚠️ 处理后的数据目录不存在: {data_dir}")
        print("请先运行数据准备脚本:")
        print("  python scripts/create_test_data.py  # 创建测试数据")
        print("  或")
        print("  python scripts/prepare_dataset.py   # 准备您的数据")
        return 1
    
    # 第二步：尝试加载数据获取统计信息
    print("正在检查数据集...")
    try:
        rating, _, _, _, _, _ = load_new_dataset(data_dir=data_dir)
        dataset_stats = {
            'num_users': rating.shape[0],
            'num_items': rating.shape[1],
            'rating_shape': rating.shape,
            'density': (rating != 0).sum().item() / (rating.shape[0] * rating.shape[1])
        }
        
        print(f"✅ 数据集统计:")
        print(f"   用户数量: {dataset_stats['num_users']}")
        print(f"   物品数量: {dataset_stats['num_items']}")
        print(f"   评分矩阵: {dataset_stats['rating_shape']}")
        print(f"   交互密度: {dataset_stats['density']:.4f}")
        
    except Exception as e:
        print(f"❌ 无法加载数据集: {e}")
        print("使用默认数据集统计...")
        dataset_stats = {
            'num_users': 100,
            'num_items': 50,
            'rating_shape': (100, 50),
            'density': 0.1
        }
    
    # 第三步：根据数据大小自动配置
    print("\n根据数据集大小自动配置模型...")
    full_config = ExperimentConfig.setup_experiment(dataset_stats)
    
    # 显示配置
    model_config = full_config['model']
    print(f"✅ 自动配置完成:")
    print(f"   模型类型: {model_config['description']}")
    print(f"   隐藏层: {model_config['hidden_dims']}")
    print(f"   训练轮数: {model_config['epochs']}")
    print(f"   学习率: {model_config['learning_rate']}")
    
    # 第四步：创建训练器并运行
    print("\n" + "="*60)
    trainer = NewDatasetTrainer(full_config)
    final_metrics = trainer.run()
    
    if final_metrics:
        print("\n" + "="*60)
        print("🎉 训练成功完成!")
        print("="*60)
        print("\n下一步:")
        print("1. 查看训练结果: analyze_results.py")
        print("2. 启动Web界面: streamlit run app.py")
        print("3. 使用模型进行推荐")
        return 0
    else:
        print("\n❌ 训练失败")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())