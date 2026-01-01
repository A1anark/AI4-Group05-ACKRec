"""
创建测试数据脚本
用于生成测试数据集
"""

import pandas as pd
import numpy as np
import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def create_interactions_csv():
    """创建测试用的interactions.csv文件"""
    
    # 创建数据目录
    raw_dir = './raw_data'
    os.makedirs(raw_dir, exist_ok=True)
    
    # 模拟数据参数
    num_users = 100      # 100个用户
    num_items = 50       # 50个物品
    num_interactions = 500  # 500条交互记录
    
    print("="*60)
    print("ACKRec 测试数据集生成器")
    print("="*60)
    print(f"生成测试数据集:")
    print(f"- 用户数: {num_users}")
    print(f"- 物品数: {num_items}")
    print(f"- 交互数: {num_interactions}")
    
    # 生成随机交互数据
    np.random.seed(42)  # 可重复的随机数
    
    user_ids = np.random.randint(1, num_users + 1, num_interactions)
    item_ids = np.random.randint(1, num_items + 1, num_interactions)
    ratings = np.random.randint(1, 6, num_interactions)  # 1-5分
    timestamps = np.random.randint(1609459200, 1640995200, num_interactions)  # 2021-2022
    
    # 创建DataFrame
    df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings,
        'timestamp': timestamps
    })
    
    # 保存为CSV
    csv_path = os.path.join(raw_dir, 'interactions.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"\n✅ 交互数据已保存到: {csv_path}")
    
    # 统计信息
    print(f"\n📊 数据统计:")
    print(f"唯一用户数: {df['user_id'].nunique()}")
    print(f"唯一物品数: {df['item_id'].nunique()}")
    print(f"平均每个用户交互数: {len(df) / df['user_id'].nunique():.2f}")
    print(f"平均每个物品被交互数: {len(df) / df['item_id'].nunique():.2f}")
    
    return df

def create_user_features(num_users=100):
    """创建用户特征文件"""
    print(f"\n👤 创建用户特征 ({num_users} 个用户)...")
    
    users_data = []
    
    for user_id in range(1, num_users + 1):
        # 模拟一些特征
        age = np.random.randint(18, 60)
        gender = np.random.choice(['M', 'F'])
        education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'])
        
        # 随机生成一些偏好特征
        pref_features = np.random.randn(5)  # 5个偏好特征
        
        users_data.append({
            'user_id': user_id,
            'age': age,
            'gender': gender,
            'education': education,
            'pref_feature1': float(pref_features[0]),
            'pref_feature2': float(pref_features[1]),
            'pref_feature3': float(pref_features[2]),
            'pref_feature4': float(pref_features[3]),
            'pref_feature5': float(pref_features[4])
        })
    
    users_df = pd.DataFrame(users_data)
    users_path = os.path.join('./raw_data', 'users.csv')
    users_df.to_csv(users_path, index=False)
    
    print(f"✅ 用户特征已保存到: {users_path}")
    print(f"   特征维度: {len(users_df.columns) - 1} 个特征")
    
    return users_df

def create_item_features(num_items=50):
    """创建物品特征文件"""
    print(f"\n📚 创建物品特征 ({num_items} 个物品)...")
    
    items_data = []
    
    # 模拟一些物品类别
    categories = ['Mathematics', 'Physics', 'Chemistry', 'Biology', 'Computer Science',
                  'Literature', 'History', 'Art', 'Music', 'Sports']
    
    for item_id in range(1, num_items + 1):
        # 模拟一些特征
        category = np.random.choice(categories)
        difficulty = np.random.uniform(0, 1)
        duration = np.random.randint(30, 180)  # 30-180分钟
        popularity = np.random.random()
        
        # 随机生成一些内容特征
        content_features = np.random.randn(10)  # 10个内容特征
        
        items_data.append({
            'item_id': item_id,
            'category': category,
            'difficulty': round(difficulty, 3),
            'duration': duration,
            'popularity': round(popularity, 3),
            'content_feature1': float(content_features[0]),
            'content_feature2': float(content_features[1]),
            'content_feature3': float(content_features[2]),
            'content_feature4': float(content_features[3]),
            'content_feature5': float(content_features[4]),
            'content_feature6': float(content_features[5]),
            'content_feature7': float(content_features[6]),
            'content_feature8': float(content_features[7]),
            'content_feature9': float(content_features[8]),
            'content_feature10': float(content_features[9])
        })
    
    items_df = pd.DataFrame(items_data)
    items_path = os.path.join('./raw_data', 'items.csv')
    items_df.to_csv(items_path, index=False)
    
    print(f"✅ 物品特征已保存到: {items_path}")
    print(f"   特征维度: {len(items_df.columns) - 1} 个特征")
    
    # 类别分布
    print(f"   类别分布:")
    category_counts = items_df['category'].value_counts()
    for category, count in category_counts.items():
        print(f"     - {category}: {count} 个物品")
    
    return items_df

def create_dataset_info():
    """创建数据集信息文件"""
    print(f"\n📋 创建数据集信息...")
    
    info = {
        'dataset_name': 'ACKRec_Test_Dataset',
        'description': 'A simulated test dataset for ACKRec recommendation system',
        'created_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_users': 100,
        'num_items': 50,
        'num_interactions': 500,
        'rating_scale': '1-5',
        'has_user_features': True,
        'has_item_features': True,
        'purpose': 'Testing and demonstration of ACKRec system'
    }
    
    info_df = pd.DataFrame([info])
    info_path = os.path.join('./raw_data', 'dataset_info.csv')
    info_df.to_csv(info_path, index=False)
    
    print(f"✅ 数据集信息已保存到: {info_path}")
    
    return info_df

def create_sample_data_for_data_dir():
    """为data目录创建样本数据"""
    print(f"\n💾 为data目录创建样本数据...")
    
    from utils.data_utils import save_sample_data
    
    save_sample_data('./data')
    
    print(f"✅ 样本数据已保存到 ./data/ 目录")

def main():
    """主函数"""
    
    print("ACKRec 测试数据集生成器")
    print("="*60)
    
    # 创建原始数据
    print("\n1. 创建原始数据...")
    interactions_df = create_interactions_csv()
    create_user_features(100)
    create_item_features(50)
    create_dataset_info()
    
    # 处理数据
    print("\n2. 处理数据...")
    try:
        from scripts.prepare_dataset import NewDatasetPreparer
        preparer = NewDatasetPreparer(
            raw_data_dir='./raw_data',
            output_dir='./processed_data'
        )
        stats = preparer.load_and_process()
        
        if stats:
            print(f"\n✅ 数据处理成功!")
            print(f"   用户数: {stats['num_users']}")
            print(f"   物品数: {stats['num_items']}")
            print(f"   评分矩阵: {stats['rating_shape']}")
    except Exception as e:
        print(f"❌ 数据处理失败: {e}")
    
    # 为data目录创建样本数据
    print("\n3. 创建样本数据...")
    create_sample_data_for_data_dir()
    
    print("\n" + "="*60)
    print("🎉 测试数据集创建完成!")
    print("="*60)
    print("\n文件结构:")
    print("raw_data/")
    print("├── interactions.csv    # 交互数据")
    print("├── users.csv          # 用户特征")
    print("├── items.csv          # 物品特征")
    print("└── dataset_info.csv   # 数据集信息")
    print("\nprocessed_data/")
    print("├── rate_matrix.p      # 评分矩阵")
    print("├── negative.p         # 负样本")
    print("├── UC.p              # 用户特征")
    print("├── concept_feature_bow.p  # 物品特征")
    print("└── ...               # 其他文件")
    print("\ndata/")
    print("├── sample_*.p        # 样本数据文件")
    print("\n现在你可以运行:")
    print("1. python scripts/train.py           # 训练模型")
    print("2. python scripts/train_new_dataset.py  # 使用新数据集训练")
    print("3. streamlit run app.py            # 启动Web界面")
    print("="*60)

if __name__ == "__main__":
    main()