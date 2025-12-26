"""
ACKRec推荐系统 - Streamlit界面
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# 添加项目路径
sys.path.append('.')
sys.path.append('./models')
sys.path.append('./utils')

from models.models import AGCNrec
from utils.data_utils import load_data
from utils.metrics import hr, ndcg, mrr, auc

# 页面配置
st.set_page_config(
    page_title="ACKRec推荐系统",
    page_icon="🎓",
    layout="wide"
)

# 标题
st.title("🎓 ACKRec - 知识概念推荐系统")
st.markdown("基于注意力图卷积网络的MOOCs知识概念推荐系统")

# 侧边栏
st.sidebar.header("⚙️ 配置")

# 1. 模型选择
st.sidebar.subheader("模型设置")
use_gpu = st.sidebar.checkbox("使用GPU加速", value=torch.cuda.is_available())
model_path = st.sidebar.text_input("模型路径", value="./saved_models/best_model.pth")

# 2. 数据设置
st.sidebar.subheader("数据设置")
data_dir = st.sidebar.text_input("数据目录", value="./data")
user_supports = st.sidebar.multiselect(
    "用户支持矩阵",
    options=['uku', 'ucu', 'uvu', 'uctcu'],
    default=['uku']
)
item_supports = st.sidebar.multiselect(
    "物品支持矩阵",
    options=['kuk'],
    default=['kuk']
)

# 3. 评估设置
st.sidebar.subheader("评估设置")
k_values = st.sidebar.multiselect(
    "Top-K评估",
    options=[1, 5, 10, 20],
    default=[1, 5, 10, 20]
)

# 主界面标签页
tab1, tab2, tab3, tab4 = st.tabs(["🏠 概览", "🔍 数据探索", "🤖 模型推理", "📊 性能评估"])

with tab1:
    st.header("系统概览")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("支持的模型", "AGCNrec", "图神经网络")
    
    with col2:
        device = "GPU" if torch.cuda.is_available() else "CPU"
        st.metric("计算设备", device, "可用" if torch.cuda.is_available() else "仅CPU")
    
    with col3:
        # 检查数据文件
        data_files = os.listdir(data_dir) if os.path.exists(data_dir) else []
        st.metric("数据文件", len(data_files), "个文件")
    
    # 系统介绍
    st.markdown("""
    ### 系统特点
    
    - **异构图卷积**: 融合多种类型的实体和关系
    - **注意力机制**: 自适应融合不同元路径的信息
    - **端到端训练**: 从原始数据到推荐结果的全流程
    - **多维度评估**: HR@K, NDCG@K, MRR, AUC等指标
    
    ### 快速开始
    
    1. 准备你的数据（使用 `scripts/prepare_dataset.py`）
    2. 训练模型（使用 `scripts/train_new_dataset.py`）
    3. 在界面中加载模型进行评估
    4. 查看分析结果
    """)
    
    # 快速操作按钮
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🚀 快速测试", help="运行快速测试脚本"):
            with st.spinner("正在运行测试..."):
                import subprocess
                result = subprocess.run([sys.executable, "quick_start.py"], 
                                      capture_output=True, text=True)
                st.code(result.stdout)
    
    with col2:
        if st.button("🔧 故障排查", help="运行故障排查工具"):
            with st.spinner("正在检查系统..."):
                import subprocess
                result = subprocess.run([sys.executable, "troubleshooting.py", "2"], 
                                      capture_output=True, text=True)
                st.code(result.stdout)
    
    with col3:
        if st.button("📈 查看示例", help="查看示例结果"):
            st.info("查看 `scripts/` 目录中的示例脚本")

with tab2:
    st.header("数据探索")
    
    if st.button("加载数据"):
        try:
            with st.spinner("正在加载数据..."):
                rating, features_item, features_user, support_user, support_item, negative = load_data(
                    user=user_supports,
                    item=item_supports
                )
                
                # 显示数据信息
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("用户数", rating.shape[0])
                with col2:
                    st.metric("物品数", rating.shape[1])
                with col3:
                    st.metric("交互密度", 
                             f"{(rating != 0).sum().item() / (rating.shape[0] * rating.shape[1]) * 100:.2f}%")
                
                # 显示数据预览
                st.subheader("评分矩阵预览")
                fig, ax = plt.subplots(figsize=(10, 6))
                non_zero_mask = (rating != 0).cpu().numpy()
                ax.spy(non_zero_mask, markersize=0.5)
                ax.set_title("评分矩阵稀疏模式")
                ax.set_xlabel("物品索引")
                ax.set_ylabel("用户索引")
                st.pyplot(fig)
                
                # 显示负样本信息
                st.subheader("负样本信息")
                st.write(f"负样本形状: {negative.shape}")
                st.write(f"每个用户的负样本数: {negative.shape[1]}")
                
        except Exception as e:
            st.error(f"数据加载失败: {e}")

with tab3:
    st.header("模型推理")
    
    if st.button("加载模型"):
        try:
            # 加载数据
            with st.spinner("正在准备数据..."):
                rating, features_item, features_user, support_user, support_item, negative = load_data(
                    user=user_supports,
                    item=item_supports
                )
                
                # 创建模型
                placeholders = {
                    'rating': rating,
                    'features_user': features_user,
                    'features_item': features_item,
                    'negative': negative
                }
                
                model = AGCNrec(
                    placeholders=placeholders,
                    input_dim_user=features_user.shape[1],
                    input_dim_item=features_item.shape[1],
                    user_dim=rating.shape[0],
                    item_dim=rating.shape[1],
                    learning_rate=0.001
                )
                
                # 加载权重
                if os.path.exists(model_path):
                    model.load(model_path)
                    st.success("✅ 模型加载成功！")
                else:
                    st.warning("⚠️ 模型文件不存在，使用随机初始化")
            
            # 用户选择
            st.subheader("选择用户进行推荐")
            user_id = st.number_input("用户ID", min_value=0, max_value=rating.shape[0]-1, value=0)
            
            if st.button("生成推荐"):
                with st.spinner("正在生成推荐..."):
                    # 前向传播
                    with torch.no_grad():
                        model.eval()
                        rate_matrix = model.forward(
                            features_user, features_item,
                            support_user, support_item
                        )
                        
                        # 获取用户的评分
                        user_ratings = rate_matrix[user_id, :]
                        top_k = 10
                        top_indices = torch.argsort(user_ratings, descending=True)[:top_k]
                        
                        # 显示推荐结果
                        st.subheader(f"用户 {user_id} 的Top-{top_k}推荐")
                        
                        results = []
                        for i, item_idx in enumerate(top_indices):
                            score = user_ratings[item_idx].item()
                            results.append({
                                "排名": i+1,
                                "物品ID": item_idx.item(),
                                "预测评分": f"{score:.4f}"
                            })
                        
                        df_results = pd.DataFrame(results)
                        st.table(df_results)
                        
                        # 可视化
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.bar(range(top_k), user_ratings[top_indices].cpu().numpy())
                        ax.set_xlabel("推荐排名")
                        ax.set_ylabel("预测评分")
                        ax.set_title(f"用户 {user_id} 的Top-{top_k}推荐评分")
                        st.pyplot(fig)
                        
        except Exception as e:
            st.error(f"模型推理失败: {e}")

with tab4:
    st.header("性能评估")
    
    if st.button("运行评估"):
        try:
            with st.spinner("正在评估模型性能..."):
                # 加载数据和模型
                rating, features_item, features_user, support_user, support_item, negative = load_data(
                    user=user_supports,
                    item=item_supports
                )
                
                placeholders = {
                    'rating': rating,
                    'features_user': features_user,
                    'features_item': features_item,
                    'negative': negative
                }
                
                model = AGCNrec(
                    placeholders=placeholders,
                    input_dim_user=features_user.shape[1],
                    input_dim_item=features_item.shape[1],
                    user_dim=rating.shape[0],
                    item_dim=rating.shape[1],
                    learning_rate=0.001
                )
                
                if os.path.exists(model_path):
                    model.load(model_path)
                
                # 评估
                device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
                model = model.to(device)
                
                features_user = features_user.to(device)
                features_item = features_item.to(device)
                rating = rating.to(device)
                negative = negative.to(device)
                support_user = [sup.to(device) for sup in support_user]
                support_item = [sup.to(device) for sup in support_item]
                
                batch_data = {
                    'features_user': features_user,
                    'features_item': features_item,
                    'rating': rating,
                    'supports_user': support_user,
                    'supports_item': support_item,
                    'negative': negative
                }
                
                with torch.no_grad():
                    metrics = model.evaluate(batch_data)
                
                # 显示结果
                st.subheader("评估结果")
                
                # 创建指标表格
                metric_data = []
                for k in k_values:
                    if f'hr@{k}' in metrics:
                        metric_data.append({
                            "指标": f"HR@{k}",
                            "值": f"{metrics[f'hr@{k}']:.4f}"
                        })
                    if f'ndcg@{k}' in metrics:
                        metric_data.append({
                            "指标": f"NDCG@{k}",
                            "值": f"{metrics[f'ndcg@{k}']:.4f}"
                        })
                
                for metric in ['mrr', 'auc']:
                    if metric in metrics:
                        metric_data.append({
                            "指标": metric.upper(),
                            "值": f"{metrics[metric]:.4f}"
                        })
                
                df_metrics = pd.DataFrame(metric_data)
                st.table(df_metrics)
                
                # 可视化
                st.subheader("指标可视化")
                
                # HR指标柱状图
                hr_values = {k: metrics.get(f'hr@{k}', 0) for k in k_values}
                fig1, ax1 = plt.subplots(figsize=(8, 4))
                ax1.bar(hr_values.keys(), hr_values.values())
                ax1.set_xlabel("K值")
                ax1.set_ylabel("Hit Rate")
                ax1.set_title("HR@K 指标")
                st.pyplot(fig1)
                
                # NDCG指标柱状图
                ndcg_values = {k: metrics.get(f'ndcg@{k}', 0) for k in k_values}
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                ax2.bar(ndcg_values.keys(), ndcg_values.values())
                ax2.set_xlabel("K值")
                ax2.set_ylabel("NDCG")
                ax2.set_title("NDCG@K 指标")
                st.pyplot(fig2)
                
        except Exception as e:
            st.error(f"评估失败: {e}")

# 页脚
st.sidebar.markdown("---")
st.sidebar.markdown("""
**关于ACKRec**
- 论文: [arXiv:2006.13257](https://arxiv.org/abs/2006.13257)
- GitHub: [AI4Edu-Group/ACKRec](https://github.com/AI4Edu-Group/ACKRec)
- 版本: 1.0.0
""")