"""
ACKRec推荐系统 - Streamlit界面
主应用程序文件
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import warnings

# 设置页面配置
st.set_page_config(
    page_title="ACKRec - 知识概念推荐系统",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 添加项目路径
sys.path.append('.')
sys.path.append('./models')
sys.path.append('./utils')

# 忽略警告
warnings.filterwarnings('ignore')

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .warning-message {
        color: #ffc107;
        font-weight: bold;
    }
    .error-message {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# 页面标题
st.markdown('<h1 class="main-header">🎓 ACKRec - 知识概念推荐系统</h1>', unsafe_allow_html=True)
st.markdown("### 基于注意力图卷积网络的MOOCs知识概念推荐系统")

# 侧边栏配置
st.sidebar.markdown('<h2 class="sub-header">⚙️ 配置</h2>', unsafe_allow_html=True)

# 1. 模型选择
st.sidebar.markdown("#### 模型设置")
use_gpu = st.sidebar.checkbox("使用GPU加速", value=torch.cuda.is_available())
model_path = st.sidebar.text_input("模型路径", value="./saved_models/best_model.pth")

# 2. 数据设置
st.sidebar.markdown("#### 数据设置")
data_dir = st.sidebar.selectbox(
    "数据目录",
    options=['./data', './processed_data', './test_data'],
    index=0
)

# 3. 评估设置
st.sidebar.markdown("#### 评估设置")
k_values = st.sidebar.multiselect(
    "Top-K评估",
    options=[1, 5, 10, 20],
    default=[1, 5, 10, 20]
)

# 4. 系统信息
st.sidebar.markdown("---")
st.sidebar.markdown("#### 系统信息")
st.sidebar.info(f"PyTorch版本: {torch.__version__}")
st.sidebar.info(f"CUDA可用: {'是' if torch.cuda.is_available() else '否'}")
if torch.cuda.is_available():
    st.sidebar.info(f"GPU设备: {torch.cuda.get_device_name(0)}")

# 主界面标签页
tab1, tab2, tab3, tab4 = st.tabs(["🏠 概览", "🔍 数据探索", "🤖 模型推理", "📊 性能评估"])

with tab1:
    st.markdown('<h2 class="sub-header">系统概览</h2>', unsafe_allow_html=True)
    
    # 系统介绍
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 系统特点
        
        - **异构图卷积**: 融合多种类型的实体和关系
        - **注意力机制**: 自适应融合不同元路径的信息
        - **端到端训练**: 从原始数据到推荐结果的全流程
        - **多维度评估**: HR@K, NDCG@K, MRR, AUC等指标
        - **可视化界面**: 交互式的模型管理和结果展示
        
        ### 核心组件
        
        1. **GraphConvolution** - 图卷积层
        2. **SimpleAttLayer** - 注意力层
        3. **RateLayer** - 评分层
        4. **GCN** - 图卷积网络
        5. **AGCNrec** - 完整的推荐模型
        """)
    
    with col2:
        # 系统状态卡片
        st.markdown("### 系统状态")
        
        col2_1, col2_2, col2_3 = st.columns(3)
        
        with col2_1:
            # 检查模型文件
            model_exists = os.path.exists(model_path)
            if model_exists:
                st.success("✅ 模型就绪")
            else:
                st.warning("⚠️ 模型未找到")
        
        with col2_2:
            # 检查数据目录
            data_exists = os.path.exists(data_dir)
            if data_exists:
                st.success("✅ 数据就绪")
            else:
                st.warning("⚠️ 数据未找到")
        
        with col2_3:
            # 设备信息
            device = "GPU" if use_gpu and torch.cuda.is_available() else "CPU"
            st.info(f"📱 {device}")
        
        # 快速操作
        st.markdown("### 快速操作")
        
        col2_4, col2_5 = st.columns(2)
        
        with col2_4:
            if st.button("🚀 运行快速测试", use_container_width=True):
                with st.spinner("正在运行测试..."):
                    try:
                        import subprocess
                        result = subprocess.run(
                            [sys.executable, "quick_start.py"], 
                            capture_output=True, 
                            text=True,
                            timeout=30
                        )
                        
                        if result.returncode == 0:
                            st.success("✅ 测试通过!")
                            with st.expander("查看测试输出"):
                                st.code(result.stdout)
                        else:
                            st.error("❌ 测试失败")
                            with st.expander("查看错误信息"):
                                st.code(result.stderr)
                    except Exception as e:
                        st.error(f"测试执行失败: {e}")
        
        with col2_5:
            if st.button("🔧 故障排查", use_container_width=True):
                with st.spinner("正在检查系统..."):
                    try:
                        import subprocess
                        result = subprocess.run(
                            [sys.executable, "troubleshooting.py", "2"], 
                            capture_output=True, 
                            text=True,
                            timeout=30
                        )
                        
                        with st.expander("查看检查结果"):
                            st.code(result.stdout)
                    except Exception as e:
                        st.error(f"检查失败: {e}")
    
    # 使用指南
    st.markdown("### 使用指南")
    
    with st.expander("📖 快速开始指南"):
        st.markdown("""
        1. **准备数据**
           - 将您的数据放入 `raw_data/` 目录
           - 运行 `python scripts/prepare_dataset.py`
        
        2. **训练模型**
           - 运行 `python scripts/train_new_dataset.py`
           - 或使用Web界面的训练功能
        
        3. **评估模型**
           - 在"性能评估"标签页中运行评估
           - 查看各项指标结果
        
        4. **使用推荐**
           - 在"模型推理"标签页中选择用户
           - 生成个性化推荐
        """)
    
    with st.expander("📁 项目结构"):
        st.code("""
        ACKRec/
        ├── data/                    # 数据目录
        ├── models/                  # 模型定义
        ├── utils/                   # 工具函数
        ├── scripts/                 # 训练脚本
        ├── saved_models/            # 训练好的模型
        ├── app.py                   # Web界面
        ├── requirements.txt         # 依赖包
        ├── config.py                # 配置文件
        └── README.md                # 项目说明
        """)
    
    with st.expander("📊 评估指标说明"):
        st.markdown("""
        - **HR@K (Hit Rate)**: 命中率，前K个推荐中是否包含用户感兴趣的项目
        - **NDCG@K**: 归一化折损累计增益，考虑推荐排名的质量评估
        - **MRR (Mean Reciprocal Rank)**: 平均倒数排名，第一个相关项目排名的倒数平均值
        - **AUC (Area Under Curve)**: 曲线下面积，衡量模型整体排序能力的指标
        """)

with tab2:
    st.markdown('<h2 class="sub-header">数据探索</h2>', unsafe_allow_html=True)
    
    # 数据加载选项
    col1, col2 = st.columns([2, 1])
    
    with col1:
        data_source = st.radio(
            "数据来源",
            options=["处理后的数据", "原始数据", "示例数据"],
            index=0
        )
    
    with col2:
        load_data_btn = st.button("📊 加载数据", type="primary")
    
    if load_data_btn:
        try:
            with st.spinner("正在加载数据..."):
                # 导入数据工具
                from utils.data_utils import load_data
                
                # 根据选择加载数据
                if data_source == "处理后的数据":
                    data_path = './processed_data'
                elif data_source == "原始数据":
                    data_path = './raw_data'
                    st.warning("原始数据需要先处理，将尝试加载处理后的数据")
                    data_path = './processed_data'
                else:
                    data_path = './data'
                
                # 加载数据
                rating, features_item, features_user, support_user, support_item, negative = load_data(
                    user=['uku'],
                    item=['kuk'],
                    data_dir=data_path
                )
                
                st.success(f"✅ 数据加载成功 ({data_path})")
                
                # 显示数据信息
                col_info1, col_info2, col_info3 = st.columns(3)
                
                with col_info1:
                    st.metric("用户数", rating.shape[0])
                
                with col_info2:
                    st.metric("物品数", rating.shape[1])
                
                with col_info3:
                    density = (rating != 0).sum().item() / (rating.shape[0] * rating.shape[1]) * 100
                    st.metric("交互密度", f"{density:.2f}%")
                
                # 数据可视化
                st.markdown("### 数据可视化")
                
                # 评分矩阵热力图
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                non_zero_mask = (rating != 0).cpu().numpy()
                ax1.spy(non_zero_mask, markersize=0.5)
                ax1.set_title("评分矩阵稀疏模式")
                ax1.set_xlabel("物品索引")
                ax1.set_ylabel("用户索引")
                st.pyplot(fig1)
                
                # 用户交互分布
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                user_interactions = (rating != 0).sum(dim=1).cpu().numpy()
                ax2.hist(user_interactions, bins=20, alpha=0.7, color='skyblue')
                ax2.set_title("用户交互次数分布")
                ax2.set_xlabel("交互次数")
                ax2.set_ylabel("用户数量")
                st.pyplot(fig2)
                
                # 物品流行度分布
                fig3, ax3 = plt.subplots(figsize=(10, 4))
                item_popularity = (rating != 0).sum(dim=0).cpu().numpy()
                ax3.hist(item_popularity, bins=20, alpha=0.7, color='lightgreen')
                ax3.set_title("物品被交互次数分布")
                ax3.set_xlabel("被交互次数")
                ax3.set_ylabel("物品数量")
                st.pyplot(fig3)
                
                # 数据统计表格
                st.markdown("### 详细统计")
                
                stats_data = {
                    "指标": ["用户数", "物品数", "总交互数", "矩阵密度", 
                           "平均用户交互数", "平均物品被交互数", "用户支持矩阵", "物品支持矩阵"],
                    "值": [
                        rating.shape[0],
                        rating.shape[1],
                        int((rating != 0).sum().item()),
                        f"{density:.4f}%",
                        f"{user_interactions.mean():.2f}",
                        f"{item_popularity.mean():.2f}",
                        len(support_user),
                        len(support_item)
                    ]
                }
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ 数据加载失败: {e}")
            st.info("💡 建议先运行数据准备脚本或使用示例数据")

with tab3:
    st.markdown('<h2 class="sub-header">模型推理</h2>', unsafe_allow_html=True)
    
    # 模型加载状态
    model_loaded = False
    model = None
    
    # 加载模型按钮
    if st.button("🤖 加载模型", type="primary"):
        try:
            with st.spinner("正在加载模型..."):
                # 导入模型
                from models.models import AGCNrec
                from utils.data_utils import load_data
                
                # 加载数据
                rating, features_item, features_user, support_user, support_item, negative = load_data(
                    user=['uku'],
                    item=['kuk'],
                    data_dir=data_dir
                )
                
                # 创建placeholders
                placeholders = {
                    'rating': rating,
                    'features_user': features_user,
                    'features_item': features_item,
                    'negative': negative
                }
                
                # 创建模型
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
                    model_loaded = True
                    
                    # 显示模型信息
                    with st.expander("📋 模型信息"):
                        model_summary = model.summary()
                        st.json(model_summary)
                else:
                    st.warning("⚠️ 模型文件不存在，使用随机初始化权重")
                    model_loaded = True
                
        except Exception as e:
            st.error(f"❌ 模型加载失败: {e}")
    
    if model_loaded and model is not None:
        # 用户选择
        st.markdown("### 选择用户进行推荐")
        
        col_user1, col_user2 = st.columns(2)
        
        with col_user1:
            max_user_id = model.user_dim - 1 if hasattr(model, 'user_dim') else 100
            user_id = st.number_input(
                "用户ID", 
                min_value=0, 
                max_value=max(0, max_user_id), 
                value=0,
                help=f"选择用户ID (0-{max_user_id})"
            )
        
        with col_user2:
            top_k = st.slider(
                "推荐数量", 
                min_value=1, 
                max_value=20, 
                value=10,
                help="选择要生成的推荐数量"
            )
        
        # 生成推荐按钮
        if st.button("🎯 生成推荐", type="primary"):
            try:
                with st.spinner("正在生成推荐..."):
                    # 确保有数据
                    from utils.data_utils import load_data
                    rating, features_item, features_user, support_user, support_item, negative = load_data(
                        user=['uku'],
                        item=['kuk'],
                        data_dir=data_dir
                    )
                    
                    # 设置设备
                    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
                    model = model.to(device)
                    
                    # 移动数据到设备
                    features_user = features_user.to(device)
                    features_item = features_item.to(device)
                    support_user = [sup.to(device) for sup in support_user]
                    support_item = [sup.to(device) for sup in support_item]
                    
                    # 前向传播
                    with torch.no_grad():
                        model.eval()
                        rate_matrix = model.forward(
                            features_user, features_item,
                            support_user, support_item
                        )
                        
                        # 获取用户的评分
                        user_ratings = rate_matrix[user_id, :]
                        top_scores, top_indices = torch.topk(user_ratings, k=min(top_k, len(user_ratings)))
                        
                        # 显示推荐结果
                        st.markdown(f"### 用户 {user_id} 的Top-{top_k}推荐")
                        
                        # 创建结果表格
                        results = []
                        for i, (score, idx) in enumerate(zip(top_scores, top_indices)):
                            results.append({
                                "排名": i + 1,
                                "物品ID": idx.item(),
                                "预测评分": f"{score.item():.4f}",
                                "星级": "⭐" * min(5, int(score.item() / 1.0 + 0.5))
                            })
                        
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # 可视化
                        col_viz1, col_viz2 = st.columns(2)
                        
                        with col_viz1:
                            # 评分柱状图
                            fig1, ax1 = plt.subplots(figsize=(8, 4))
                            ax1.bar(range(len(top_scores)), top_scores.cpu().numpy(), color='skyblue')
                            ax1.set_xlabel("推荐排名")
                            ax1.set_ylabel("预测评分")
                            ax1.set_title(f"用户 {user_id} 的Top-{top_k}推荐评分")
                            ax1.set_xticks(range(len(top_scores)))
                            ax1.set_xticklabels([str(i+1) for i in range(len(top_scores))])
                            st.pyplot(fig1)
                        
                        with col_viz2:
                            # 评分分布
                            fig2, ax2 = plt.subplots(figsize=(8, 4))
                            all_scores = user_ratings.cpu().numpy()
                            ax2.hist(all_scores, bins=20, alpha=0.7, color='lightgreen')
                            ax2.axvline(x=top_scores[-1].item(), color='red', linestyle='--', label='Top-K阈值')
                            ax2.set_xlabel("预测评分")
                            ax2.set_ylabel("物品数量")
                            ax2.set_title(f"用户 {user_id} 的所有物品评分分布")
                            ax2.legend()
                            st.pyplot(fig2)
                        
                        # 推荐解释（可选）
                        with st.expander("🔍 推荐解释"):
                            st.markdown(f"""
                            ### 推荐结果分析
                            
                            - **用户ID**: {user_id}
                            - **推荐数量**: {top_k}
                            - **最高评分**: {top_scores[0].item():.4f}
                            - **平均评分**: {top_scores.mean().item():.4f}
                            - **评分范围**: {top_scores[-1].item():.4f} - {top_scores[0].item():.4f}
                            
                            ### 推荐质量
                            
                            根据预测评分，这些物品与用户的兴趣匹配度较高。
                            高评分的物品表明模型认为这些内容最符合用户的学习需求。
                            """)
                            
            except Exception as e:
                st.error(f"❌ 推荐生成失败: {e}")

with tab4:
    st.markdown('<h2 class="sub-header">性能评估</h2>', unsafe_allow_html=True)
    
    # 评估选项
    col_eval1, col_eval2 = st.columns(2)
    
    with col_eval1:
        eval_model_path = st.selectbox(
            "选择评估模型",
            options=["./saved_models/best_model.pth", "./saved_models/final_model.pth", "当前加载模型"],
            index=0
        )
    
    with col_eval2:
        run_eval_btn = st.button("📊 运行评估", type="primary")
    
    if run_eval_btn:
        try:
            with st.spinner("正在评估模型性能..."):
                # 导入必要的模块
                from models.models import AGCNrec
                from utils.data_utils import load_data
                from utils.metrics import print_metrics
                
                # 加载数据
                rating, features_item, features_user, support_user, support_item, negative = load_data(
                    user=['uku'],
                    item=['kuk'],
                    data_dir=data_dir
                )
                
                # 创建placeholders
                placeholders = {
                    'rating': rating,
                    'features_user': features_user,
                    'features_item': features_item,
                    'negative': negative
                }
                
                # 创建模型
                model = AGCNrec(
                    placeholders=placeholders,
                    input_dim_user=features_user.shape[1],
                    input_dim_item=features_item.shape[1],
                    user_dim=rating.shape[0],
                    item_dim=rating.shape[1],
                    learning_rate=0.001
                )
                
                # 加载指定模型
                if eval_model_path != "当前加载模型":
                    if os.path.exists(eval_model_path):
                        model.load(eval_model_path)
                        st.success(f"✅ 加载模型: {eval_model_path}")
                    else:
                        st.warning(f"⚠️ 模型文件不存在: {eval_model_path}")
                        st.info("使用随机初始化模型进行评估")
                
                # 设置设备
                device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
                model = model.to(device)
                
                # 移动数据
                features_user = features_user.to(device)
                features_item = features_item.to(device)
                rating = rating.to(device)
                negative = negative.to(device)
                support_user = [sup.to(device) for sup in support_user]
                support_item = [sup.to(device) for sup in support_item]
                
                # 准备批处理数据
                batch_data = {
                    'features_user': features_user,
                    'features_item': features_item,
                    'rating': rating,
                    'supports_user': support_user,
                    'supports_item': support_item,
                    'negative': negative
                }
                
                # 评估
                with torch.no_grad():
                    metrics = model.evaluate(batch_data)
                
                # 显示结果
                st.success("✅ 评估完成!")
                
                # 创建指标卡片
                st.markdown("### 评估结果")
                
                # 按K值分组显示
                hr_metrics = {k: v for k, v in metrics.items() if k.startswith('hr@') and int(k.split('@')[1]) in k_values}
                ndcg_metrics = {k: v for k, v in metrics.items() if k.startswith('ndcg@') and int(k.split('@')[1]) in k_values}
                
                # HR指标
                st.markdown("#### Hit Rate (HR)")
                cols_hr = st.columns(len(hr_metrics))
                for idx, (k, v) in enumerate(sorted(hr_metrics.items(), key=lambda x: int(x[0].split('@')[1]))):
                    with cols_hr[idx]:
                        st.metric(f"HR@{k.split('@')[1]}", f"{v:.4f}")
                
                # NDCG指标
                st.markdown("#### Normalized DCG")
                cols_ndcg = st.columns(len(ndcg_metrics))
                for idx, (k, v) in enumerate(sorted(ndcg_metrics.items(), key=lambda x: int(x[0].split('@')[1]))):
                    with cols_ndcg[idx]:
                        st.metric(f"NDCG@{k.split('@')[1]}", f"{v:.4f}")
                
                # 其他指标
                st.markdown("#### 其他指标")
                other_metrics = {k: v for k, v in metrics.items() if not k.startswith('hr@') and not k.startswith('ndcg@')}
                cols_other = st.columns(len(other_metrics))
                for idx, (k, v) in enumerate(other_metrics.items()):
                    with cols_other[idx]:
                        st.metric(k.upper(), f"{v:.4f}")
                
                # 可视化
                st.markdown("### 指标可视化")
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                # HR@K曲线
                hr_values = [metrics.get(f'hr@{k}', 0) for k in sorted(k_values)]
                axes[0].plot(sorted(k_values), hr_values, marker='o', linewidth=2, color='blue')
                axes[0].set_xlabel('K')
                axes[0].set_ylabel('Hit Rate')
                axes[0].set_title('HR@K 曲线')
                axes[0].grid(True, alpha=0.3)
                
                # NDCG@K曲线
                ndcg_values = [metrics.get(f'ndcg@{k}', 0) for k in sorted([k for k in k_values if k in [5, 10, 20]])]
                ndcg_k_values = [k for k in k_values if k in [5, 10, 20]]
                if ndcg_values:
                    axes[1].plot(ndcg_k_values, ndcg_values, marker='s', linewidth=2, color='green')
                    axes[1].set_xlabel('K')
                    axes[1].set_ylabel('NDCG')
                    axes[1].set_title('NDCG@K 曲线')
                    axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # 详细指标表格
                st.markdown("### 详细指标")
                
                metrics_df = pd.DataFrame([
                    {"指标": k, "值": f"{v:.4f}"}
                    for k, v in metrics.items()
                ])
                
                st.dataframe(metrics_df, use_container_width=True)
                
                # 评估总结
                st.markdown("### 评估总结")
                
                best_hr = max([v for k, v in metrics.items() if k.startswith('hr@')])
                best_hr_k = [k for k, v in metrics.items() if k.startswith('hr@') and v == best_hr][0]
                
                col_sum1, col_sum2 = st.columns(2)
                
                with col_sum1:
                    st.info(f"**最佳命中率**: {best_hr:.4f} ({best_hr_k})")
                    st.info(f"**平均倒数排名**: {metrics.get('mrr', 0):.4f}")
                
                with col_sum2:
                    st.info(f"**曲线下面积**: {metrics.get('auc', 0):.4f}")
                    st.info(f"**评估用户数**: {model.user_dim if hasattr(model, 'user_dim') else 'N/A'}")
                
        except Exception as e:
            st.error(f"❌ 评估失败: {e}")
            st.info("💡 请确保模型文件存在且数据可用")

# 页脚
st.sidebar.markdown("---")
st.sidebar.markdown("""
#### 📚 关于ACKRec

- **论文**: [arXiv:2006.13257](https://arxiv.org/abs/2006.13257)
- **GitHub**: [AI4Edu-Group/ACKRec](https://github.com/AI4Edu-Group/ACKRec)
- **版本**: 1.0.0

#### 📧 支持

如有问题或建议，请提交GitHub Issue或联系我们。
""")

# 运行状态
if st.sidebar.button("🔄 检查系统状态"):
    import subprocess
    result = subprocess.run([sys.executable, "--version"], capture_output=True, text=True)
    st.sidebar.code(f"Python: {result.stdout.strip()}")
    st.sidebar.code(f"PyTorch: {torch.__version__}")
    st.sidebar.code(f"CUDA: {'可用' if torch.cuda.is_available() else '不可用'}")

# 添加刷新按钮
if st.sidebar.button("🔄 刷新页面"):
    st.rerun()