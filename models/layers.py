import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from .layers import GraphConvolution, SimpleAttLayer, RateLayer
import warnings

class Model(nn.Module):
    def __init__(self, **kwargs):
        super(Model, self).__init__()
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, f'Invalid keyword argument: {kwarg}'
        self.name = kwargs.get('name', self.__class__.__name__.lower())
        self.logging = kwargs.get('logging', False)
        
        self.layers = nn.ModuleList()
        self.activations = []
        self.outputs = None
        self.test = None
        self.alphas = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """构建模型"""
        self._build()

    def forward(self, inputs, supports):
        """
        PyTorch前向传播
        
        Args:
            inputs: 输入特征
            supports: 支持矩阵
            
        Returns:
            模型输出
        """
        self.activations = [inputs]
        
        for i, layer in enumerate(self.layers):
            current_input = self.activations[-1]
            
            if isinstance(layer, GraphConvolution):
                # GraphConvolution处理
                hidden = layer(current_input, supports)
                
                # GraphConvolution返回列表，取平均值
                if isinstance(hidden, list) and len(hidden) > 0:
                    hidden = torch.stack(hidden).mean(dim=0)
                    
            elif isinstance(layer, SimpleAttLayer):
                # SimpleAttLayer处理
                hidden = layer(current_input)
            else:
                hidden = layer(current_input)
            
            # 保存测试输出
            if i == 2:  # 第三个GCN层后保存
                self.test = hidden
                
            self.activations.append(hidden)
        
        self.outputs = self.activations[-1]
        return self.outputs

    def _loss(self):
        raise NotImplementedError

    def summary(self):
        """打印模型摘要"""
        print(f"Model: {self.name}")
        print("-" * 50)
        total_params = 0
        trainable_params = 0
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name:30} {tuple(param.shape):20} {param.numel():8,} params")
                trainable_params += param.numel()
            total_params += param.numel()
        
        print("-" * 50)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Non-trainable params: {total_params - trainable_params:,}")
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': total_params - trainable_params
        }


class GCN(Model):
    def __init__(self, input_dim, tag, length, hidden_dims=[256, 128, 64], dropout_rate=0.5, **kwargs):
        super(GCN, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1]
        self.tag = tag
        self.length = length
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.build()

    def _loss(self):
        """计算L2正则化损失"""
        l2_loss = 0.0
        for param in self.parameters():
            if param.requires_grad:
                l2_loss += 5e-4 * torch.norm(param, p=2)
        return l2_loss

    def _build(self):
        """构建图卷积网络"""
        input_dim = self.input_dim
        for i, hidden_dim in enumerate(self.hidden_dims):
            self.layers.append(
                GraphConvolution(
                    input_dim=input_dim,
                    output_dim=hidden_dim,
                    length=self.length,
                    tag=self.tag,
                    dropout=self.dropout_rate,
                    act=F.relu,
                    sparse_inputs=False,
                    featureless=False
                )
            )
            input_dim = hidden_dim
        
        # 注意力层
        self.layers.append(
            SimpleAttLayer(
                attention_size=32,
                tag=self.tag
            )
        )
    
    def get_layer_outputs(self):
        """获取各层的输出（用于可视化）"""
        return self.activations if hasattr(self, 'activations') else []


class AGCNrec(nn.Module):
    def __init__(self, placeholders, input_dim_user, input_dim_item, user_dim, item_dim, learning_rate=0.001):
        super(AGCNrec, self).__init__()
        self.placeholders = placeholders
        self.negative = placeholders['negative']
        self.length = user_dim
        self.user_dim = user_dim
        self.item_dim = item_dim
        
        # 初始化用户和物品的GCN模型
        self.userModel = GCN(
            input_dim=input_dim_user,
            tag='user',
            length=user_dim,
            hidden_dims=[256, 128, 64],
            dropout_rate=0.5
        )
        
        self.itemModel = GCN(
            input_dim=input_dim_item,
            tag='item',
            length=item_dim,
            hidden_dims=[256, 128, 64],
            dropout_rate=0.5
        )
        
        # 评分层
        self.rate_layer = RateLayer(
            user_dim=user_dim,
            item_dim=item_dim,
            latent_dim=30,
            output_dim=64
        )
        
        # 优化器
        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        self.rate_matrix = None
        
        # 训练历史
        self.train_history = {
            'loss': [],
            'metrics': {}
        }

    def forward(self, features_user, features_item, supports_user, supports_item):
        """
        前向传播
        
        Args:
            features_user: 用户特征
            features_item: 物品特征
            supports_user: 用户支持矩阵
            supports_item: 物品支持矩阵
            
        Returns:
            评分矩阵
        """
        # 前向传播计算用户和物品嵌入
        user_emb = self.userModel(features_user, supports_user)
        item_emb = self.itemModel(features_item, supports_item)
        
        # 计算评分矩阵
        self.rate_matrix = self.rate_layer(user_emb, item_emb)
        return self.rate_matrix

    def loss(self, rating_matrix):
        """
        计算总损失
        
        Args:
            rating_matrix: 真实评分矩阵
            
        Returns:
            总损失值
        """
        total_loss = self.userModel._loss() + self.itemModel._loss()
        
        # 评分层参数的L2正则化
        for param in self.rate_layer.parameters():
            if param.requires_grad:
                total_loss += 5e-4 * torch.norm(param, p=2)
        
        # MSE损失（与真实评分的误差）
        if rating_matrix is not None:
            total_loss += F.mse_loss(self.rate_matrix, rating_matrix)
        else:
            warnings.warn("Rating matrix is None, skipping MSE loss calculation")
        
        return total_loss

    def train_step(self, batch_data):
        """
        训练步骤
        
        Args:
            batch_data: 批处理数据
            
        Returns:
            损失值
        """
        features_user = batch_data['features_user']
        features_item = batch_data['features_item']
        supports_user = batch_data['supports_user']
        supports_item = batch_data['supports_item']
        rating_matrix = batch_data.get('rating', None)
        
        self.optimizer.zero_grad()
        self.forward(features_user, features_item, supports_user, supports_item)
        loss = self.loss(rating_matrix)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, batch_data):
        """
        计算评估指标
        
        Args:
            batch_data: 批处理数据
            
        Returns:
            评估指标字典
        """
        try:
            from utils.metrics import hr, ndcg, mrr, auc
        except ImportError:
            # 如果在utils中不可用，使用本地实现
            def hr(rate, negative, length, k=5):
                # 简化实现
                import numpy as np
                test_scores = []
                for i in range(length):
                    user_indices = negative[i, :, 0]
                    item_indices = negative[i, :, 1]
                    scores = rate[user_indices, item_indices]
                    test_scores.append(scores)
                
                test_scores = np.array(test_scores)
                topk_indices = np.argsort(-test_scores, axis=1)[:, :k]
                is_in = (topk_indices == 99)
                row_hits = np.sum(is_in, axis=1)
                total_hits = np.sum(row_hits)
                return total_hits / length
            
            def ndcg(rate, negative, length, k=5):
                # 简化实现
                import numpy as np
                test_scores = []
                for i in range(length):
                    user_indices = negative[i, :, 0]
                    item_indices = negative[i, :, 1]
                    scores = rate[user_indices, item_indices]
                    test_scores.append(scores)
                
                test_scores = np.array(test_scores)
                topk_indices = np.argsort(-test_scores, axis=1)[:, :k]
                dcg_sum = 0
                for i in range(length):
                    pos = np.where(topk_indices[i] == 99)[0]
                    if len(pos) > 0:
                        rank = pos[0] + 1
                        dcg_sum += np.log(2) / np.log(rank + 1)
                
                idcg_sum = 0
                for r in range(1, min(k, 1) + 1):
                    idcg_sum += np.log(2) / np.log(r + 1)
                
                return dcg_sum / length if idcg_sum > 0 else 0.0
            
            def mrr(rate, negative, length):
                # 简化实现
                import numpy as np
                test_scores = []
                for i in range(length):
                    user_indices = negative[i, :, 0]
                    item_indices = negative[i, :, 1]
                    scores = rate[user_indices, item_indices]
                    test_scores.append(scores)
                
                test_scores = np.array(test_scores)
                topk_indices = np.argsort(-test_scores, axis=1)[:, :100]
                mrr_sum = 0
                for i in range(length):
                    pos = np.where(topk_indices[i] == 99)[0]
                    if len(pos) > 0:
                        rank = pos[0] + 1
                        mrr_sum += 1.0 / rank
                
                return mrr_sum / length
            
            def auc(rate, negative, length):
                # 简化实现
                import numpy as np
                test_scores = []
                for i in range(length):
                    user_indices = negative[i, :, 0]
                    item_indices = negative[i, :, 1]
                    scores = rate[user_indices, item_indices]
                    test_scores.append(scores)
                
                test_scores = np.array(test_scores)
                topk_indices = np.argsort(-test_scores, axis=1)[:, :100]
                auc_sum = 0
                for i in range(length):
                    pos = np.where(topk_indices[i] == 99)[0]
                    if len(pos) > 0:
                        rank = pos[0]
                        auc_sum += (100 - rank) / 100.0
                    else:
                        auc_sum += 0.5
                
                return auc_sum / length
        
        with torch.no_grad():
            features_user = batch_data['features_user']
            features_item = batch_data['features_item']
            supports_user = batch_data['supports_user']
            supports_item = batch_data['supports_item']
            
            self.forward(features_user, features_item, supports_user, supports_item)
            
            # 转换为numpy进行计算
            rate_matrix_np = self.rate_matrix.detach().cpu().numpy()
            negative_np = self.negative.cpu().numpy()
            length = self.length
            
            # 计算评估指标
            metrics = {}
            for k in [1, 5, 10, 20]:
                try:
                    metrics[f'hr@{k}'] = hr(rate_matrix_np, negative_np, length, k=k)
                except:
                    metrics[f'hr@{k}'] = 0.0
                
                try:
                    metrics[f'ndcg@{k}'] = ndcg(rate_matrix_np, negative_np, length, k=k)
                except:
                    metrics[f'ndcg@{k}'] = 0.0
            
            try:
                metrics['mrr'] = mrr(rate_matrix_np, negative_np, length)
            except:
                metrics['mrr'] = 0.0
            
            try:
                metrics['auc'] = auc(rate_matrix_np, negative_np, length)
            except:
                metrics['auc'] = 0.0
            
            return metrics

    def predict(self, user_id, top_k=10):
        """
        为指定用户生成推荐
        
        Args:
            user_id: 用户ID
            top_k: 返回的推荐数量
            
        Returns:
            推荐物品ID和评分
        """
        if self.rate_matrix is None:
            raise ValueError("Model not trained. Please run forward() first.")
        
        user_ratings = self.rate_matrix[user_id, :]
        top_scores, top_indices = torch.topk(user_ratings, k=min(top_k, len(user_ratings)))
        
        recommendations = []
        for score, idx in zip(top_scores, top_indices):
            recommendations.append({
                'item_id': idx.item(),
                'score': score.item()
            })
        
        return recommendations

    def save(self, path):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'user_dim': self.user_dim,
            'item_dim': self.item_dim,
            'train_history': self.train_history
        }, path)
        print(f"✅ Model saved to {path}")

    def load(self, path):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'train_history' in checkpoint:
            self.train_history = checkpoint['train_history']
        
        print(f"✅ Model loaded from {path}")
        return checkpoint

    def summary(self):
        """打印模型摘要"""
        print("=" * 60)
        print("AGCNrec Model Summary")
        print("=" * 60)
        
        total_params = 0
        trainable_params = 0
        
        print("\nUser Model:")
        print("-" * 40)
        for name, param in self.userModel.named_parameters():
            if param.requires_grad:
                print(f"  {name:30} {tuple(param.shape):20} {param.numel():8,} params")
                trainable_params += param.numel()
            total_params += param.numel()
        
        print("\nItem Model:")
        print("-" * 40)
        for name, param in self.itemModel.named_parameters():
            if param.requires_grad:
                print(f"  {name:30} {tuple(param.shape):20} {param.numel():8,} params")
                trainable_params += param.numel()
            total_params += param.numel()
        
        print("\nRate Layer:")
        print("-" * 40)
        for name, param in self.rate_layer.named_parameters():
            if param.requires_grad:
                print(f"  {name:30} {tuple(param.shape):20} {param.numel():8,} params")
                trainable_params += param.numel()
            total_params += param.numel()
        
        print("-" * 60)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Non-trainable params: {total_params - trainable_params:,}")
        print("=" * 60)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': total_params - trainable_params
        }


# 导出所有模型
__all__ = [
    'Model',
    'GCN',
    'AGCNrec'
]