import torch
import numpy as np
from typing import Union
import torch.nn as nn
from torch.nn import functional as F


class multimodal_proxy_loss(torch.nn.Module):
    def __init__(self, numclass, output_dim, bit, threshold=0, alpha=0.8):
        torch.nn.Module.__init__(self)
        # 随机数种子选择0
        torch.manual_seed(0)
        # Initialization
        self.alpha = alpha
        self.numclass = numclass
        self.output_dim = output_dim
        self.threshold = threshold
        self.bit = bit
        self.proxies = torch.nn.Parameter(torch.randn(self.numclass, self.bit).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

    def forward(self, x, y, label):
        """
        parem: x  图像哈希 [128, 16]
        parem: y, 文本哈希 [128, 16]
        label [128, 24]
        """
        # 初始化
        P_one_hot = label

        """
        计算了 x 和 self.proxies 中每个归一化向量之间的余弦相似度，并将结果存储在 cos 中。这通常用于计算查询（x）与一组原型（self.proxies）之间的相似度，例如在最近邻搜索、分类或聚类等任务中。
        """
        cos = F.normalize(x, p=2, dim=1).mm(F.normalize(self.proxies, p=2, dim=1).T)
        # 正负相似度
        pos = 1 - cos
        neg = F.relu(cos - self.threshold)
        # 计算余弦相似度:
        cos_t = F.normalize(y, p=2, dim=1).mm(F.normalize(self.proxies, p=2, dim=1).T)
        # 计算正向
        # 由于余弦相似度的值范围在 [-1, 1] 之间，1 - cos_t 的结果将是一个在 [0, 2] 范围内的值，其中 0 表示完全不相似，2 表示完全相似。
        pos_t = 1 - cos_t
        # 负项 neg_t 是通过应用 ReLU 函数（F.relu）到 cos_t - threshold 的结果来计算的。这里，threshold 是一个阈值，
        # 用于确定何时应该考虑一个样本和一个代理之间的相似度为负。只有当 cos_t 小于 threshold 时，neg_t 的值才会是非零的（具体为 cos_t - threshold），
        # 否则为 0。这通常用于鼓励模型学习不同类别之间的更大间隔。
        neg_t = F.relu(cos_t - self.threshold)

        # 统计每个样本有多少个正标签
        P_num = len(P_one_hot.nonzero())
        # 统计这代表了每个样本有多少个负标签（即多少个位置上的值是0）。
        N_num = len((P_one_hot == 0).nonzero())

        """
        这段代码的目的是计算多标签分类任务中的正项和负项，这些项将用于计算最终的损失函数。正项鼓励模型将相同类别的样本表示得更接近，而负项则鼓励模型将不同类别的样本表示得更远。
        """
        pos_term = torch.where(P_one_hot == 1, pos.to(torch.float32),
                               torch.zeros_like(cos).to(torch.float32)).sum() / P_num
        neg_term = torch.where(P_one_hot == 0, neg.to(torch.float32),
                               torch.zeros_like(cos).to(torch.float32)).sum() / N_num

        pos_term_t = torch.where(P_one_hot == 1, pos_t.to(torch.float32),
                                 torch.zeros_like(cos_t).to(torch.float32)).sum() / P_num
        neg_term_t = torch.where(P_one_hot == 0, neg_t.to(torch.float32),
                                 torch.zeros_like(cos_t).to(torch.float32)).sum() / N_num

        # 判断其是否启用了正则化，若未启用则直接设置正则化为0
        if self.alpha > 0:

            """
            这里筛选出了那些具有多于一个标签的样本。label.sum(dim=1) > 1 会返回一个布尔向量，
            其中每个元素表示对应样本是否有多于一个的标签。然后，根据这个布尔向量筛选出对应的标签、特征 x 和目标 t。
            """
            index = label.sum(dim=1) > 1
            label_ = label[index].float()

            x_ = x[index]
            t_ = y[index]
            # 计算了筛选后的标签之间的余弦相似度矩阵 cos_sim
            cos_sim = label_.mm(label_.T)
            """
            如果 self.args.alpha 不大于 0，则正则化项被设置为 0；否则，它们是基于样本之间的相似度计算得到的。这些正则化项将用于最终的损失计算，以鼓励模型学习到更好的表示。
            """
            if len((cos_sim == 0).nonzero()) == 0:
                # 如果 cos_sim 中没有值为 0 的元素（即所有样本对至少共享一个标签），则不计算正则化项，直接设置为 0。
                reg_term = 0
                reg_term_t = 0
                reg_term_xt = 0
            else:
                x_sim = F.normalize(x_, p=2, dim=1).mm(F.normalize(x_, p=2, dim=1).T)
                t_sim = F.normalize(t_, p=2, dim=1).mm(F.normalize(t_, p=2, dim=1).T)
                xt_sim = F.normalize(x_, p=2, dim=1).mm(F.normalize(t_, p=2, dim=1).T)

                neg = self.alpha * F.relu(x_sim - self.threshold)
                neg_t = self.alpha * F.relu(t_sim - self.threshold)
                neg_xt = self.alpha * F.relu(xt_sim - self.threshold)

                reg_term = torch.where(cos_sim == 0, neg, torch.zeros_like(x_sim)).sum() / len((cos_sim == 0).nonzero())
                reg_term_t = torch.where(cos_sim == 0, neg_t, torch.zeros_like(t_sim)).sum() / len(
                    (cos_sim == 0).nonzero())
                reg_term_xt = torch.where(cos_sim == 0, neg_xt, torch.zeros_like(xt_sim)).sum() / len(
                    (cos_sim == 0).nonzero())
        else:
            reg_term = 0
            reg_term_t = 0
            reg_term_xt = 0
        return pos_term + neg_term + pos_term_t + neg_term_t + reg_term + reg_term_t + reg_term_xt


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes=range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T


class Proxy_Anchor(torch.nn.Module):
    def __init__(self, numclass, output_dim, bit, mrg=0.1, alpha=32):  # 类别总数，嵌入向量的维度，边距（margin）默认0.1，
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization  是一个可训练的参数，表示每个类别的代理向量 使用 torch.randn初始化代理向量，形状为 (nb_classes, sz_embed)
        self.bit = bit
        self.nb_classes = numclass
        self.sz_embed = output_dim
        self.proxies = torch.nn.Parameter(torch.randn(self.nb_classes, self.bit).cuda())
        # 使用 Kaiming 初始化方法（nn.init.kaiming_normal_）对代理向量进行初始化，mode='fan_out' 表示按输出维度进行归一化
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.mrg = mrg
        self.alpha = alpha

    def forward(self, X, T):  # X：输入的嵌入向量，形状为 (batch_size, sz_embed)。T：目标标签，形状为 (batch_size,)，表示每个样本的类别索引。
        P = self.proxies  # 是代理向量
        # 使用 F.linear 计算输入嵌入向量 X 和代理向量 P 的余弦相似度。由于 X 和 P 都是归一化的，因此 F.linear 的结果就是它们的点积，即余弦相似度
        cos = F.normalize(X, p=2, dim=1).mm(F.normalize(self.proxies, p=2, dim=1).T)  # Calcluate cosine similarity
        # 将目标标签 T 转换为独热编码形式，形状为 (batch_size, nb_classes)
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)  # 表示正样本的独热编码
        N_one_hot = 1 - P_one_hot  # 表示负样本的独热编码
        # 计算正样本的指数项 pos_exp
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        # 计算负样本的指数项 pos_exp
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))
        # P_one_hot.sum(dim=0) 按列求和，得到每个类别的正样本数量。
        # torch.nonzero(P_one_hot.sum(dim = 0) != 0) 找出有正样本的类别索引。
        # squeeze(dim=1) 去掉多余的维度。
        # num_valid_proxies 是有正样本的代理数量
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies
        # 使用 torch.where，将正样本的指数项 pos_exp 和负样本的指数项 neg_exp 分别累加到对应的类别上
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        #
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term

        return loss


class RobustProxyAnchorHashLoss(nn.Module):
    def __init__(self, n_label, n_bit, mrg=0.1, alpha=32, tau=1.0, shift=1.0, lambda_q=0.1, lambda_cross=1.0,
                 lambda_inner=0.5):
        super().__init__()
        # 一对代理
        self.proxies_u = nn.Parameter(torch.randn(n_label, n_bit))  # 图像代理
        self.proxies_v = nn.Parameter(torch.randn(n_label, n_bit))  # 文本代理
        nn.init.kaiming_normal_(self.proxies_u, mode='fan_out')
        nn.init.kaiming_normal_(self.proxies_v, mode='fan_out')

        self.n_label = n_label
        self.bit = n_bit
        self.mrg = mrg
        self.alpha = alpha
        self.tau = tau
        self.shift = shift

        # 权重系数
        self.lambda_q = lambda_q
        self.lambda_cross = lambda_cross
        self.lambda_inner = lambda_inner

    # ----------- 工具：Proxy-Anchor 基础单元 -----------
    def _anchor_loss(self, Z, P, mask):
        """
        Z : [B, b]  样本哈希（已 tanh+归一化）
        P : [L, b]  代理哈希（已 tanh+归一化）
        mask: [B, L]  多热标签 0/1
        """
        Z = Z.to(Z.device)
        P = P.to(Z.device)
        mask = mask.to(Z.device)
        cos = Z.mm(P.t())  # [B, L]

        mask_pos = mask
        mask_neg = 1 - mask

        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        P_sum = (mask_pos * pos_exp).sum(0)  # 每个代理的正样本
        N_sum = (mask_neg * neg_exp).sum(0)

        # 只考虑 batch 里出现过的标签
        valid = mask.sum(0).nonzero(as_tuple=False).squeeze(1)
        n_valid = max(len(valid), 1)

        pos_term = torch.log(1 + P_sum[valid]).sum() / n_valid
        neg_term = torch.log(1 + N_sum).sum() / self.n_label
        return pos_term + neg_term

    # ----------- 主入口 -----------
    def forward(self, U, V, Y):
        """
        U : [B, b] 图像塔 tanh 前输出
        V : [B, b] 文本塔 tanh 前输出
        Y : [B, L] 多热标签 0/1
        """
        # 1. 压缩 + 归一化
        U = F.normalize(U.tanh(), p=2, dim=1)
        V = F.normalize(V.tanh(), p=2, dim=1)
        Pu = F.normalize(self.proxies_u.tanh(), p=2, dim=1)
        Pv = F.normalize(self.proxies_v.tanh(), p=2, dim=1)

        # 2. 跨模态 proxy-anchor
        loss_u2v = self._anchor_loss(U, Pv, Y)  # 图像→文本代理
        loss_v2u = self._anchor_loss(V, Pu, Y)  # 文本→图像代理
        loss_cross = (loss_u2v + loss_v2u) / 2

        # 3. 模态内（可选）
        loss_u2u = self._anchor_loss(U, Pu, Y)
        loss_v2v = self._anchor_loss(V, Pv, Y)
        loss_inner = (loss_u2u + loss_v2v) / 2

        # 4. 量化
        quant_u = (U.abs() - 1 / np.sqrt(self.bit)).pow(2).mean()
        quant_v = (V.abs() - 1 / np.sqrt(self.bit)).pow(2).mean()
        quant_pu = (self.proxies_u.abs() - 1 / np.sqrt(self.bit)).pow(2).mean()
        quant_pv = (self.proxies_v.abs() - 1 / np.sqrt(self.bit)).pow(2).mean()
        quant_loss = quant_u + quant_v + quant_pu + quant_pv

        loss = self.lambda_cross * loss_cross + \
               self.lambda_inner * loss_inner + \
               self.lambda_q * quant_loss

        return loss

