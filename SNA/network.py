import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torchvision import models
import time
import torch
import scipy.misc
import scipy.io
from torch import nn


class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.module_name = str(type(self))

    def load(self, path, use阿_gpu=False):
        if not use_gpu:
            self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(torch.load(path))

    def save(self, name=None):
        if name is None:
            prefix = self.module_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), 'checkpoint/' + name)
        return name

    def forward(self, *input):
        pass


class ImgModule(BasicModule):
    def __init__(self, y_dim, bit, n_class,norm=True, mid_num1=1024 * 8, mid_num2=1024 * 8, hiden_layer=3):
        super(ImgModule, self).__init__()
        self.module_name = "image_model"
        mid_num1 = mid_num1 if hiden_layer > 1 else bit  # 就直接 y_dim → bit，不需要中间大通道。
        modules = [nn.Linear(y_dim, mid_num1)]  # 列表逐层拼装，首层：(B, y_dim) → (B, mid_num1)
        if hiden_layer >= 2:
            modules += [nn.ReLU(inplace=True)]
            pre_num = mid_num1
            for i in range(hiden_layer - 2):  # hiden_layer-2 是“中间重复块”个数
                if i == 0:
                    modules += [nn.Linear(mid_num1, mid_num2),
                                nn.ReLU(inplace=True)]  # 第一次把 8192 → 8192，后面都是 8192 → 8192。
                else:
                    modules += [nn.Linear(mid_num2, mid_num2), nn.ReLU(inplace=True)]  #
                pre_num = mid_num2  # 每块形状：(B, 8192) → (B, 8192)
            modules += [nn.Linear(pre_num, bit)]  # 最后一层把通道压到哈希长度：(B, pre_num) → (B, bit)
        self.fc = nn.Sequential(*modules)  # 把列表展开成顺序容器，前向一次性跑完
        self.norm = norm  # 保存标志位，后面决定要不要做 L2 归一化。
        self.classifier = nn.Linear(bit, n_class)

    def forward(self, x,return_logits=False):
        out = self.fc(x).tanh()  # (B, y_dim) → (B, bit) 再 tanh 到 (-1,1)
        if self.norm:
            norm_x = torch.norm(out, dim=1, keepdim=True)  # (B,1) 各样本的 L2 范数
            out = out / norm_x  # 单位超球面投影，(B, bit)
        logits = self.classifier(out)  # ✅ 分类 logits
        if return_logits:
            return out, logits
        return out


class TxtModule(BasicModule):
    def __init__(self, y_dim, bit, n_class,norm=True, mid_num1=1024 * 8, mid_num2=1024 * 8, hiden_layer=2):
        super(TxtModule, self).__init__()
        self.module_name = "text_model"
        mid_num1 = mid_num1 if hiden_layer > 1 else bit  # 若只建 1 层，则直接 y_dim → bit，省去中间大宽层。
        modules = [nn.Linear(y_dim, mid_num1)]  # 列表拼装，首层：(B, y_dim) → (B, mid_num1)
        if hiden_layer >= 2:  # 至少两层时才加激活。
            modules += [nn.ReLU(inplace=True)]
            pre_num = mid_num1
            for i in range(
                    hiden_layer - 2):  # hiden_layer-2 是“中间重复块”个数。第一次 8192 → 8192，后面都是 8192 → 8192。每块形状：(B, 8192) → (B, 8192)
                if i == 0:
                    modules += [nn.Linear(mid_num1, mid_num2), nn.ReLU(inplace=True)]
                else:
                    modules += [nn.Linear(mid_num2, mid_num2), nn.ReLU(inplace=True)]
                pre_num = mid_num2
            modules += [nn.Linear(pre_num, bit)]  # 末层把通道压到哈希长度：(B, pre_num) → (B, bit)
        self.fc = nn.Sequential(*modules)  # 把列表展开成顺序容器，前向一次性跑完。
        self.norm = norm  # 保存标志位，决定出口是否做 L2 归一化。
        self.classifier = nn.Linear(bit, n_class)

    def forward(self, x,return_logits=False):
        out = self.fc(x).tanh()  # (B, y_dim) → (B, bit) 再 tanh 到 (-1,1)
        if self.norm:
            norm_x = torch.norm(out, dim=1, keepdim=True)  # (B,1) 各样本 L2 范数
            out = out / norm_x  # 单位超球面投影，(B, bit)
        logits = self.classifier(out)  # ✅ 分类 logits
        if return_logits:
            return out, logits
        return out


class MyNet(BasicModule):
    def __init__(self, ori_featI, ori_featT):
        super(MyNet, self).__init__()

        self.EncoderImg = nn.Sequential(nn.Linear(ori_featI, 512),
                                        nn.BatchNorm1d(512),
                                        nn.ReLU(inplace=True)
                                        )
        self.AttentionLayerImg = AttentionLayer(512, ori_featT, n_heads=4)
        self.FcImg = nn.Linear(2 * 512, 512)

        self.EncoderTxt = nn.Sequential(nn.Linear(ori_featT, 512),
                                        nn.BatchNorm1d(512),
                                        nn.ReLU(inplace=True)
                                        )
        self.AttentionLayerTxt = AttentionLayer(512, ori_featI, n_heads=4)
        self.FcTxt = nn.Linear(2 * 512, 512)

        # 维度适配层，将增强后的特征转换为哈希网络的输入维度（y_dim）
        self.AdjustDimsImg = nn.Linear(512, ori_featI)  # 调整图像特征维度
        self.AdjustDimsTxt = nn.Linear(512, ori_featT)  # 调整图像特征维度

    def forward(self, img, txt):
        self.batch_size = img.size(0)

        img_coarse = self.EncoderImg(img)
        img_fine = self.AttentionLayerImg(img, txt, txt)
        img_feature = self.FcImg(torch.cat((img_coarse, img_fine), 1))
        img_feature = self.AdjustDimsImg(img_feature)

        txt_coarse = self.EncoderTxt(txt)
        txt_fine = self.AttentionLayerTxt(txt, img, img)
        txt_feature = self.FcTxt(torch.cat((txt_coarse, txt_fine), 1))
        txt_feature = self.AdjustDimsTxt(txt_feature)

        return img_feature, txt_feature


class AttentionLayer(BasicModule):
    def __init__(self, data_dim, hidden_dim, n_heads=4):
        super(AttentionLayer, self).__init__()

        assert hidden_dim % n_heads == 0

        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.fc_q = nn.Linear(data_dim, hidden_dim)
        self.fc_k = nn.Linear(data_dim, hidden_dim)
        self.fc_v = nn.Linear(data_dim, hidden_dim)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).cuda()
        self.dense = nn.Linear(hidden_dim, data_dim)
        self.bn = nn.BatchNorm1d(data_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        Q = self.fc_q(query).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).cuda()
        K = self.fc_k(key).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).cuda()
        V = self.fc_v(value).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).cuda()

        att_map = torch.softmax((torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale), dim=-1)
        output = torch.matmul(att_map, V).view(batch_size, -1)

        output = self.dense(output)
        output = self.bn(output)
        output = self.relu(output)

        return output


class AttentionLayerimg(BasicModule):
    def __init__(self, data_dim, label_dim, hidden_dim, n_heads=4):
        super(AttentionLayerimg, self).__init__()

        assert hidden_dim % n_heads == 0

        self.data_dim = data_dim
        self.label_dim = label_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.fc_q = nn.Linear(label_dim, hidden_dim)
        self.fc_k = nn.Linear(4096, hidden_dim)
        self.fc_v = nn.Linear(4096, hidden_dim)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).cuda()
        self.dense = nn.Linear(hidden_dim, data_dim)
        self.bn = nn.BatchNorm1d(data_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, data_tensor, label_tensor):
        batch_size = data_tensor.shape[0]
        Q = self.fc_q(label_tensor).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).cuda()
        K = self.fc_k(data_tensor).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).cuda()
        V = self.fc_v(data_tensor).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).cuda()

        att_map = torch.softmax((torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale), dim=-1)
        output = torch.matmul(att_map, V).view(batch_size, -1)

        output = self.dense(output)
        output = self.bn(output)
        output = self.relu(output)

        return output
