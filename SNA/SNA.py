from utils.tools import *
import itertools
from scipy.linalg import hadamard
from network import *
import pdb
import os
import torch
import torch.optim as optim
import time
import numpy as np
import argparse
import random
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.mixture import GaussianMixture
from bmm import BetaMixture1D

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--hash_dim', type=int, default=32)
parser.add_argument('--noise_rate', type=float, default=0.2)
parser.add_argument('--dataset', type=str, default='flickr')
parser.add_argument('--Lambda', type=float, default=0.6)
parser.add_argument('--num_gradual', type=int, default=100)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

bit_len = args.hash_dim
noise_rate = args.noise_rate
dataset = args.dataset
Lambda = args.Lambda
num_gradual = args.num_gradual

if dataset == 'flickr':
    train_size = 10000
elif dataset == 'ms-coco':
    train_size = 10000
elif dataset == 'nuswide21':
    train_size = 10500
elif dataset == 'iapr':
    train_size = 10000
n_class = 0
tag_len = 0
torch.multiprocessing.set_sharing_strategy('file_system')


def get_config():
    config = {
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 3e-5, "weight_decay": 10 ** -5}},
        "txt_optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 3e-5, "weight_decay": 10 ** -5}},
        "info": "[CSQ]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 128,
        "dataset": dataset,
        "epoch": 100,
        "device": torch.device("cuda:0"),
        "bit_len": bit_len,
        "noise_type": 'symmetric',
        "noise_rate": noise_rate,
        "random_state": 1,
        "n_class": n_class,
        "lambda": Lambda,
        "tag_len": tag_len,
        "train_size": train_size,
        "threshold_rate": 0.3,
        "output_dim": 64,
        "beta": 0.6
    }
    return config


class Robust_Loss_noise(nn.Module):
    def __init__(self, config, bit):
        super(Robust_Loss_noise, self).__init__()
        self.tau = 1.
        self.shift = 1
        self.bit = bit
        self.margin = .2

    def forward(self, u, v, y, config):
        u = u.tanh()
        v = v.tanh()
        T = self.calc_neighbor(y, y)
        T.diagonal().fill_(0)
        S = u.mm(v.t())
        d = S.diag().view(v.size(0), 1)
        d1 = d.expand_as(S)
        d2 = d.t().expand_as(S)
        modal_consist_loss = F.mse_loss(u, v)
        mask_te = (S >= (d1 - self.margin)).float().detach()
        cost_te = S * mask_te + (1. - mask_te) * (S - self.shift)

        cost_te_max = torch.zeros_like(cost_te)
        cost_te_max.copy_(cost_te)
        identity_matrix_te = torch.eye(cost_te_max.size(0), cost_te_max.size(1), device=cost_te_max.device,
                                       dtype=cost_te_max.dtype)
        diagonal_te = torch.diag(cost_te_max).clamp(min=0)
        modified_diagonal_matrix_te = torch.diag_embed(diagonal_te)
        cost_te_max = cost_te_max * (1 - identity_matrix_te) + modified_diagonal_matrix_te

        mask_im = (S >= (d2 - self.margin)).float().detach()
        cost_im = S * mask_im + (1. - mask_im) * (S - self.shift)

        cost_im_max = torch.zeros_like(cost_im)
        cost_im_max.copy_(cost_im)
        identity_matrix_im = torch.eye(cost_im_max.size(0), cost_im_max.size(1), device=cost_im_max.device,
                                       dtype=cost_im_max.dtype)
        diagonal_im = torch.diag(cost_im_max).clamp(min=0)
        modified_diagonal_matrix_im = torch.diag_embed(diagonal_im)
        cost_im_max = cost_im_max * (1 - identity_matrix_im) + modified_diagonal_matrix_im

        loss_r = (-cost_te.diag() + self.tau * ((cost_te_max / self.tau * (1 - T))).exp().sum(
            1).log() + self.margin) + (-cost_im.diag() + self.tau * ((cost_te_max / self.tau * (1 - T))).exp().sum(
            1).log() + self.margin)

        positive_mask = ((T - torch.eye(T.shape[0], device=T.device, dtype=T.dtype)) > 0).float()
        separation_term = torch.mean(positive_mask * (self.shift - S).exp())

        Q_loss = (u.abs() - 1 / np.sqrt(self.bit)).pow(2).mean(dim=1) + (v.abs() - 1 / np.sqrt(self.bit)).pow(2).mean(
            dim=1)
        loss = config["lambda"] * (loss_r + separation_term - torch.diag(S).mean()) + (
                    1 - config["lambda"]) * Q_loss + modal_consist_loss
        final_loss = torch.mean(loss)
        return final_loss

    def calc_neighbor(self, label1, label2):
        label1 = label1.float()
        label2 = label2.float()
        return (label1 @ label2.T > 0).float()


class Robust_Loss_clean(nn.Module):
    def __init__(self, config, bit):
        super(Robust_Loss_clean, self).__init__()
        self.tau = 0.5
        self.shift = 0.2
        self.bit = bit

    def forward(self, u, v, y, gmm_probs):
        u = u.tanh()
        v = v.tanh()

        T = self.calc_neighbor(y, y)
        T.fill_diagonal_(0)

        S = u @ v.T
        d = S.diag().view(-1, 1)
        d1 = d.expand_as(S)
        d2 = d.T.expand_as(S)
        margin_matrix = 0.2
        modal_consist_loss = F.mse_loss(u, v)
        
        mask_te = (S >= (d1 - margin_matrix)).float().detach()
        cost_te = S * mask_te + (1. - mask_te) * (S - self.shift)
        cost_te_max = cost_te.clone()
        cost_te_max.fill_diagonal_(0)
        cost_te_max += torch.diag(torch.diag(cost_te).clamp(min=0))

        mask_im = (S >= (d2 - margin_matrix)).float().detach()
        cost_im = S * mask_im + (1. - mask_im) * (S - self.shift)
        cost_im_max = cost_im.clone()
        cost_im_max.fill_diagonal_(0)
        cost_im_max += torch.diag(torch.diag(cost_im).clamp(min=0))

        loss_r = (
                -torch.diag(cost_te) + self.tau * ((cost_te_max / self.tau) * (1 - T)).exp().sum(1).log() +
                -torch.diag(cost_im) + self.tau * ((cost_im_max / self.tau) * (1 - T)).exp().sum(1).log()
        ).mean()

        Q_loss = ((u.abs() - 1 / np.sqrt(self.bit)).pow(2).mean(dim=1) +
                  (v.abs() - 1 / np.sqrt(self.bit)).pow(2).mean(dim=1)).mean()
        return config["lambda"] * loss_r + (1 - config["lambda"]) * Q_loss + modal_consist_loss

    def calc_neighbor(self, label1, label2):
        label1 = label1.float()
        label2 = label2.float()
        return (label1 @ label2.T > 0).float()


def split_prob(prob, threshld):
    pred = (prob >= threshld)
    return (pred + 0)


def get_loss(net, txt_net, config, data_loader, Threshold, epoch, W):
    tau = 0.05
    sample_losses = []
    for image, tag, tlabel, label, ind in data_loader:
        image = image.to('cuda')
        image = image.float()
        tag = tag.to('cuda')
        tag = tag.float()
        label = label.to('cuda')
        tlabel = tlabel.to('cuda')
        u = net(image)
        v = txt_net(tag)
        with torch.no_grad():
            label_ = (label - 0.5) * 2  
            u_sims = u @ W.tanh().t()  
            v_sims = v @ W.tanh().t()  
            loss_ = (label_ - u_sims) ** 2
            loss_ += (label_ - v_sims) ** 2
            loss = (loss_ * label).max(1)[0]
        right = ((tlabel == label).float().mean(1) == 1).float()
        for i in range(len(loss)):
            sample_losses.append((ind[i].item(), loss[i].item(), right[i].item()))
    sample_losses_sorted = sorted(sample_losses, key=lambda x: x[0])
    sorted_losses = [item[1] for item in sample_losses_sorted]
    sorted_losses = np.array(sorted_losses)
    sorted_losses = (sorted_losses - sorted_losses.min() + 1e-8) / (sorted_losses.max() - sorted_losses.min() + 1e-8)
    sorted_losses = sorted_losses.reshape(-1, 1)
    save_path = f'/home/wangla/My_method/final/pdf/loss{epoch}_dataset{config["dataset"]}_bit{config["bit_len"]}_noiserate{config["noise_rate"]}.pdf'
    labels = np.array([item[2] for item in sample_losses_sorted])

    bmm = BetaMixture1D(max_iters=10)
    bmm.fit(sorted_losses)
    means = bmm.alphas / (bmm.alphas + bmm.betas)
    low_idx = int(means.argmin())
    post0 = bmm.posterior(sorted_losses, 0)  
    post1 = bmm.posterior(sorted_losses, 1)  
    post = np.stack([post0, post1], axis=1)
    prob = post[:, low_idx]
    if epoch + 1 >= 20:
        pred = split_prob(prob, Threshold)
    else:
        pred = split_prob(prob, 0)
    clean_index = np.where(labels == 1)[0]
    smaller_mean_indices = [i for i, p in enumerate(pred) if p == 1]
    true_positives = set(smaller_mean_indices).intersection(clean_index)
    false_positives = set(smaller_mean_indices).difference(clean_index)
    precision = len(true_positives) / (len(true_positives) + len(false_positives))
    return sorted_losses, torch.Tensor(pred), precision, prob


def train(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train  
    net = ImgModule(y_dim=4096, bit=bit, n_class=n_class, hiden_layer=3).to('cuda')
    txt_net = TxtModule(y_dim=tag_len, n_class=n_class, bit=bit, hiden_layer=2).to('cuda')
    W = torch.Tensor(n_class, bit_len)
    W = torch.nn.init.orthogonal_(W, gain=1)
    W = torch.tensor(W, requires_grad=True).cuda()
    W = torch.nn.Parameter(W)
    net.register_parameter('W', W)  
    get_grad_params = lambda model: [x for x in model.parameters() if x.requires_grad]
    params_dnet = get_grad_params(net)
    optimizer = config["optimizer"]["type"](params_dnet, **(config["optimizer"]["optim_params"]))
    txt_optimizer = config["txt_optimizer"]["type"](txt_net.parameters(), **(config["txt_optimizer"]["optim_params"]))
    
    criterion_clean = Robust_Loss_clean(config, bit)
    criterion_noise = Robust_Loss_noise(config, bit)
    threshold_schedule = np.linspace(0, config["threshold_rate"], num_gradual)
    threshold_schedule = np.concatenate(
        (threshold_schedule, np.ones(config["epoch"] - num_gradual) * config["threshold_rate"]))
    i2t_mAP_list = []
    t2i_mAP_list = []
    epoch_list = []
    precision_list = []
    bestt2i = 0
    besti2t = 0
    os.makedirs('./checkpoint', exist_ok=True)
    for epoch in range(config["epoch"]):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")
        net.eval()
        txt_net.eval()
        net.train()
        txt_net.train()
        train_loss = 0
        if (epoch + 1) % 20 == 0:
            print("calculating test binary code......")
            img_tst_binary, img_tst_label = compute_img_result(test_loader, net, device=device)
            print("calculating dataset binary code.......")
            img_trn_binary, img_trn_label = compute_img_result(dataset_loader, net, device=device)
            txt_tst_binary, txt_tst_label = compute_tag_result(test_loader, txt_net, device=device)
            txt_trn_binary, txt_trn_label = compute_tag_result(dataset_loader, txt_net, device=device)
            print("calculating map.......")
            t2i_mAP = calc_map_k(img_trn_binary.numpy(), txt_tst_binary.numpy(), img_trn_label.numpy(),
                                 txt_tst_label.numpy())
            i2t_mAP = calc_map_k(txt_trn_binary.numpy(), img_tst_binary.numpy(), txt_trn_label.numpy(),
                                 img_tst_label.numpy())
            if t2i_mAP + i2t_mAP > bestt2i + besti2t:
                bestt2i = t2i_mAP
                besti2t = i2t_mAP
                torch.save({
                    'net_state_dict': net.state_dict(),
                    'txt_net_state_dict': txt_net.state_dict(),
                }, './checkpoint/best_model.pth')
            t2i_mAP_list.append(t2i_mAP.item())
            i2t_mAP_list.append(i2t_mAP.item())
            epoch_list.append(epoch)
            print("%s epoch:%d, bit:%d, dataset:%s,noise_rate:%.1f,t2i_mAP:%.3f, i2t_mAP:%.3f" % (
                config["info"], epoch + 1, bit, config["dataset"], config["noise_rate"], t2i_mAP, i2t_mAP))
        sorted_losses, pred, precision, prob = get_loss(net, txt_net, config, train_loader, config["threshold_rate"],
                                                        epoch, W)
        for image, tag, tlabel, label, ind in train_loader:
            ind_np = ind.cpu().numpy()
            current_pred = pred[ind]
            clean_samples = current_pred == 1
            noise_samples = current_pred == 0
            if clean_samples.sum() > 0:
                clean_samples = clean_samples.squeeze(-1)
                image_clean = image[clean_samples].cuda().float()
                tag_clean = tag[clean_samples].cuda().float()
                label_clean = label[clean_samples].cuda().float()

                optimizer.zero_grad()
                txt_optimizer.zero_grad()

                u_clean = net(image_clean)
                v_clean = txt_net(tag_clean)
                loss_clean = criterion_clean(u_clean, v_clean, label_clean, config)

                loss_clean.backward()
                optimizer.step()
                txt_optimizer.step()

                train_loss += loss_clean.item()

                clean_features = (u_clean + v_clean) / 2.0  
                clean_labels = label_clean 

            if noise_samples.sum() > 0 and clean_features is not None and len(clean_features) >= 3:
                noise_samples = noise_samples.squeeze(-1)
                image_noise = image[noise_samples].cuda().float()
                tag_noise = tag[noise_samples].cuda().float()
                label_noise = label[noise_samples].cuda().float()

                with torch.no_grad():

                    u_noise = net(image_noise)
                    v_noise = txt_net(tag_noise)
                    noise_features = (u_noise + v_noise) / 2.0 

                    noise_features_norm = F.normalize(noise_features, dim=1)
                    clean_features_norm = F.normalize(clean_features, dim=1)

                    similarity = torch.mm(noise_features_norm, clean_features_norm.t())

                    topk_sim, topk_idx = torch.topk(similarity, k=3, dim=1)

                    topk_labels = clean_labels[topk_idx]
                    corrected_label_noisy = torch.mean(topk_labels, dim=1)

                optimizer.zero_grad()
                txt_optimizer.zero_grad()

                u_noise = net(image_noise)
                v_noise = txt_net(tag_noise)
                loss_noise = criterion_noise(u_noise, v_noise, corrected_label_noisy, config)

                loss_noise.backward()
                optimizer.step()
                txt_optimizer.step()

                train_loss += loss_noise.item()
        train_loss = train_loss / len(train_loader)
        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))
        print("%s epoch:%d, bit:%d, dataset:%s,noise_rate:%.1f" % (
            config["info"], epoch + 1, bit, config["dataset"], config["noise_rate"]))
        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))


def test(config, bit, model_path='./checkpoint/best_model.pth'):
    device = config["device"]
    _, test_loader, dataset_loader, _, _, _ = get_data(config)
    net = ImgModule(y_dim=4096, bit=bit, n_class=n_class, hiden_layer=3).to('cuda')
    txt_net = TxtModule(y_dim=tag_len, n_class=n_class, bit=bit, hiden_layer=2).to('cuda')
    W = torch.Tensor(n_class, bit_len)
    W = torch.nn.init.orthogonal_(W, gain=1)
    W = torch.tensor(W, requires_grad=True).cuda()
    W = torch.nn.Parameter(W)
    net.register_parameter('W', W)
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['net_state_dict'])
    txt_net.load_state_dict(checkpoint['txt_net_state_dict'])
    net.eval()
    txt_net.eval()
    print("calculating test binary code......")
    print("calculating test binary code......")
    img_tst_binary, img_tst_label = compute_img_result(test_loader, net, device=device)
    print("calculating dataset binary code.......")
    img_trn_binary, img_trn_label = compute_img_result(dataset_loader, net, device=device)
    txt_tst_binary, txt_tst_label = compute_tag_result(test_loader, txt_net, device=device)
    txt_trn_binary, txt_trn_label = compute_tag_result(dataset_loader, txt_net, device=device)
    print("calculating map.......")
    t2i_mAP = calc_map_k(img_trn_binary.numpy(), txt_tst_binary.numpy(), img_trn_label.numpy(), txt_tst_label.numpy())
    i2t_mAP = calc_map_k(txt_trn_binary.numpy(), img_tst_binary.numpy(), txt_trn_label.numpy(), img_tst_label.numpy())
    print("Test Results: t2i_mAP: %.3f, i2t_mAP: %.3f" % (t2i_mAP, i2t_mAP))


if __name__ == "__main__":
    data_name_list = ['ms-coco']
    bit_list = [16]
    noise_rate_list = [0.8]
    for data_name in data_name_list:
        for rate in noise_rate_list:
            for bit in bit_list:
                bit_len = bit
                noise_rate = rate
                dataset = data_name
                if dataset == 'nuswide21':
                    n_class = 10
                    tag_len = 1000
                elif dataset == 'flickr':
                    n_class = 24
                    tag_len = 1386
                elif dataset == 'ms-coco':
                    n_class = 80
                    tag_len = 300
                elif dataset == 'iapr':
                    n_class = 255
                    tag_len = 2912
                config = get_config()
                print(config)
                train(config, bit)
                test(config, bit)
