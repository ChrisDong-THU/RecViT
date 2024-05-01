import torch
import torch.nn.functional as F

import argparse
import pickle

from model import ReFieldVit
from dataset import CylinderDataset
from utils import plot3x1
from config import sensorset

parser = argparse.ArgumentParser(description='field reconstruction based on Vit')
parser.add_argument('--model', type=str, default='logs/ckpt/ReFieldViT_cylinder2D_l1_0/best_epoch_155_loss_0.02057363.pth',
                    help='path to the model')
parser.add_argument('--index', type=int, default=1000)
parser.add_argument('-s', '--sensorset', type=int, default=1)
args = parser.parse_args()


def eval(args):
    # 准备数据
    dataset = CylinderDataset(index=[i for i in range(5000)], positions=sensorset[args.sensorset - 1])
    inputs, labels = dataset[args.index]

    # 补上批次维度
    inputs = inputs.unsqueeze(0)
    labels = labels.unsqueeze(0)

    # 加载模型
    net = ReFieldVit().cuda()
    net.load_state_dict(torch.load(args.model)['state_dict'])
    print(f'>>> load model {args.model} ...')
    net.eval()

    inputs, labels = inputs.cuda(), labels.cuda()
    with torch.no_grad():
        pre = net(inputs)
    eval_mse = F.mse_loss(labels, pre)

    print(f'>>> eval mse: {eval_mse.item()}')
    plot3x1(labels[0, 0, :].cpu().numpy(), pre[0, 0, :].cpu().numpy(), f'./mse_eval_{args.index}_s{args.sensorset}.png')


if __name__ == '__main__':
    eval(args)
