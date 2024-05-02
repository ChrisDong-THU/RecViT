import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import argparse

from model import RecFieldViT
from dataset import CylinderDataset
from utils import plot3x1
from config import sensorset

parser = argparse.ArgumentParser(description='field reconstruction based on Vit')
parser.add_argument('--model', type=str, default='logs/ckpt/ReFieldViT_cylinder2D_l1_0/best_epoch_155_loss_0.02057363.pth',
                    help='path to the model')
parser.add_argument('--index', type=str, default='1000-2000',
                    help='range of indexes, formatted as start-end')
args = parser.parse_args()

def eval(args):
    # 准备数据
    start, end = map(int, args.index.split('-'))
    eval_dataset = CylinderDataset(index=[i for i in range(start, end)], positions=sensorset[0])
    eval_loader = DataLoader(eval_dataset, batch_size=32, num_workers=0)

    # 加载模型
    net = RecFieldViT().cuda()
    net.load_state_dict(torch.load(args.model)['state_dict'])
    print(f'>>> load model {args.model} ...')

    net.eval()
    # 均方误差，测试样本数
    eval_mse, eval_num = 0.0, 0.0
    for i, (inputs, labels) in enumerate(eval_loader):
        inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            pre = net(inputs)
        eval_mse += F.mse_loss(labels, pre).item() * inputs.shape[0]
        eval_num += inputs.shape[0]

    eval_mse = eval_mse / eval_num
    print(f'>>> eval num: {eval_num}, eval mse: {eval_mse}')
    plot3x1(labels[-1, 0, :].cpu().numpy(), pre[-1, 0, :].cpu().numpy(), f'./eval_{end}.png')


if __name__ == '__main__':
    try:
        eval(args)
    except KeyboardInterrupt:
        print('eval stopprd by user.')