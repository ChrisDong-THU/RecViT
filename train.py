import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import os
import sys
import argparse
import numpy as np
import logging

from model import RecFieldViT
from dataset import CylinderDataset
from utils import prep_experiment, save_model, plot3x1
from config import sensorset

parser = argparse.ArgumentParser(description='field reconstruction based on Vit')
# 训练参数
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
# 验证间隔与早停
parser.add_argument('--plot_freq', type=int, default=5)
parser.add_argument('--val_interval', type=int, default=1)
parser.add_argument('--val_patience', type=int, default=5)
# 训练记录与保存
parser.add_argument('--exp', type=str, default='ReFieldViT_cylinder2D',
                    help='experiment directory name')
parser.add_argument('--ckpt_path', type=str, default='logs/ckpt',
                    help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='logs/tb',
                    help='Save Tensorboard Path')
parser.add_argument('--snapshot', type=str, default=None)

args = parser.parse_args()

torch.cuda.set_device(0)
cudnn.benchmark = True


def train(args):
    tb_writer = prep_experiment(args)
    args.fig_path = args.ckpt_exp_path + '/figure'
    os.makedirs(args.fig_path, exist_ok=True)
    args.best_record = {'epoch': -1, 'loss': 1e10}

    # 实例化模型
    net = RecFieldViT().cuda()
    net.train()

    # 数据加载器
    train_dataset = CylinderDataset(index=[i for i in range(4000)], positions=sensorset[0])
    mean, std = train_dataset.mean, train_dataset.std
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    val_dataset = CylinderDataset(index=[i for i in range(4000, 5000)], positions=sensorset[0])
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4) # 不打乱，用最后同一张图像对比
    no_improve_val_epoch = 0 # 早停计数

    # 设置优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98) # l1适用
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30], gamma=0.5) # mse适用

    for epoch in range(args.epochs):
        train_loss, train_num = 0., 0.
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            pre = net(inputs)

            # 均方误差损失
            loss = F.mse_loss(labels, pre)
            # 平均绝对误差L1损失
            # loss = F.l1_loss(labels, pre)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.shape[0]
            train_num += inputs.shape[0]

        train_loss = train_loss / train_num
        # Tensorboard记录
        tb_writer.add_scalar('train_loss', train_loss, epoch)
        logging.info("Epoch: {}, Avg_loss: {}".format(epoch, train_loss))
        scheduler.step()  # 调整学习率

        if epoch % args.val_interval == 0:
            net.eval()
            val_loss, val_num = 0., 0.
            for i, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.cuda(), labels.cuda()
                with torch.no_grad():
                    pre = net(inputs)
                loss = F.mse_loss(labels, pre)
                # loss = F.l1_loss(labels, pre)

                val_loss += loss.item() * inputs.shape[0]
                val_num += inputs.shape[0]

            # Tensorboard记录
            val_loss = val_loss / val_num
            tb_writer.add_scalar('val_loss', val_loss, epoch)
            logging.info("Epoch: {}, Val_loss: {}".format(epoch, val_loss))
            if val_loss < args.best_record['loss']:
                save_model(args, epoch, val_loss, net)
                no_improve_val_epoch = 0
            else:
                no_improve_val_epoch += 1

            if no_improve_val_epoch >= args.val_patience:
                logging.info("Early stopping triggered at epoch {}".format(epoch))
                break

            net.train()

            # 绘制最后的验证集恢复情况
            if epoch % args.plot_freq == 0:
                # 反归一化
                labels = labels * std + mean
                pre = pre * std + mean
                plot3x1(labels[-1, 0, :, :].cpu().numpy(), pre[-1, 0, :, :].cpu().numpy(),
                        file_name=args.fig_path + f'/epoch{epoch}.png')


if __name__ == '__main__':
    try:
        train(args)
    except KeyboardInterrupt:
        print('Training stopped by user.')