import torch
import os
from datetime import datetime, timedelta, timezone
import logging
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
import cmocean

def prep_experiment(args):
    """
    Make output directories, setup logging, Tensorboard, snapshot code.
    """
    ckpt_path = args.ckpt_path
    tb_path = args.tb_path
    args.ckpt_exp_path = os.path.join(ckpt_path, args.exp)
    args.tb_exp_path = os.path.join(tb_path, args.exp)
    args.date_str = (datetime.now(timezone.utc) + timedelta(hours=8)).strftime('%Y_%m_%d_%H_%M_%S')
    os.makedirs(args.ckpt_exp_path, exist_ok=True)
    os.makedirs(args.tb_exp_path, exist_ok=True)
    save_log('log', args.ckpt_exp_path, args.date_str)
    open(os.path.join(args.ckpt_exp_path, args.date_str + '.txt'), 'w').write(str(args) + '\n\n')
    writer = SummaryWriter(log_dir=args.tb_exp_path, comment=args.date_str)
    return writer

def save_model(args, epoch, loss, model):
    # remove old models
    if epoch > 0:
        best_snapshot = 'best_epoch_{}_loss_{:.8f}.pth'.format(
            args.best_record['epoch'], args.best_record['loss'])
        best_snapshot = os.path.join(args.ckpt_exp_path, best_snapshot)
        assert os.path.exists(best_snapshot), 'cant find old snapshot {}'.format(best_snapshot)
        os.remove(best_snapshot)

    # save new best
    args.best_record['epoch'] = epoch
    args.best_record['loss'] = loss

    best_snapshot = 'best_epoch_{}_loss_{:.8f}.pth'.format(
        args.best_record['epoch'], args.best_record['loss'])
    best_snapshot = os.path.join(args.ckpt_exp_path, best_snapshot)

    torch.save({
        'state_dict': model.state_dict(),
        'epoch': epoch,
    }, best_snapshot)
    logging.info('save best models in ' + best_snapshot)

def save_log(prefix, output_dir, date_str):
    fmt = '%(asctime)s.%(msecs)03d %(message)s'
    date_fmt = '%m-%d %H:%M:%S'
    filename = os.path.join(output_dir, prefix + '_' + date_str + '.log')
    print("Logging :", filename)
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=date_fmt,
                        filename=filename, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)
    console.setFormatter(formatter)
    # 控制台输出且文件记录
    logging.getLogger('').addHandler(console)

# TODO: 可视化
def plot3x1(fields, pres, file_name):
    size = fields.shape
    x, y = np.linspace(0, size[1], size[1]), np.linspace(size[0], 0, size[0])
    x, y = np.meshgrid(x, y)

    plt.figure(figsize=(5, 8))
    plt.subplot(3, 1, 1)
    plt.contourf(x, y, fields, levels=100, cmap=cmocean.cm.balance)
    plt.colorbar()
    plt.subplot(3, 1, 2)
    plt.contourf(x, y, pres, levels=100, cmap=cmocean.cm.balance)
    plt.colorbar()
    plt.subplot(3, 1, 3)
    plt.contourf(x, y, pres - fields, levels=100, cmap=cmocean.cm.balance)
    plt.colorbar()
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()