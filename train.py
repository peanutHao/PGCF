import torch
from torch.autograd import Variable
import os
import torch.nn as nn
import argparse
from datetime import datetime
from lib.model import PGCF
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging


from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir ="//a/train/")


device_ids = [0,1]
torch.cuda.set_device(device_ids[0])
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def test(model, path, dataset):
    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    DSC = 0.0
    IOU = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        # image = image.to(device)
        image = image.to(device_ids[0])

        res, _, _ = model(image)

        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        union = input_flat + target_flat - intersection
        iou = (intersection.sum() + smooth) / (union.sum() + smooth)
        iou = '{:.4f}'.format(iou)
        iou = float(iou)
        DSC = DSC + dice
        IOU = IOU + iou

    return DSC / num1, IOU / num1


def train(train_loader, model, optimizer, epoch, test_path):
    model.train()
    global best
    size_rates = [0.75, 1, 1.25]
    loss_P1_record = AvgMeter()
    # loss_P2_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            iamges = Variable(images).to(device_ids[0])

            gts = Variable(gts).cuda()
            gts = Variable(gts).to(device_ids[0])

            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            P1, P2, P3= model(images)
            if P1.size()!=gts.size():
                P1 = F.interpolate(P1, size=(trainsize, trainsize), mode='bilinear', align_corners=False)
            gts1 = F.interpolate(gts, size=(P2.shape[3], P2.shape[2]), mode='bilinear', align_corners=False)
            gts2 = F.interpolate(gts, size=(P3.shape[3], P3.shape[2]), mode='bilinear', align_corners=False)

            loss_P1 = structure_loss(P1, gts)
            loss_P2 = structure_loss(P2, gts1)
            loss_P3 = structure_loss(P3, gts2)
            loss = (loss_P1 + loss_P2 + loss_P3)/3.0
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss
            if rate == 1:
                loss_P1_record.update(loss_P1.data, opt.batchsize)
                if i == total_step:
                    writer.add_scalars("loss", {
                        "P1": loss_P1,
                        "P2": loss_P2,
                        "P3": loss_P3,
                        "total": loss
                    }, epoch)

        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_P1_record.show()))

    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    global dict_plot

    test1path = '/home/data/qianhao/Endoscope-WL/TestDataset'
    if epoch % 1 == 0:
        dataset1 = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
        print('##################################epoch', epoch)

        for j in range(0, 5):
            dataset = dataset1[j]
            dataset_dice, dataset_iou = test(model, test1path, dataset)
            writer.add_scalars("dice", {
                dataset: dataset_dice,
            }, epoch)
            writer.add_scalars("iou", {
                dataset: dataset_iou,
            }, epoch)
            if dataset_dice > best[j] and dataset_iou > best1[j]:
                best[j] = dataset_dice
                best1[j] = dataset_iou
                print('##################################bestdice bestiou', dataset, dataset_dice, dataset_iou)
                loggers_and_handlers[j][0].info(
                    '{}############################################################################bestdice:{} bestiou:{}'.format(
                        epoch, dataset_dice, dataset_iou))
                torch.save(model.state_dict(),
                           save_path + 'train_doublebest'+dataset+'.pth')
            elif dataset_iou > best1[j]:
                best1[j] = dataset_iou
                print('##################################dice bestiou', dataset, dataset_dice, dataset_iou)
                loggers_and_handlers[j][0].info(
                    '{}###########################################################################dice:{} bestiou:{} '.format(
                        epoch, dataset_dice, dataset_iou))
                torch.save(model.state_dict(),
                           save_path + 'train_bestiou'+dataset+'.pth')
            elif dataset_dice > best[j]:
                best[j] = dataset_dice
                print('##################################bestdice iou', dataset, dataset_dice, dataset_iou)
                loggers_and_handlers[j][0].info(
                    '{}############################################################################bestdice:{} iou:{} '.format(
                        epoch, dataset_dice, dataset_iou))
                torch.save(model.state_dict(),
                           save_path + 'train_bestdice'+dataset+'.pth')
            else:
                print('##################################dice iou', dataset, dataset_dice, dataset_iou)
                loggers_and_handlers[j][0].info(
                    '{}############################################################################dice:{} iou:{}'.format(
                        epoch, dataset_dice, dataset_iou))


if __name__ == '__main__':
    import pynvml
    import time

    a = 10000
    while a > 1000:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_ids[0])
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        a = meminfo.used / 1024 ** 2
        print(a)
        time.sleep(10)
    dict_plot = {'CVC-300': [], 'CVC-ClinicDB': [], 'Kvasir': [], 'CVC-ColonDB': [], 'ETIS-LaribPolypDB': [],
                 'test': []}
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
    ##################model_name#############################
    model_name = 'train'
    ###############################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=200, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=24, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')


    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='/home/data/qianhao/Endoscope-WL/TrainDataset/Mix-Kvasir-ClinicDB',
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default='/home/data/qianhao/Endoscope-WL/TestDataset',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='/home/users/qianhao/PycharmProjects/PGCF/model/')

    opt = parser.parse_args()


    # ---- build models ----
    model = PGCF().to(device_ids[0])
    model = nn.DataParallel(model, device_ids=device_ids).to(device_ids[0])

    best = [0, 0, 0, 0, 0]
    best1 = [0, 0, 0, 0, 0]

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    d = [ 'CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
    loggers_and_handlers = []
    for i in range(5):
        # 创建 logger
        logger = logging.getLogger(f'logger_{d[i]}')
        logger.setLevel(logging.INFO)

        # 创建文件处理器，将日志写入到对应的文件
        file_handler = logging.FileHandler(f'//model/log/train_{d[i]}.log', mode='a')
        file_handler.setLevel(logging.INFO)

        # 定义日志格式
        formatter = logging.Formatter(f'[%(asctime)s-%(filename)s-%(levelname)s:{i}:%(message)s]',
                                      datefmt='%Y-%m-%d %I:%M:%S %p')
        file_handler.setFormatter(formatter)

        # 将处理器添加到 logger
        logger.addHandler(file_handler)

        # 将 logger 和处理器元组添加到列表
        loggers_and_handlers.append((logger, file_handler))
        loggers_and_handlers[i][0].info('##########################optimizer:{}'.format(optimizer))

    print(optimizer)
    logging.info('##############################################################################optimizer:{}'.format(optimizer))
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(0, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch+1, 0.1, 200)
        train(train_loader, model, optimizer, epoch+1, opt.test_path)
writer.close()

