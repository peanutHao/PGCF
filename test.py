import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
from lib.model import PGCF
from utils.dataloader import test_dataset
import torch.nn as nn



device_ids = [0,1]
device = torch.device('cuda:0')

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
path = '/home/users/qianhao/PycharmProjects/PGCF/model//train_doublebestCVC-ClinicDB.pth'
for _data_name in ['CVC-ClinicDB']:

    data_path = '/home/data/qianhao/Endoscope-WL/TestDataset/{}/'.format(_data_name)
    save_path = '/home/users/qianhao/PycharmProjects/PGCF/result/{}/'.format(_data_name)
    opt = parser.parse_args()
    model = PGCF().to(device_ids[0])
    model = nn.DataParallel(model, device_ids=device_ids).to(device_ids[0])
    model.load_state_dict(torch.load(path,map_location=device))
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res,_,_ = model(image)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path+name, res*255)