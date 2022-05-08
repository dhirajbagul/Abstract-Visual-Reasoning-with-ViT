import os
import numpy as np
import argparse
from tqdm import tqdm
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

#from utility import dataset, ToTensor
from utility import RAVENdataset, ToTensor
from utility import logwrapper, plotwrapper
import models
torch.cuda.empty_cache()
parser = argparse.ArgumentParser(description='baseline')
parser.add_argument('--model', type=str, default='WReN')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--config', type=str, default='all')
parser.add_argument('--load_workers', type=int, default=16)
parser.add_argument('--path', type=str, default='/filer/tmp1/dvb30/Dataset - Pritish/')
parser.add_argument('--save', type=str, default='./results/checkpoint/')
parser.add_argument('--log', type=str, default='./results/log/')
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--img_size', type=int, default=80)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--epsilon', type=float, default=1e-8)
parser.add_argument('--meta_beta', type=float, default=10.0)
parser.add_argument('--tag', type=int, default=1)


args = parser.parse_args()

args.save = '/common/users/dvb30/Results/Dataset - pritish/{}/{}/checkpoint/'.format(args.config, args.model)
args.log = '/common/users/dvb30/Results/Dataset - pritish/{}/{}/log/'.format(args.config, args.model)
args.csv = '/common/users/dvb30/Results/Dataset - pritish/{}/{}/eval.csv'.format(args.config, args.model)


args.cuda = torch.cuda.is_available()
torch.cuda.set_device(args.device)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.exists(args.log):
    os.makedirs(args.log)

if 'all' in args.config:
    args.train_figure_configurations = [0, 1, 2, 3]
    args.val_figure_configurations = args.train_figure_configurations
    args.test_figure_configurations = [0, 1, 2, 3]

if 'center_single' in args.config:
    args.train_figure_configurations = [0]
    args.val_figure_configurations = args.train_figure_configurations
    args.test_figure_configurations = [0]

if 'distribute_four' in args.config:
    args.train_figure_configurations = [1]
    args.val_figure_configurations = args.train_figure_configurations
    args.test_figure_configurations = [1]

if 'in_distribute_four_out_center_single' in args.config:
    args.train_figure_configurations = [2]
    args.val_figure_configurations = args.train_figure_configurations
    args.test_figure_configurations = [2]

if 'left_center_single_right_center_single' in args.config:
    args.train_figure_configurations = [3]
    args.val_figure_configurations = args.train_figure_configurations
    args.test_figure_configurations = [3]


train = RAVENdataset(args.path, "train", args.train_figure_configurations, args.img_size, transform=transforms.Compose([ToTensor()]), shuffle = True)
valid = RAVENdataset(args.path, "val", args.val_figure_configurations, args.img_size, transform=transforms.Compose([ToTensor()]))
test = RAVENdataset(args.path, "test", args.test_figure_configurations, args.img_size, transform=transforms.Compose([ToTensor()]))


#train = dataset(args.path, "train", args.img_size, transform=transforms.Compose([ToTensor()]))
#valid = dataset(args.path, "val", args.img_size, transform=transforms.Compose([ToTensor()]))
#test = dataset(args.path, "test", args.img_size, transform=transforms.Compose([ToTensor()]))

trainloader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=16)
validloader = DataLoader(valid, batch_size=args.batch_size, shuffle=False, num_workers=16)
testloader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=16)

print ('Dataset: I-RAVEN')
print ('Train/Validation/Test:{0}/{1}/{2}'.format(len(train), len(valid), len(test)))
print ('Image size:', args.img_size)

model = None
current_epoch = 0
if args.model == 'WReN':
    model = models.WReN(args)
elif args.model == 'CNN_MLP':
    model = models.CNN_MLP(args)
elif args.model == 'Resnet50_MLP':
    model = models.Resnet50_MLP(args)
elif args.model == 'LSTM':
    model = models.CNN_LSTM(args)
elif args.model == 'ViT_MLP':
    model = models.ViT_MLP(args)
elif args.model == 'ViT_WReN':
    model = models.ViT_WReN(args)
elif args.model == 'ViT_LSTM':
    model = models.ViT_LSTM(args)

if args.load_model == True:
    model.load_model('/freespace/local/dvb30/Machin Learning/results/checkpoint/', 199)
    model.load_state_dict(model.state_dict)
    current_epoch = 199
#    model.eval()
#    test_acc = test(199)




if args.cuda:
    model = model.cuda()

log = logwrapper(args.log)

def train(epoch):
    model.train()
    train_loss = 0
    accuracy = 0

    loss_all = 0.0
    acc_all = 0.0
    counter = 0
    train_iter = iter(trainloader)
    for _ in tqdm(range(len(train_iter))):
        counter += 1
        image, target, meta_target = next(train_iter)
        if args.cuda:
            image = image.cuda()
            target = target.cuda()
            meta_target = meta_target.cuda()
        loss, acc = model.train_(image, target, meta_target)
        # print('Train: Epoch:{}, Batch:{}, Loss:{:.6f}, Acc:{:.4f}.'.format(epoch, batch_idx, loss, acc))
        loss_all += loss
        acc_all += acc
    if counter > 0:
        print("Avg Training Loss: {:.6f}, Acc: {:.4f}".format(loss_all/float(counter), acc_all/float(counter)))
    return loss_all/float(counter), acc_all/float(counter)

def validate(epoch):
    model.eval()
    val_loss = 0
    accuracy = 0

    loss_all = 0.0
    acc_all = 0.0
    loss_all = 0.0
    counter = 0
    valid_iter = iter(validloader)
    for _ in tqdm(range(len(valid_iter))):
        counter += 1
        image, target, meta_target = next(valid_iter)

        if args.cuda:
            image = image.cuda()
            target = target.cuda()
            meta_target = meta_target.cuda()
        loss, acc = model.validate_(image, target, meta_target)
        # print('Validate: Epoch:{}, Batch:{}, Loss:{:.6f}, Acc:{:.4f}.'.format(epoch, batch_idx, loss, acc))
        loss_all += loss
        acc_all += acc
        loss_all += loss
    if counter > 0:
        print("Total Validation Loss: {:.6f}, Acc: {:.4f}".format(loss_all/float(counter), acc_all/float(counter)))
    return loss_all/float(counter), acc_all/float(counter)

def test(epoch):
    model.eval()
    accuracy = 0

    acc_all = 0.0
    counter = 0
    test_iter = iter(testloader)
    for _ in tqdm(range(len(test_iter))):
        counter += 1
        image, target, meta_target = next(test_iter)
        if args.cuda:
            image = image.cuda()
            target = target.cuda()
            meta_target = meta_target.cuda()
        acc = model.test_(image, target, meta_target)
        # print('Test: Epoch:{}, Batch:{}, Acc:{:.4f}.'.format(epoch, batch_idx, acc))
        acc_all += acc
    if counter > 0:
        print("Total Testing Acc: {:.4f}".format(acc_all / float(counter)))
    return acc_all/float(counter)

def main():
    for epoch in range(current_epoch, args.epochs+1):
        print("\n\n\t\tEpoch: {}\n".format(epoch))
        train_loss, train_acc = train(epoch)
        val_loss, val_acc = validate(epoch)
        test_acc = test(epoch)
        row = [epoch, train_acc, train_loss, val_acc, val_loss, test_acc]
        with open(args.csv, 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        if epoch % 49 == 0:
            model.save_model(args.save, epoch)
        loss = {'train':train_loss, 'val':val_loss}
        acc = {'train':train_acc, 'val':val_acc, 'test':test_acc}
        log.write_scalars('Loss', loss, epoch)
        log.write_scalars('Accuracy', acc, epoch)

if __name__ == '__main__':
    main()
