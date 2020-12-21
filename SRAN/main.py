import os
import numpy as np
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import visdom

from utility import PGMdataset, RAVENdataset, ToTensor
from sran import SRAN

parser = argparse.ArgumentParser(description='our_model')
parser.add_argument('--model', type=str, default='SRAN')
parser.add_argument('--dataset', type=str, default='I-RAVEN', choices=['PGM', 'I-RAVEN'])
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--load_workers', type=int, default=16)
parser.add_argument('--resume', type=str, default='')
#parser.add_argument('--dataset_path', type=str, default='/media/dsg3/datasets/PGM')
parser.add_argument('--dataset_path', type=str, default='/media/dsg3/datasets/I-RAVEN')
parser.add_argument('--save', type=str, default='/media/dsg3/hs')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--epsilon', type=float, default=1e-8)
parser.add_argument('--meta_beta', type=float, default=0.0)
parser.add_argument('--visdom', default=True, help='Use visdom for visualization')
parser.add_argument('--cuda', default=True )
parser.add_argument('--debug', default=False)

args = parser.parse_args()
torch.cuda.manual_seed(args.seed)
args.save += '/' + args.dataset+'/'
start_time = time.strftime ('%Y-%m-%d_%H-%M-%S') 
if args.debug:
    args.save += args.dataset + '/'
    args.visdom = False
else:
    args.save += args.dataset +'_' + start_time + '/'
if not os.path.exists(args.save):
    os.makedirs(args.save)

if args.dataset == 'PGM':
    train = PGMdataset(args.dataset_path, "train", args.img_size, transform=transforms.Compose([ToTensor()]), shuffle = True)
    valid = PGMdataset(args.dataset_path, "val", args.img_size, transform=transforms.Compose([ToTensor()]))
    test = PGMdataset(args.dataset_path, "test", args.img_size, transform=transforms.Compose([ToTensor()]))
elif args.dataset == 'I-RAVEN':      
    args.train_figure_configurations = [0,1,2,3,4,5,6]
    args.val_figure_configurations = args.train_figure_configurations
    args.test_figure_configurations = [0,1,2,3,4,5,6]
    train = RAVENdataset(args.dataset_path, "train", args.train_figure_configurations, args.img_size, transform=transforms.Compose([ToTensor()]), shuffle = True)
    valid = RAVENdataset(args.dataset_path, "val", args.val_figure_configurations, args.img_size, transform=transforms.Compose([ToTensor()]))
    test = RAVENdataset(args.dataset_path, "test", args.test_figure_configurations, args.img_size, transform=transforms.Compose([ToTensor()]))

trainloader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.load_workers)
validloader = DataLoader(valid, batch_size=args.batch_size, shuffle=False, num_workers=args.load_workers)
testloader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.load_workers)

print ('Dataset:', args.dataset)
print ('Train/Validation/Test:{0}/{1}/{2}'.format(len(train), len(valid), len(test)))
print ('Image size:', args.img_size)

if args.model == 'SRAN':
    model = SRAN(args)
start_epoch = 0
if args.resume:
    args.resume_epoch = 78
    model.load_model(args.resume, args.resume_epoch)
    print('Loaded model')
    start_epoch = args.resume_epoch+1
with open(os.path.join(args.save, 'results.log'), 'w') as f:
    for key, value in vars(args).items():
        f.write('{0}: {1}\n'.format(key, value))
    f.write('--------------------------------------------------\n')

pmodel = torch.nn.DataParallel(model)
torch.backends.cudnn.benchmark = True
pmodel = pmodel.cuda()

if args.visdom:
    viz = visdom.Visdom(port = 9527, env = args.dataset)

def train(epoch):
    model.train()
    train_loss = 0
    accuracy = 0
    loss_all = 0.0
    acc_all = 0.0
    counter = 0
    for batch_idx, (image, target, meta_target) in enumerate(trainloader):
        counter += 1
        if args.cuda:
            image = image.cuda()
            target = target.cuda()
            meta_target = meta_target.cuda()  
        model.optimizer.zero_grad()
        output = pmodel(image)
        loss = model.compute_loss(output, target, meta_target)
        loss.backward()
        model.optimizer.step()
        pred = output[0].data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        accuracy = correct * 100. / target.size()[0]
        loss, acc = loss.item(), accuracy
        print('Train: Epoch:{}, Batch:{}, Loss:{:.6f}, Acc:{:.4f}.'.format(epoch, batch_idx, loss, acc))
        loss_all += loss
        acc_all += acc
    if counter > 0:
        print("Avg Training Loss: {:.6f}".format(loss_all/float(counter)))
    return loss_all/float(counter), acc_all/float(counter)

def validate(epoch):
    model.eval()
    accuracy = 0
    acc_all = 0.0
    counter = 0
    with torch.no_grad():
        for batch_idx, (image, target, meta_target) in enumerate(validloader):
            counter += 1
            if args.cuda:
                image = image.cuda()
                target = target.cuda()
                meta_target = meta_target.cuda()    
            output = pmodel(image)           
            pred = output[0].data.max(1)[1]
            correct = pred.eq(target.data).cpu().sum().numpy()
            accuracy = correct * 100. / target.size()[0]          
            acc = accuracy          
            acc_all += acc
    if counter > 0:
        print("Total Validation Acc: {:.4f}".format(acc_all/float(counter)))
    return acc_all/float(counter)

def test(epoch):
    model.eval()
    accuracy = 0
    acc_all = 0.0
    counter = 0
    with torch.no_grad():
        for batch_idx, (image, target, meta_target) in enumerate(testloader):
            counter += 1
            if args.cuda:
                image = image.cuda()
                target = target.cuda()
                meta_target = meta_target.cuda()   
            output = pmodel(image)
            pred = output[0].data.max(1)[1]
            correct = pred.eq(target.data).cpu().sum().numpy()
            accuracy = correct * 100. / target.size()[0]   
            acc = accuracy
            acc_all += acc
    if counter > 0:
        print("Total Testing Acc: {:.4f}".format(acc_all / float(counter)))
    return acc_all/float(counter)

def main():

    if args.visdom:        
        vis_title =  args.dataset + ' ' + start_time
        vis_legend = ['Train Acc', 'Val Acc', 'Test Acc']
        epoch_plot = create_vis_plot('Epoch', 'Acc', vis_title, vis_legend)
    for epoch in range(start_epoch, args.epochs):
        avg_train_loss, avg_train_acc = train(epoch)
        avg_acc = validate(epoch)
        avg_test_acc = test(epoch)
        model.save_model(args.save, epoch)
        with open(os.path.join(args.save, 'results.log'), 'a') as f:
            f.write('Epoch {}, Training loss: {:.6f}, Validation Acc: {:.4f}, Testing Acc: {:.4f}\n'.format(epoch, avg_train_loss, avg_acc, avg_test_acc))
        if args.visdom:
            viz.line(
                X=torch.ones((1, 3)) * epoch,
                Y=torch.Tensor([avg_train_acc, avg_acc, avg_test_acc]).unsqueeze(0),
                win=epoch_plot,
                update='append'
            )
def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)),
        Y=torch.zeros((1, 3)),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )
if __name__ == '__main__':
    main()
