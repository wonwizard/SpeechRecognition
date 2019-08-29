import os
import time
import random
import argparse
import logging
import numpy as np
import torch
from torch import nn, autograd
from torch.autograd import Variable
import torch.nn.functional as F
import kaldi_io
from seq2seq.seq2seq import Seq2seq
import tensorboard_logger as tb
from DataLoader import SequentialLoader, TokenAcc

parser = argparse.ArgumentParser(description='PyTorch Seq2seq-Attention Acoustic Model on TIMIT.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--bi', default=False, action='store_true', 
                    help='whether use bidirectional lstm')
parser.add_argument('--sr', type=float, default=0.4,
                    help='decoder sampleing rate, only for training')
parser.add_argument('--noise', default=False, action='store_true',
                    help='add Gaussian weigth noise')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--stdout', default=False, action='store_true', help='log in terminal')
parser.add_argument('--out', type=str, default='exp/att_lr1e-3',
                    help='path to save the final model')
parser.add_argument('--cuda', default=True, action='store_false')
parser.add_argument('--init', type=str, default='',
                    help='Initial am parameters')
parser.add_argument('--gradclip', default=False, action='store_true')
parser.add_argument('--schedule', default=False, action='store_true')
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, 'args'), 'w') as f:
    f.write(str(args))
if args.stdout: logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%H:%M:%S', level=logging.INFO)
else: logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%H:%M:%S', filename=os.path.join(args.out, 'train.log'), level=logging.INFO)
tb.configure(args.out)
random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed_all(1024)

model = Seq2seq(123, 63, 250, 3, args.dropout, args.bi, args.sr)
if args.init: model.load_state_dict(torch.load(args.init))
else: 
    for param in model.parameters(): torch.nn.init.uniform(param, -0.1, 0.1)
if args.cuda: model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9)

# data set
trainset = SequentialLoader('train', args.batch_size, True)
devset = SequentialLoader('dev', args.batch_size, True)

tri = cvi = 0
def eval():
    global cvi
    losses = []
    for xs, ys, xlen, ylen in devset:
        x = Variable(torch.FloatTensor(xs), volatile=True).cuda()
        y = Variable(torch.LongTensor(ys), volatile=True).cuda()
        model.eval()
        loss = model(x, y)
        loss = float(loss.data) # batch size
        losses.append(loss)
        tb.log_value('cv_loss', loss, cvi)
        cvi += 1
    return sum(losses) / len(devset)

def train():
    def adjust_learning_rate(optimizer, lr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        # lr = args.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    def add_noise(x):
        dim = x.shape[-1]
        noise = torch.normal(torch.zeros(dim), 0.075)
        if x.is_cuda: noise = noise.cuda()
        x.data += noise

    global tri
    prev_loss = 1000
    best_model = None
    lr = args.lr
    for epoch in range(1, args.epochs):
        totloss = 0; losses = []
        start_time = time.time()
        for i, (xs, ys, xlen, ylen) in enumerate(trainset):
            x = Variable(torch.FloatTensor(xs)).cuda()
            if args.noise: add_noise(x)
            y = Variable(torch.LongTensor(ys)).cuda()
            model.train()
            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            loss = float(loss.data) # batch size
            totloss += loss; losses.append(loss)
            if args.gradclip: grad_norm = nn.utils.clip_grad_norm(model.parameters(), 200)
            optimizer.step()

            tb.log_value('train_loss', loss, tri)
            if args.gradclip: tb.log_value('train_grad_norm', grad_norm, tri)
            tri += 1

            if i % args.log_interval == 0 and i > 0:
                loss = totloss / args.batch_size / args.log_interval
                logging.info('[Epoch %d Batch %d] loss %.2f'%(epoch, i, loss))
                totloss = 0

        losses = sum(losses) / len(trainset)
        val_l = eval()
        logging.info('[Epoch %d] time cost %.2fs, train loss %.2f; cv loss %.2f; lr %.3e'%(
            epoch, time.time()-start_time, losses, val_l, lr
        ))

        if val_l < prev_loss:
            prev_loss = val_l
            best_model = '{}/params_epoch{:02d}_tr{:.2f}_cv{:.2f}'.format(args.out, epoch, losses, val_l)
            torch.save(model.state_dict(), best_model)
        else:
            torch.save(model.state_dict(), '{}/params_epoch{:02d}_tr{:.2f}_cv{:.2f}_rejected'.format(args.out, epoch, losses, val_l))
            model.load_state_dict(torch.load(best_model))
            if args.cuda: model.cuda()
            if args.schedule:
                lr /= 2
                adjust_learning_rate(optimizer, lr)

if __name__ == '__main__':
    train()
