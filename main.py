'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

# [wandb] NEW
import math
import time
import wandb


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')

# [wandb] NEW: enable wandb
parser.add_argument('--wandb', action='store_true', help='enable Weights & Biases logging')
parser.add_argument('--project', default='cifar10', type=str, help='wandb project name')
parser.add_argument('--entity', default=None, type=str, help='wandb entity (team/user)')
parser.add_argument('--run-name', default=None, type=str, help='wandb run name')
parser.add_argument('--tags', nargs='*', default=None, help='wandb tags')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
arch_name = net.__class__.__name__  # [wandb] NEW: for logging

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth', map_location=device)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# [wandb] NEW: 초기화
if args.wandb:
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        tags=args.tags,
        config={
            'dataset': 'CIFAR10',
            'arch': arch_name,
            'lr': args.lr,
            'optimizer': 'SGD',
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'scheduler': 'CosineAnnealingLR',
            'epochs': 200,
            'batch_size_train': trainloader.batch_size,
            'batch_size_test': testloader.batch_size,
            'num_workers': 2,
            'augmentations': ['RandomCrop(32,pad=4)','RandomHorizontalFlip','Normalize'],
        }
    )
    # print the config to console
    wandb.watch(net, log='gradients', log_freq=100)

global_step = 0  # [wandb] NEW: batch step count


# Training
def train(epoch):
    global global_step
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0.0
    correct = 0
    total = 0
    t0 = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # [wandb] per-batch logging
        if args.wandb:
            wandb.log({
                'batch/train_loss': loss.item(),
                'batch/train_acc': 100.0 * predicted.eq(targets).sum().item() / targets.size(0),
                'lr': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
            }, step=global_step)
        global_step += 1

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # [wandb] per-epoch logging
    if args.wandb:
        wandb.log({
            'train/loss': train_loss / (batch_idx + 1),
            'train/acc': 100. * correct / total,
            'train/throughput(img/s)': total / (time.time() - t0),
            'epoch': epoch,
        }, step=global_step)


def test(epoch):
    global best_acc, global_step
    net.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100. * correct / total

    # [wandb] per-epoch logging
    if args.wandb:
        wandb.log({
            'test/loss': test_loss / (batch_idx + 1),
            'test/acc': acc,
            'test/throughput(img/s)': total / (time.time() - t0),
            'epoch': epoch,
            'lr': optimizer.param_groups[0]['lr'],
        }, step=global_step)

    # Save checkpoint.
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

        # [wandb] checkpoint logging
        if args.wandb:
            art = wandb.Artifact('best-checkpoint', type='model',
                                 metadata={'acc': acc, 'epoch': epoch})
            art.add_file('./checkpoint/ckpt.pth')
            wandb.log_artifact(art)


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()

# [wandb] NEW: finish the run
if args.wandb:
    wandb.finish()
