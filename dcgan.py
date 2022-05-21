import torch
import torch.nn as nn
import torch.nn.functional as F

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)
        #self.dense = nn.Linear(9, 1)
        self.leaky_relu = nn.LeakyReLU()

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = self.conv1(input)
        x = self.leaky_relu(self.conv2_bn(self.conv2(x)))
        x = self.leaky_relu(self.conv3_bn(self.conv3(x)))
        x = self.leaky_relu(self.conv4_bn(self.conv4(x)))
        x = self.conv5(x)#.sigmoid()

        return x

# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))

        return x

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

import torch.optim as optim
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler

g_lr=2e-5
d_lr=1e-5
# network
G = generator(128)
D = discriminator(128)
# G.weight_init(mean=0.0, std=0.02)
# D.weight_init(mean=0.0, std=0.02)
G = G.cuda()
D = D.cuda()
ema_G = deepcopy(G)

# Binary Cross Entropy loss
BCE_loss = nn.BCEWithLogitsLoss()

# Adam optimizer
G_optimizer = optim.SGD(G.parameters(), lr=g_lr)
D_optimizer = optim.SGD(D.parameters(), lr=d_lr)

scaler_g = GradScaler(enabled=True)
scaler_d = GradScaler(enabled=True)

ema = EMA(0.995)

from torch.utils.data import Dataset, DataLoader, IterableDataset
import glob
from torchvision.io import read_image
from torchvision.transforms import Normalize, Compose, Resize, ColorJitter, CenterCrop, RandomVerticalFlip

class FacesDataSet(Dataset):
    def __init__(self, path):
        self.paths = glob.glob(path+r'*.jpg')
        self.transform = Compose([
            # ColorJitter(0.1),
            RandomVerticalFlip(p=.5),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            Resize(64)   
        ])
        
    def __getitem__(self, index):     
        p = self.paths[index]
        img = read_image(p).float() # [3, h, w], e.g. h=w=96
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    
dataset = FacesDataSet('./faces/')
dl = DataLoader(dataset, batch_size=64, drop_last=True, shuffle=True)

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./logs/exp1')

import time
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

train_epoch = 300
fixed_z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1).cuda()    # fixed noise

print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    D_losses = []
    G_losses = []
    epoch_start_time = time.time()
    with autocast(enabled=True):
        for x_ in dl:
            # train discriminator D
            D.zero_grad()

            mini_batch = x_.size()[0]

            y_real_ = torch.ones(mini_batch) - .1
            y_fake_ = torch.zeros(mini_batch) +.1

            x_, y_real_, y_fake_ = x_.cuda(), y_real_.cuda(), y_fake_.cuda()
            D_result = D(x_).squeeze()
            D_real_loss = BCE_loss(D_result, y_real_)

            z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            z_ = z_.cuda()
            G_result = G(z_)

            D_result = D(G_result).squeeze()
            D_fake_loss = BCE_loss(D_result, y_fake_)
            D_fake_score = D_result.data.mean()

            D_train_loss = D_real_loss + D_fake_loss

            scaler_d.scale(D_train_loss).backward()
            scaler_d.step(D_optimizer)
            scaler_d.update()

            # D_losses.append(D_train_loss.data[0])
            D_losses.append(D_train_loss.item())

            # train generator G
            G.zero_grad()

            z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            z_ = z_.cuda()

            G_result = G(z_)
            D_result = D(G_result).squeeze()
            G_train_loss = BCE_loss(D_result, y_real_)
            
            scaler_g.scale(G_train_loss).backward()
            scaler_g.step(G_optimizer)
            scaler_g.update()
            
            G_losses.append(G_train_loss.item())

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    d_loss = torch.mean(torch.FloatTensor(D_losses))
    g_loss = torch.mean(torch.FloatTensor(G_losses))
    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                               torch.mean(torch.FloatTensor(G_losses))))

    writer.add_scalars('loss', {'d_loss':d_loss.item(), 'g_loss': g_loss.item()}, global_step=epoch)

    if epoch%20==0:
        grid = make_grid(G_result, 8, normalize=True)
        writer.add_image('generate_img', grid, epoch)
        # save_image(grid, f'D:\desktop\generation\{epoch}.jpg')
        data = {
            'g': G.state_dict(),
            'd': D.state_dict(),
            'scaler_d': scaler_d.state_dict(),
            'scaler_g': scaler_g.state_dict()
        }
        


end_time = time.time()
total_ptime = end_time - start_time
