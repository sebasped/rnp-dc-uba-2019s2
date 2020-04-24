#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 22:18:28 2019

@author: sebas
"""

import torch
import torchvision as tv


class CNN(torch.nn.Module):
    def __init__(self, in_chann=3, out_classes=10): 
        super().__init__()
        self.c1 = torch.nn.Conv2d(in_chann, 16, 5, 2, 2)
        self.c2 = torch.nn.Conv2d(16, 32, 3, 1, 1)
        self.c3 = torch.nn.MaxPool2d(2,2)
        self.H = 32*8*8
        self.L1 = torch.nn.Linear(self.H, 512)
        self.L2 = torch.nn.Linear(512,out_classes)

    def forward(self, x):
        y1 = self.c1(x).relu()
        y2 = self.c2(y1)
        y3 = self.c3(y2).relu()
        y4 = self.L1(y3.view(-1,self.H)).tanh()
        y5 = self.L2(y4)

        return y5


if __name__ == '__main__':
    
    transf = tv.transforms.Compose( [tv.transforms.ToTensor(), tv.transforms.Normalize([0.5],[0.5],[0.5]) ] )
    B = 100
    trn_data =tv.datasets.CIFAR10(root='./data', train = True, download = True, transform = transf) 
    tst_data =tv.datasets.CIFAR10(root='./data', train = False, download = True, transform = transf)
    trn_load =torch.utils.data.DataLoader(dataset = trn_data, batch_size = B, shuffle = True)
    tst_load =torch.utils.data.DataLoader(dataset = tst_data, batch_size = B, shuffle = True)
    
    #idata = iter(trn_load)
    #image, label = next(idata)
    #print('image shape: ', image.shape)
    #print('label shape: ', label.shape)
    
    #c1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2)
    
    #y1 = c1(image)
    #print('c1 shape: ', y1.shape)
    
    #c2 = torch.nn.Conv2d(16, 32, 3, 1, 1)
    #y2 = c2(y1)
    
    #c3 = torch.nn.MaxPool2d(2,2)
    #y3 = c3(y2)
    
    #view(-1,32*8*8) #porque los Linears reciben vectores y no tensores
    #Linear(32*8*8, 512)
    #Linear(512, 10)
    
    model = CNN()
    optim = torch.optim.Adam(model.parameters())
    costf = torch.nn.CrossEntropyLoss()
    
    T = 20
    model.train()
    for t in range(T):
      E = 0
      for image, label in trn_load:
        optim.zero_grad()
        y = model(image)
        error = costf(y, label)
        error.backward()
        optim.step()
        E += error.item()
      print(t, E) 
      
      
    model.eval()
    right, total = 0, 0
    with torch.no_grad():
        for images, labels in tst_load:
            y = model(images)
            right += (y.argmax(dim=1)==labels).sum().item()
            total += len(labels)

    accuracy = right / total
    print('Accuracy: ', accuracy)



"""
puedo usar la red convolucional para hacer un autoencoder

_.enc = torch.nn.Sequential(
  conv2d(...),
  torch.nn.ReLU(True))

_.dec = torch.nn.ConvTranspose2d()
Tanh()
"""