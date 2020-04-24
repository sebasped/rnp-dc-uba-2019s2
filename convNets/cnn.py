#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:35:37 2019

@author: sebas
"""

import torch
import torchvision as tv


# el Compose recibe una lista
# el primer arg del Compose es mean, (value-mean)/std ; y el segundo es std
transf = tv.transforms.Compose( [tv.transforms.ToTensor(), tv.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))] )
B = 100 #tamaño del batch
trn_data =tv.datasets.CIFAR10(root='./data', train = True, download = True, transform = transf) 
tst_data =tv.datasets.CIFAR10(root='./data', train = False, download = True, transform = transf)
trn_load =torch.utils.data.DataLoader(dataset = trn_data, batch_size = B, shuffle = True) # Suffle: que los datos est'en armados al azar. Lo mismo para ambos conjuntos.
tst_load =torch.utils.data.DataLoader(dataset = tst_data, batch_size = B, shuffle = True)

# Datos: 10 labels, 3 canales, de 32x32 cada uno


#Convoluciones
#kernel_size: entero ventana cuadrada, o tupla para dimensiones
#1d, 2d, 3d dependiendo de cómo es la entrada
#kernel size = 5
#stride = 2
#padding = 2
#torch.nn.Conv2d(in_chann, out_chann, kernel_size, stride, padding)

#Pooling
#kernel_size = 3

#torch.nn.MaxPool2d(kernel_size, padding)


#Probamos a mano el tema de las dimnensiones para pasarle al MLP final
idata = iter(trn_load)
image, label = next(idata)
print('image shape: ', image.shape)
print('label shape: ', label.shape)
#
#in_chann, out_chann, kernel_size, stride, padding = 3, 16, 5, 2, 2
c1 = torch.nn.Conv2d(3, 16, 5, 2, 2)

y1 = c1(image)
print('c1 shape: ', y1.shape)

#c2 = torch.nn.Conv2d(16, 32, 3, 1, 1)
#y2 = c2(y1)

#c3 = torch.nn.MaxPool2d(2,2)
#y3 = c3(y2)

#view(-1,32*8*8) #porque los Linears reciben vectores y no tensores
#Linear(32*8*8, 512)
#Linear(512, 10) 