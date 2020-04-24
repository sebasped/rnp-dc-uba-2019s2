#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 18:15:55 2019

@author: sebas
"""


import torch

# quedamos atrapados en mínimos locales
# muy supeditado a datos random de entrada

# probar de modificar los hiperparámetros
# más capas ocultas
# demasiadas capas ocultas


# incorporar ruido al entrenamiento


# esto es entrenamiento por batch

# sol a mitad de camino: mini batch
# ver si mejora 
#bs = 10
#rp = torch.randperm(P)
#for mb in range(0,P,bs):
#    i = rp[mb:mb+bs]
#    ... x[i], bias[i]
#    ... h, bias[i]


# sol. incremental de a 1: stochastic gradient descent



P = 100
N = 8
H = N+1  # capa oculta con N+1 unidades
M = 1

#m = torch.randn(N+1,M) ### randn es para distribucion normal
x = torch.randn(P,N).sign() ### rand es para distribucion uniforme
z = torch.prod(x,dim=1).view(P,1)

w1 = torch.randn(N+1,H, requires_grad=True)
w2 = torch.randn(H+1,M, requires_grad=True)

bias = torch.ones(P,1)

lr = 1e-2
E, t = 1.,0 # inicializo un error y un contador de epocas

while E>0.0001 and t<10001:
    h = torch.cat((x,bias),dim=1).mm(w1).tanh()
    y = torch.cat((h,bias),dim=1).mm(w2).tanh()
    error = (y-z).pow(2).sum() ### error para los datos con ruido
    error.backward() ### calcula gradiente y lo guarda en w
    with torch.no_grad():
        w1 -= lr*w1.grad
        w2 -= lr*w2.grad
        w1.grad.zero_() # del tensor existente, lo completa con zeros
        w2.grad.zero_() # del tensor existente, lo completa con zeros
    E = error.item()/P
    t += 1 #paso de época
    if t%100 ==0:
        print(t,E)
        
# funciones en torch para pasar hacia o desde un array de numpy