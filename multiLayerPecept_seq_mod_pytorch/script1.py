#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 17:11:42 2019

@author: sebas
"""

# ídem clase pasada con implementación "más copada"

import torch

P = 100
N = 8
H = N+1  # capa oculta con N+1 unidades
M = 1

#m = torch.randn(N+1,M) ### randn es para distribucion normal
x = torch.randn(P,N).sign() ### rand es para distribucion uniforme
z = torch.prod(x,dim=1).view(P,M)

#w1 = torch.randn(N+1,H, requires_grad=True)
#w2 = torch.randn(H+1,M, requires_grad=True)

#bias = torch.ones(P,1)

lr = 1e-2
E, t = 1.,0 # inicializo un error y un contador de epocas

model = torch.nn.Sequential( torch.nn.Linear(N,H), 
                            torch.nn.Tanh(), 
                            torch.nn.Linear(H,M), 
                            torch.nn.Tanh() )

costf = torch.nn.MSELoss(reduction='sum') #para que haga la suma y no el promedio



while E>0.0001 and t<10001:
#    h = torch.cat((x,bias),dim=1).mm(w1).tanh()
#    y = torch.cat((h,bias),dim=1).mm(w2).tanh()
    y = model(x)
    
#    error = (y-z).pow(2).sum() ### error para los datos con ruido
#    error.backward() ### calcula gradiente y lo guarda en w
    
    model.zero_grad() # tengo que resetear el gradiente antes de recalcular los pesos
    
    error = costf(y,z) # calculo el error entre target y modelo
    error.backward() # calcula el gradiente y lo guarda 
    
    
    with torch.no_grad():
#        w1 -= lr*w1.grad
#        w2 -= lr*w2.grad
#        w1.grad.zero_() ### del tensor existente, lo completa con zeros
#        w2.grad.zero_() ### del tensor existente, lo completa con zeros
    
        for param in model.parameters():
            param -= lr*param.grad
     
    E = error.item()/P
    
    t += 1 #paso de época
    
    if t%100 ==0:
        print(t,E)
    
