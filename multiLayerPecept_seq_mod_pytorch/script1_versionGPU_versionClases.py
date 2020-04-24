#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 17:11:42 2019

@author: sebas
"""

# ídem clase pasada con implementación "más copada"

import torch

# defino la clase del modelo (por ahora con una sola capa oculta)
# el guión bajo _ es el self
# mínimo para def clase: funciones init y forward
# arquitectura y aprendizaje

class mlp(torch.nn.Module):
    def __init__(_, isize, hsize, osize):
        super().__init__() # inicializo la super clase
        _.L1 = torch.nn.Linear(isize, hsize)
        _.L2 = torch.nn.Linear(hsize, osize)

    def forward(_,x):
        h = torch.tanh(_.L1(x))
        y = torch.tanh(_.L2(h))
        return y

#device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )

P = 100
N = 8
H = N+1  # capa oculta con N+1 unidades
M = 1

x = torch.randn(P,N).sign() ### rand es para distribucion uniforme
z = torch.prod(x,dim=1).view(P,M) # el signo 8 bit como entrenamiento. NO es separable!

#x = x.to(device)
#z = z.to(device) # ¿Quizás adentro del while esto?

lr = 1e-2
E, t = 1.,0 # inicializo un error y un contador de epocas

model = mlp(N,H,M)#.to(device)
costf = torch.nn.MSELoss()

# agregamos optimización por stochastic gradient descent
# otra opción es el RProp o el Adam

optim = torch.optim.SGD(model.parameters(), lr=0.1)



while E>0.0001 and t<10001:
#    h = torch.cat((x,bias),dim=1).mm(w1).tanh()
#    y = torch.cat((h,bias),dim=1).mm(w2).tanh()
#    print(model)
    y = model(x)
#    y = model.foward(x)
    
#    error = (y-z).pow(2).sum() ### error para los datos con ruido
#    error.backward() ### calcula gradiente y lo guarda en w
    
#    model.zero_grad() # tengo que resetear el gradiente antes de recalcular los pesos
    optim.zero_grad() # conviene resetear el gradiente por optim 
    
    error = costf(y,z) # calculo el etorch.cuda.is_available()rror entre target y modelo
    error.backward() # calcula el gradiente y lo guarda 
#    optim.backward() # calcula el gradiente y lo guarda 
    
#    with torch.no_grad():
#        w1 -= lr*w1.grad
#        w2 -= lr*w2.grad
#        w1.grad.zero_() ### del tensor existente, lo completa con zeros
#        w2.grad.zero_() ### del tensor existente, lo completa con zeros
    
#        for param in model.parameters():
#            param -= lr*param.grad
    optim.step() #recalcular por gradiente descendiente
     
    E = error.item()
    t += 1 #paso de época
    
    if t%100 ==0:
        print(t,E)
    
