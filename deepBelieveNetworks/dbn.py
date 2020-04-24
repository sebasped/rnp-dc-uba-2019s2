#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 17:26:31 2019

@author: sebas
"""

import torch
import torchvision as tv

from rbm import RBM


# intentamos reconstruir el modelo original de Hinton
#[28*28, 500, 500, 2000, 10]

class DBN(torch.nn.Module):
    def __init__(_, sizes, CD_k=1): #sizes es una lista resp a la arquitectura profunda
        super().__init__()
        _.subnet = torch.nn.ModuleList() #lista para recuperar los parámetros de todas las RBM
        for i in range(len(sizes)-2):
            _.subnet.append( RBM(sizes[i],sizes[i+1], CD_k) )
        _.output = torch.nn.Linear(sizes[-2],sizes[-1])
    
    def forward(_, v, depth=None): # depth indica qué capa quiero entrenar
        vi = v
        # Para entrenar con profundidad creciente
        if depth is not None:
            for i in range(depth):
                hp, vi = _.subnet[i].sample_h(vi)
            vp, vo = _.subnet[depth].forward(vi)
        else: # Para hacer el backpropagation del final usando toda la arquitectura entera
            for rbm in _.subnet:
                hp, vi = rbm.sample_h(vi)
            vo = _.output(hp)
        return vi, vo
    
    
sizes = [28*28, 500, 500, 2000, 10]
model = DBN(sizes)

B = 100
trn_data =tv.datasets.MNIST(root='./data', train = True, download = True, transform = tv.transforms.ToTensor()) # Primer par'ametro: En qu'e subdirectorio guardo los datos. Train son los datos de entrenamiento. Si no los tiene que los descargue. Transform: que los datos sean interpretados como tensores
tst_data =tv.datasets.MNIST(root='./data', train = False, download = True, transform = tv.transforms.ToTensor())
trn_load =torch.utils.data.DataLoader(dataset = trn_data, batch_size = B, shuffle = True) # Suffle: que los datos est'en armados al azar. Lo mismo para ambos conjuntos.
tst_load =torch.utils.data.DataLoader(dataset = tst_data, batch_size = B, shuffle = True)


T = 5 # cant de épocas

# primero (pre)entrenamos parcialmente cada RBM. Vamos en profundidad creciente.
for depth in range(len(model.subnet)):
    rbm = model.subnet[depth]
    optim = torch.optim.SGD(rbm.parameters(), 0.01) #el último es el learning rate
    
    errorTotal = []
    for t in range(T):
        errorBatch = []
        for images, labels in trn_load:
            optim.zero_grad()
            
            data = images.view(-1,28*28)
            v0, vk = model(data, depth) #por defecto llama a forward. Voy entrenando cada vez más profundo.
            loss = rbm.free_energy(v0) - rbm.free_energy(vk)
            
            loss.backward()
            optim.step()
            
            errorBatch.append(loss.item())
        
        errorEpoca = sum(errorBatch)#/len(error)
        errorTotal.append(errorEpoca) #por si después quiero graficar error para cada época
        print('Depth', depth,'Época',t,'Error',errorEpoca)
        

#lincl = torch.nn.Linear(M,C) #M features y C clases
optim = torch.optim.Adam(model.parameters()) #entreno TODOS los parámetros del modeo, i.e. de TODAS las RBMs.
costf = torch.nn.CrossEntropyLoss()

#T2 = T
T2 = T*3
errorTotal = []
for t in range(T2):
    errorBatch = []
    for images, labels in trn_load:
        optim.zero_grad()
        
        data = images.view(-1,28*28)
        x, y = model(data) # sin depth porque hago una especie de backpropagation con todo

        error = costf(y,labels)
        
        error.backward()
        optim.step()
        
        errorBatch.append(error.item()) 
    
    errorEpoca = sum(errorBatch)/len(errorBatch)
    errorTotal.append(errorEpoca)
    print('Época',t,'Error',errorEpoca)



right, total = 0., 0.
with torch.no_grad():
    for images, labels in tst_load:
        x = images.view(-1,28*28)
        bla, y = model(x)
#        y = lincl(hp)
        right += (y.argmax(dim=1)==labels).sum().item()
        total += len(labels)
        
acc = right/total
print('Accuracy:',acc)