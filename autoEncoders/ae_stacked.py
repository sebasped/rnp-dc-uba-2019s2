#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 17:26:31 2019

@author: sebas
"""

import torch
import torchvision as tv

from ae import AE



# Autoencoders
class AES(torch.nn.Module):
    def __init__(_, sizes): #sizes es una lista resp a la arquitectura profunda
        super().__init__()
        _.subnet = torch.nn.ModuleList() #lista para recuperar los parámetros de todas las RBM
#        _.L1 = torch.nn.Linear(vsize, hsize)
#        _.L2 = torch.nn.Linear(hsize, vsize)
        for i in range(len(sizes)-1):
            _.subnet.append( AE(sizes[i],sizes[i+1]) )
#        _.output = torch.nn.Linear(sizes[-2],sizes[-1])


    def forward(_, x, depth=None):
#        h = _.enc(x)
#        out = _.dec(h)
#        return out
#        xi = x
        if depth is not None:
            for i in range(depth):
                x = _.subnet[i].enc(x)
            for i in range(depth):
                x = _.subnet[depth-i].dec(x)
#            xo = _.subnet[depth].forward(xi)
        else:
            for ae in _.subnet:
                x = ae.enc(x)
            for ae in reversed(_.subnet):
                x = ae.dec(x)
#            vo = _.output(hp)
        return x
    
    # enc y dec se pueden reemplazar por un torch Sequential: probarlo luego.
#    def enc(_,x, depth=None):
#        return torch.tanh(_.L1(x))
    
#    def dec(_,h, depth=None):
#        return torch.tanh(_.L2(h))


#transf = tv.transforms.Compose( tv.transforms.ToTensor(), tv.transforms.Normalize([0.5],[0.5]) )
#
#B = 100
#trn_data =tv.datasets.MNIST(root='./data', train = True, download = True, transform = transf) 
#tst_data =tv.datasets.MNIST(root='./data', train = False, download = True, transform = transf)
#trn_load =torch.utils.data.DataLoader(dataset = trn_data, batch_size = B, shuffle = True) # Suffle: que los datos est'en armados al azar. Lo mismo para ambos conjuntos.
#tst_load =torch.utils.data.DataLoader(dataset = tst_data, batch_size = B, shuffle = True)


# Pasamos de 10 a 3 dimensiones, con 1000 instancias
N = 10
M = 3
P = 1000

# me armo una entrada
z = torch.randn(P,M)
n = torch.randn(M,N)
x = torch.tanh(torch.mm(z,n)) # torch.mm es matrix multiply


aes = AES([N,M])
optim = torch.optim.SGD(aes.parameters(), 0.1) #el último es el learning rate
    
costf = torch.nn.MSELoss()

T = 1000
#errorTotal = []
aes.train()
for t in range(T):
#    errorBatch = []
#    for images, labels in trn_load:
    optim.zero_grad()
            
#            data = images.view(-1,28*28)
#            v0, vk = model(data, depth) #por defecto llama a forward
    y = aes(x)
    
    error = costf(y,x) # el autoencoder chequea si el input es igual al output
    error.backward()
#    loss = rbm.free_energy(v0) - rbm.free_energy(vk)
            
#    loss.backward()
    optim.step()
            
#    errorBatch.append(loss.item())
        
#   errorEpoca = sum(errorBatch)#/len(error)
#   errorTotal.append(errorEpoca) #por si después quiero graficar error para cada época
#    print('Depth', depth,'Época',t,'Error',errorEpoca)
#    print(Época',t,'Error',errorEpoca)
    E = error.item()
    
    if t%100 == 0:
        print(t,E)




#
#sizes = [28*28, 500, 500, 2000, 10]
#model = DBN(sizes)
#
#B = 100
#trn_data =tv.datasets.MNIST(root='./data', train = True, download = True, transform = tv.transforms.ToTensor()) # Primer par'ametro: En qu'e subdirectorio guardo los datos. Train son los datos de entrenamiento. Si no los tiene que los descargue. Transform: que los datos sean interpretados como tensores
#tst_data =tv.datasets.MNIST(root='./data', train = False, download = True, transform = tv.transforms.ToTensor())
#trn_load =torch.utils.data.DataLoader(dataset = trn_data, batch_size = B, shuffle = True) # Suffle: que los datos est'en armados al azar. Lo mismo para ambos conjuntos.
#tst_load =torch.utils.data.DataLoader(dataset = tst_data, batch_size = B, shuffle = True)
#
#
#T = 5 # cant de épocas
#
## primero (pre)entrenamos parcialmente cada RBM
#for depth in range(len(model.subnet)):
#    rbm = model.subnet[depth]
#    optim = torch.optim.SGD(rbm.parameters(), 0.01) #el último es el learning rate
#    
#    errorTotal = []
#    for t in range(T):
#        errorBatch = []
#        for images, labels in trn_load:
#            optim.zero_grad()
#            
#            data = images.view(-1,28*28)
#            v0, vk = model(data, depth) #por defecto llama a forward
#            loss = rbm.free_energy(v0) - rbm.free_energy(vk)
#            
#            loss.backward()
#            optim.step()
#            
#            errorBatch.append(loss.item())
#        
#        errorEpoca = sum(errorBatch)#/len(error)
#        errorTotal.append(errorEpoca) #por si después quiero graficar error para cada época
#        print('Depth', depth,'Época',t,'Error',errorEpoca)
#        
#
##lincl = torch.nn.Linear(M,C) #M features y C clases
#optim = torch.optim.Adam(model.parameters()) #entreno TODOS los parámetros del modeo, i.e. de TODAS las RBMs.
#costf = torch.nn.CrossEntropyLoss()
#
##T2 = T
#T2 = T*3
#errorTotal = []
#for t in range(T2):
#    errorBatch = []
#    for images, labels in trn_load:
#        optim.zero_grad()
#        
#        data = images.view(-1,28*28)
#        x, y = model(data)
#
#        error = costf(y,labels)
#        
#        error.backward()
#        optim.step()
#        
#        errorBatch.append(error.item()) 
#    
#    errorEpoca = sum(errorBatch)/len(errorBatch)
#    errorTotal.append(errorEpoca)
#    print('Época',t,'Error',errorEpoca)
#
#
#
#right, total = 0., 0.
#with torch.no_grad():
#    for images, labels in tst_load:
#        x = images.view(-1,28*28)
#        bla, y = model(x)
##        y = lincl(hp)
#        right += (y.argmax(dim=1)==labels).sum().item()
#        total += len(labels)
#        
#acc = right/total
#print('Accuracy:',acc)