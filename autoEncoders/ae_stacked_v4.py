#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 17:26:31 2019

@author: sebas
"""

import torch
import torchvision as tv

from ae import AE



# Autoencoders stacked
class AES(torch.nn.Module):
    def __init__(_, sizes): #sizes es una lista resp a la arquitectura profunda
        super().__init__()
        _.subnet = torch.nn.ModuleList() #lista para recuperar los parámetros de todas las RBM
        for i in range(len(sizes)-1):
            _.subnet.append( AE(sizes[i],sizes[i+1]) )


    def forward(_, x, depth=None):
        enc = _.enc(x,depth)
        dec = _.dec(enc,depth)
        return enc, dec
    
    # enc y dec se pueden reemplazar por un torch Sequential: probarlo luego.
    def enc(_,x, depth=None):
        if depth is not None:
            for i in range(depth):
                x = _.subnet[i].enc(x)
            x = _.subnet[depth].enc(x)
        else:
            for i in range(len(_.subnet)):
                x = _.subnet[i].enc(x)
#            x = _.subnet[depth].enc(x)
            
        return x
    
    def dec(_,x, depth=None):
        if depth is not None:
            for i in range(depth):
                x = _.subnet[depth-i].dec(x)
            x = _.subnet[0].dec(x)
        else:
            for i in range(len(_.subnet)):
                x = _.subnet[len(_.subnet)-i-1].dec(x)
        return x


# el Compose recibe una lista
transf = tv.transforms.Compose( [tv.transforms.ToTensor(), tv.transforms.Normalize([0.5],[0.5]) ] )

B = 100
trn_data =tv.datasets.MNIST(root='./data', train = True, download = True, transform = transf) 
tst_data =tv.datasets.MNIST(root='./data', train = False, download = True, transform = transf)
trn_load =torch.utils.data.DataLoader(dataset = trn_data, batch_size = B, shuffle = True) # Suffle: que los datos est'en armados al azar. Lo mismo para ambos conjuntos.
tst_load =torch.utils.data.DataLoader(dataset = tst_data, batch_size = B, shuffle = True)


N = 28*28 #input
M = 64 #output
C = 10 #cant classes para clasificar

sizes = [N,28*15,28*10,28*5,M]
model = AES(sizes)
#optim = torch.optim.SGD(model.parameters(), 0.1) #el último es el learning rate
#optim = torch.optim.Adam(model.parameters()) #el último es el learning rate
costf = torch.nn.MSELoss()

T = 10
#errorTotal = []
model.train()
#pre entreno por profundidad creciente solamente los AEs
for depth in range(len(model.subnet)):
    for t in range(T):
        #solamente entreno hasta la profundidad correspondiente
        ae = model.subnet[:depth+1]
        optim = torch.optim.Adam(ae.parameters()) #el último es el learning rate

        errorBatch = []
        for images, labels in trn_load:
            optim.zero_grad()
            
            x = images.view(-1,N)
            end, dec = model(x,depth)
    
            error = costf(dec,x) # el autoencoder chequea si el input es igual al output
            error.backward()

            optim.step()

            errorBatch.append(error.item())
        
        errorEpoca = sum(errorBatch)#/len(error)
#   errorTotal.append(errorEpoca) #por si después quiero graficar error para cada época
        print('Depth', depth,'Época',t,'Error',errorEpoca)


        
# Ahora entreno con la profundidad entera (i.e. AEs + el clasificador final)
lincl = torch.nn.Linear(M,C) #M features y C clases
optim = torch.optim.Adam(model.parameters()) #entreno TODOS los parámetros del modelo
costf = torch.nn.CrossEntropyLoss()

T2 = T
#errorTotal = []
for t in range(T2):
    errorBatch = []
    for images, labels in trn_load:
        optim.zero_grad()
        
        data = images.view(-1,N)
        enc = model.enc(data)
        y = lincl(enc)
        
        error = costf(y,labels) 
        error.backward()
        optim.step()
        
        errorBatch.append(error.item()) 
    
    errorEpoca = sum(errorBatch)/len(errorBatch)
#    errorTotal.append(errorEpoca)
    print('Época',t,'Error',errorEpoca)



model.eval()
right, total = 0., 0.
with torch.no_grad():
    for images, labels in tst_load:
        x = images.view(-1,N)
        enc, bla = model(x)
        y = lincl(enc)
        right += (y.argmax(dim=1)==labels).sum().item()
        total += len(labels)
        
acc = right/total
print('Accuracy:',acc)