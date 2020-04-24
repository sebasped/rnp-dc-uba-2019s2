# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:11:07 2019

@author: Sebas
"""


import torch
import torchvision as tv
from matplotlib import pyplot as plt
#import numpy
import torch.nn.functional as F  #para usar el linear.



class RBM(torch.nn.Module):
    def __init__(_,vsize,hsize,CD_k=1):
        super().__init__() 
        _.w = torch.nn.Parameter(torch.randn(hsize,vsize)*1.e-2)
        _.bv = torch.nn.Parameter(torch.randn(vsize)*1.e-2)
        _.bh = torch.nn.Parameter(torch.randn(hsize)*1.e-2)
        _.k=CD_k
        
    def sample_h(_,v):
        #la probabilidad (dsitribución) la calculamos con el perceptron
        prob_h = torch.sigmoid(F.linear(v,_.w,_.bh))
        samp_h = torch.bernoulli(prob_h)
        return prob_h, samp_h
        
    def sample_v(_,h):
        prob_v = torch.sigmoid(F.linear(h,_.w.t(),_.bv)) #ojo transponemos los pesos w porque de hidden a visible es al revés las dimensiones
        samp_v = torch.bernoulli(prob_v)
        return prob_v, samp_v
        
    def forward(_,v):
        vs = v
        for i in range(_.k):
            hp, hs = _.sample_h(vs)
            vp, vs = _.sample_v(hs)
        return v, vs
        
    def free_energy(_,v):
        v_bv = v.mv(_.bv)
        hlin = torch.clamp(F.linear(v,_.w,_.bh),-80,80)
        slog = hlin.exp().add(1).log().sum(1)
        return (-slog-v_bv).mean()
    



if __name__ == '__main__':
    T = 20 # cant épocas
    B = 64
#    B = 100

    trn_data =tv.datasets.MNIST(root='./data', train = True, download = True, transform = tv.transforms.ToTensor()) # Primer par'ametro: En qu'e subdirectorio guardo los datos. Train son los datos de entrenamiento. Si no los tiene que los descargue. Transform: que los datos sean interpretados como tensores
    tst_data =tv.datasets.MNIST(root='./data', train = False, download = True, transform = tv.transforms.ToTensor())
    trn_load =torch.utils.data.DataLoader(dataset = trn_data, batch_size = B, shuffle = True) # Suffle: que los datos est'en armados al azar. Lo mismo para ambos conjuntos.
    tst_load =torch.utils.data.DataLoader(dataset = tst_data, batch_size = B, shuffle = True)
    
    N = 28*28 # cant visibles: entrada
    M = 64 # cant ocultas, i.e. features a aprender
#    M = N
    C = 10
    
    
    rbmfd = RBM(N,M)
    optim = torch.optim.SGD(rbmfd.parameters(), 0.1) #el último es el learning rate
    
    rbmfd.train() # lo pongo al modelo en modo entrenamiento. Ver docs, funciona para algunos módulos
    
    
    errorTotal = []
    for t in range(T):
        errorEpoca = []
        for images, labels in trn_load:
            optim.zero_grad() # conviene resetear el gradiente por optim 
            
            data = images.view(-1,N)
            v0, vk = rbmfd(data) #por defecto llama a forward

            loss = rbmfd.free_energy(v0) - rbmfd.free_energy(vk)
            errorEpoca.append(loss.item()) 

            loss.backward() # calcula el gradiente y lo guarda 
            optim.step() #recalcular por gradiente descendiente
     
        E = sum(errorEpoca)#/len(error)
        errorTotal.append(E)
        print(t,E)
    
    
    plt.plot(range(T),errorTotal)


    lincl = torch.nn.Linear(M,C) #M features y C clases
    optim = torch.optim.SGD(lincl.parameters(), 0.1) # el 0.1 es el learning rate
    costf = torch.nn.CrossEntropyLoss()

    T2 = T*4
    errorTotal2 = []
    for t in range(T2):
        error2 = []
        for images, labels in trn_load:
            optim.zero_grad() # conviene resetear el gradiente por optim 
            
            v = images.view(-1,N)
            hp, hs = rbmfd.sample_h(v)
            cp = lincl(hp)
            
            errorAux = costf(cp,labels)
            error2.append(errorAux.item()) 

            errorAux.backward() # calcula el gradiente y lo guarda
            optim.step() #recalcular por gradiente descendiente
     
        E = sum(error2)/len(error2)
        errorTotal2.append(E)
        print(t,E)
    
    
    plt.plot(range(T2),errorTotal2)
    
    
    
    rbmfd.eval() # lo pongo al modelo en modo validación. Ver docs, funciona para algunos módulos


    with torch.no_grad():
        right, total = 0., 0.
        for images, labels in tst_load:
            x = images.view(-1,N)
            hp, hs = rbmfd.sample_h(x)
            y = lincl(hp)
            right += (y.argmax(dim=1)==labels).sum().item()
            total += len(labels)
        
    acc = right/total
    print('Accuracy:',acc)



#        w = model.layers[0].weight[2] #se puede probar cambiando las neuronas
#        plt.matshow(w.view(28,28).detach().numpy())
#        plt.show()