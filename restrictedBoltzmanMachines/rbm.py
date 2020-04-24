# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:11:07 2019

@author: Sebas
"""


import torch
import torchvision as tv
from matplotlib import pyplot as plt
#import numpy
import torch.nn.functional as F



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
    T = 20
#    B = 64
    B = 100

    trn_data =tv.datasets.MNIST(root='./data', train = True, download = True, transform = tv.transforms.ToTensor()) # Primer par'ametro: En qu'e subdirectorio guardo los datos. Train son los datos de entrenamiento. Si no los tiene que los descargue. Transform: que los datos sean interpretados como tensores
    tst_data =tv.datasets.MNIST(root='./data', train = False, download = True, transform = tv.transforms.ToTensor())
    trn_load =torch.utils.data.DataLoader(dataset = trn_data, batch_size = B, shuffle = True) # Suffle: que los datos est'en armados al azar. Lo mismo para ambos conjuntos.
    tst_load =torch.utils.data.DataLoader(dataset = tst_data, batch_size = B, shuffle = True)
    
    N = 28*28 # cant visibles: entrada
    M = 64 # cant ocultas
#    M = N
    C = 10
    
    
    rbmfd = RBM(N,M)
    optim = torch.optim.SGD(rbmfd.parameters(), 0.1) #el último es el learning rate
    
    rbmfd.train() # lo pongo al modelo en modo entrenamiento. Ver docs, funciona para algunos módulos
    T = 40 # cant épocas
    
    
    errorTotal = []
    for t in range(T):
        errorEpoca = []
        for images, labels in trn_load:
            
            data = images.view(-1,N)
            v0, vk = rbmfd(data)
            loss = rbmfd.free_energy(v0) - rbmfd.free_energy(vk)
#            x = images.reshape(-1,N)
#            y = model(x)
#            z = torch.zeros(size = (len(labels),C))
#            z[torch.arange(len(labels)),labels] = 1

            optim.zero_grad() # conviene resetear el gradiente por optim 
        
#            errorAux = loss
            errorEpoca.append(loss.item()) 
#            errorEpoca.append(errorAux)
            loss.backward() # calcula el gradiente y lo guarda 
    
            optim.step() #recalcular por gradiente descendiente
     
        #    E = error.item()
        E = sum(errorEpoca)#/len(error)
        errorTotal.append(E)
#        t += 1 #paso de época
#    
    #    if t%100 ==0:
    #        print(t,E)
        print(t,E)
    
    
    plt.plot(range(T),errorTotal)
    
    
    
    rbmfd.eval() # lo pongo al modelo en modo validación. Ver docs, funciona para algunos módulos
    
    lincl = torch.nn.Linear(M,C) #M features y C clases
    optim2 = torch.optim.SDG(lincl.parameters(), 0.1)
    costf = torch.nn.CrossEntropyLoss()

        
    
    
    
        
# Vamos a usar ahora el perceptr'on multicapa que hicimos la semana pasada.
#model = mlp(N,m,C)
#model = mlp([N,m,m,C])
# La funci'on de costo:
#costf = torch.nn.MSELoss()
#costf = torch.nn.CrossEntropyLoss() # bastante popular para problemas de clasificación
#optim = torch.optim.Adam(model.parameters(), lr = 1e-3)


# Para armar el ciclo de entrenamiento hay que lupear sobre los lotes minibach. Lo hacemos de esta forma:
# Iterando sobre trn_load 
# Imagenews con dimensiones raras, debo acomodarlas. La red tiene un vector de dimension mxm. Al lote de imagenes hacerle un reshape.
# No importa cuantas dimensiones tenga el tensor internamente, calcula el n'umero de filas pero lo que es seguro es que tenga M columnas.
#Al tensor entero de algo multidimensional a un matris de xporN
# La salida de red es un vector de 10 unidades. Quiero que sehaga un one_hot (todos ceros y un solo uno en posici'on correspondiente)
# Si label es 5, en la posici'on 5 estar'a en uno, el resto en cero. C'omo hacer esto en dos l'ineas?
# Ver z. Size: cuantas etiquetas vinieron en el lote por C clases. En z genero un tensor datos en lote filas columnas: data sets en clases. Luego poner los unos en los lugares correspondientes.
# Lo que quiero hacer es z[0,5] = 1, z[1,3] = 1, z[2,9] = 1. C'omo hacer esto?
# torch.arange: La 'ultima linea() hace esto.
#while ....:
#    for imagees, labels in trn_load:
#        x = images.reshape(-1,N)
#        z = torch.zeros(size = (len(labels),C))
#        z[torch.arange(len(labels),labels] = 1

#
#E, t = 1.,0 # inicializo un error y un contador de epocas
#error = [] 
##tv
##from matplotlib import pyplot as plt
##import numpy
#
#
#
#model.train() # lo pongo al modelo en modo entrenamiento. Ver docs, funciona para algunos módulos
#
#while E>0.05 and t<101:
#    
#
#    for images, labels in trn_load:
#        x = images.reshape(-1,N)
#        y = model(x)
#        z = torch.zeros(size = (len(labels),C))
#        z[torch.arange(len(labels)),labels] = 1
#
#        optim.zero_grad() # conviene resetear el gradiente por optim 
#        
##        errorAux = costf(y,z)
#        errorAux = costf(y,labels)
#        error.append(errorAux.item()) # calculo el torch.cuda.is_available()rror entre target y modelo
#        errorAux.backward() # calcula el gradiente y lo guarda 
#    
##    with torch.no_grad():
##        w1 -= lr*w1.grad
##        w2 -= lr*w2.grad
##        w1.grad.zero_() ### del tensor existente, lo completa con zeros
##        w2.grad.zero_() ### del tensor existente, lo completa con zeros
#    
##        for param in model.parameters():
##            param -= lr*param.grad
#        optim.step() #recalcular por gradiente descendiente
#     
##    E = error.item()
#    E = sum(error)/len(error)
#    t += 1 #paso de época
#    
##    if t%100 ==0:
##        print(t,E)
#    print(t,E)
#    
#
#model.eval() # lo pongo al modelo en modo validación. Ver docs, funciona para algunos módulos
#
#right, total = 0., 0.
#for images, labels in tst_load:
#    x = images.reshape(-1,N)
#    y = model(x)
#    right += (y.argmax(dim=1)==labels).sum().item()
#    total += len(labels)
#        
#acc = right/total
#print('Accuracy:',acc)
#
#
#
#with torch.no_grad():
#    w = model.layers[0].weight[2] #se puede probar cambiando las neuronas
##    print(w.numpy())
#    
#    plt.matshow(w.view(28,28).detach().numpy())
#    plt.show()