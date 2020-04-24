# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:11:07 2019

@author: Administrator
"""

import torch
import torchvision as tv


trn_data =tv.datasets.MNIST(root='./data', train = True, download = True, transform = tv.transforms.ToTensor()) # Primer par'ametro: En qu'e subdirectorio guardo los datos. Train son los datos de entrenamiento. Si no los tiene que los descargue. Transform: que los datos sean interpretados como tensores
tst_data =tv.datasets.MNIST(root='./data', train = False, download = True, transform = tv.transforms.ToTensor())
# Descarga entonces dos datasets distintos, uno para entrenar y otro para testear. Sin embargo, los datasets no son distintos.
# Torch al parecer no carga todos los datos en memoria, pues los datasets suelen ser muy grandes.
# Nos da una facilidad extra para trabajar con minibacht: nos da el DataLoader. Uno para entrenamiento y otro para validaci'on.

B = 100

trn_load =torch.utils.data.DataLoader(dataset = trn_data, batch_size = B, shuffle = True) # Suffle: que los datos est'en armados al azar. Lo mismo para ambos conjuntos.
tst_load =torch.utils.data.DataLoader(dataset = tst_data, batch_size = B, shuffle = True)

# El P ue ven'iamos definiendo se lo decimos :
P = len(trn_data)
# Al parecer no carga todos los datos pero al pedirle el tama;o da el n'umero corecto.
# Ahora el N:
N = trn_data[0][0].nelement() # Es una lista. Cada elemento da el input y el target. La posici'on cero es un tensor de la forma (1,28,28). En algunos casos esta dimension es el tama;o del minibach. 
# Necesitamos saber cu'antas unidades tendremos a la salida. Se puede calcular de forma autom'atica de forma engorrosa pero solo hay 10 posibilidades:
C = 10


# Pegar definicion de la classe aqu'i:
class mlp(torch.nn.Module):
    # Ahora hacemos un constructor. El primer par'ametro es la autoreferencia al objeto
    # Al crear un objeto mlp debo proporcionar los siguientes tres par'ametros
    def __init__(_,sizes):
        super().__init__() 
        # Ahora todas las capas ocultas que querramos
        _.layers = torch.nn.ModuleList()
        for i in range(len(sizes)-1):
            _.layers.append(torch.nn.Linear(sizes[i],sizes[i+1]))
#        _.L1 = torch.nn.Linear(isize,hsize)
#        _.L2 = torch.nn.Linear(hsize,osize)
        
    def forward(_,x):
        h = x
        for hidden in _.layers[:-1]:
            h = torch.sigmoid(hidden(h))
        output = _.layers[-1]
#        h = torch.tanh(_.L1(x))
#        y = torch.tanh(_.L2(h))
#        h = torch.sigmoid(_.L1(x))
#        y = torch.sigmoid(_.L2(h))
        y = torch.softmax(output(h), dim=1)

        return y  


#LR = 1e-2

m = N+1 # jugar con este. Lo definimos nosotros

   
# Vamos a usar ahora el perceptr'on multicapa que hicimos la semana pasada.
#model = mlp(N,m,C)
model = mlp([N,m,m,C])
# La funci'on de costo:
costf = torch.nn.MSELoss()
#costf = torch.nn.CrossEntropyLoss() # bastante popular para problemas de clasificación
optim = torch.optim.Adam(model.parameters(), lr = 1e-3)

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


E, t = 1.,0 # inicializo un error y un contador de epocas
error = []

while E>0.005 and t<101:
    

    for images, labels in trn_load:
        x = images.reshape(-1,N)
        y = model(x)
        z = torch.zeros(size = (len(labels),C))
        z[torch.arange(len(labels)),labels] = 1

        optim.zero_grad() # conviene resetear el gradiente por optim 
        
        errorAux = costf(y,z)
        error.append(errorAux.item()) # calculo el torch.cuda.is_available()rror entre target y modelo
        errorAux.backward() # calcula el gradiente y lo guarda 
    
#    with torch.no_grad():
#        w1 -= lr*w1.grad
#        w2 -= lr*w2.grad
#        w1.grad.zero_() ### del tensor existente, lo completa con zeros
#        w2.grad.zero_() ### del tensor existente, lo completa con zeros
    
#        for param in model.parameters():
#            param -= lr*param.grad
        optim.step() #recalcular por gradiente descendiente
     
#    E = error.item()
    E = sum(error)/len(error)
    t += 1 #paso de época
    
#    if t%100 ==0:
#        print(t,E)
    print(t,E)