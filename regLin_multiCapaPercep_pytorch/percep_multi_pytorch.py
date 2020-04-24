#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 18:15:55 2019

@author: sebas
"""


import torch

P = 1000
N = 9
M = 3
s = 0.01

m = torch.randn(N+1,M) ### randn es para distribucion normal
x = torch.rand(P,N+1) ### rand es para distribucion uniforme

x[:,-1] = 1
z = torch.mm(x,m) ### mm = matrix multiply, que es igual al dot product de numpy
w = torch.randn(N+1, M, requires_grad=True) ### va construyendo el grafo de computo para obtener los gradientes
xn = x + s * torch.randn(P,N+1) ### creamos datos con un poco de ruido
lr = 1e-4
E, t = 1., 0 # inicializo un error y un contador de epocas

while E>0.00001 and t<10001:
    y = torch.mm(xn,w) ### tensor de respuesta para todas mis instancias
    error = (y-z).pow(2).sum()
    error.backward() ### calcula gradiente y lo guarda en w
    with torch.no_grad():
        w -= lr * w.grad
        w.grad.zero_() ### del tensor existente, lo completa con zeros
    E = error.item()/P
    t += 1 #paso de época
    if t%10 ==0:
        print(t,E)

# el ruido ayuda a que converja a una mejor solución: es decir, generaliza mejor
# ojo, no es que converja más rápido, sino que converge a una mejor solución
# en el sentido antes descripto de generalización.
        
# Incluso mejor es que el ruido sea distinto para cada época.
# pues eso genera "nuevos" datos, y por lo tanto reduce riesgo de overfitting.


print( (torch.mm(x,w)-z).pow(2).mean().item() ) #error contra los datos SIN ruido.

# ¡El error es menor en los datos sin ruido! A pesar de que entrenamos con ruido.







#print(E)

#¿se parecen m y w? (es decir la posta y la entrenada)
#print(m)
#print(w)

# con este escaleado ahora sí se parecen
#c = w[0,0]/m[0,0]
#print(m*c)
#print(w)

#¿Generaliza bien el modelo entrenado?
#xp = np.random.uniform(-25,25,(100,N+1))
#xp[:,-1] = 1 #por el bias
##zp = np.sign(np.dot(xp,m))
#zp = np.sign(np.prod(xp,axis=1))
#yp = np.tanh(np.dot(xp,w))
##queremos comparar zp (el posta) con yp (la predicción)
#dp = zp - yp
#Ep = np.mean(np.square(dp))
#print(Ep) #la métrica de error es el MSE