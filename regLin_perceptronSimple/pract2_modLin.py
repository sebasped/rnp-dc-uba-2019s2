#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 18:36:55 2019

@author: sebas
"""


# Problema Regresión Lineal

import numpy as np

N = 6
M = 2
P = 1000

m = np.random.uniform(-1,1,(N+1,M))
x = np.random.uniform(-9,9,(P,N+1))
x[:,-1] = 1 #por el bias
z = np.dot(x,m)

Lr = 1e-5 #learning rate
#Lr = 0.00001

w = np.random.uniform(-0.1,0.1,(N+1,M)) #el modelo a entrenar

E, t = 1., 0 #inicializo error y contador de épocas.

while E > 1e-30 and t < 1e4:  #eps=0.01, T=900
    y = np.dot(x,w) # predicciones del modelo
    d = z - y #diferencia entre target y predicción
    dw = Lr*np.dot(x.T,d) #regla delta simplificada
    w += dw #el modelo aprende de los errores. Batch porque usa todo el lote de una entero
    E = np.mean(np.square(d)) #actualizo el error cuadrático medio MSE
    t += 1 #paso de época
    if t%10==0: #cada tanto monitoreo el error
        print(t,E)

print(E)

#¿se parecen m y w? (es decir la posta y la entrenada)
print(m)
print(w)

#¿Generaliza bien el modelo entrenado?
xp = np.random.uniform(-25,25,(100,N+1))
xp[:,-1] = 1 #por el bias
zp = np.dot(xp,m)
yp = np.dot(xp,w)
#queremos comparar zp (el posta) con yp (la predicción)
dp = zp - yp
Ep = np.mean(np.square(dp))
print(Ep) #la métrica de error es el MSE