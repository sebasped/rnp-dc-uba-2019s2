#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 18:36:55 2019

@author: sebas
"""

import numpy as np

m = np.array([[3,-9,0,5],[2,-5,-3,1],[-1,5,8,4]])

#a = np.zeros((1,3))
#a[:] = [-1,1,2]
a = np.array([[-1,1,2]])

b = np.zeros((1,4))
b[:] = [-4,2,1,-1]

print(np.dot(a,m))

print(np.dot(a.T,b))

print(np.dot(m,b.T))

print(np.dot(a,a.T))



from matplotlib import pyplot as plt

x = np.linspace(-2,2,100)
y1 = 1/(1+np.exp(-x))
y2 = np.tanh(x)
y3 = np.sign(x)
y4 = (1-y1)*y1
y5 = 1-y2**2

g1 = plt.plot(x,y1,x,y2,x,y3)
plt.show()
#g2 = plt.plot(x,y2)
#g3 = plt.plot(x,y3)
g4 = plt.plot(x,y5)
#plt.show()
g5 = plt.plot(x,y4)
plt.show()