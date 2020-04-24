import torch
import reviewer

#from reviewer import *

reviews = reviewer.load( "imdb_labelled.txt")
wordlist, worddict = reviewer.vocabulary( reviews)
seqs = reviewer.sequence( reviews, worddict)

P = len(reviews) #tamaño data set
N = len(wordlist) #tamaño vocabulario
T = 5

# Recordar: el _ es el self

class SRNN( torch.nn.Module):
    def __init__( _, isize, hsize, osize):
        #isize tamaño entrada
        #hsize tamaño capa oculta
        #osize tamaño clasificación final
        #Wc es el recurrente: capa contextual
        super().__init__()
        _.context_size = hsize
        _.Wi = torch.nn.Linear(isize, hsize)
        _.Wc = torch.nn.Linear(hsize, hsize)
        _.Wh = torch.nn.Linear(hsize, isize)
        _.Wo = torch.nn.Linear(hsize, osize)
        # se puede usar Parameter ¿la diferencia qué onda?

    def forward( _, x0, h0):
        h1 = torch.tanh(_.Wi(x0) + _.Wc(h0))
        x1 = _.Wh(h1) #sin activación porque el crossentropy le hace soft max solo
        return x1, h1

    def predict( _, h): #es el que clasifica luego de todas las palabras
        y = _.Wo(h) #sin activación porque el crossentropy le hace soft max solo
        return y

    def context( _, B=1):
        return torch.zeros( B, _.context_size)


def one_hot( p, N):
    assert( 0 <= p < N)
    pat = torch.zeros( 1, N)
    pat[ 0, p] = 1.
    return pat


model = SRNN(N, 200, 2)
optim = torch.optim.SGD( model.parameters(), lr=0.01)
costf = torch.nn.CrossEntropyLoss()

for t in range(T):
    E = 0.
    for b, (words, label) in enumerate(seqs):
        error = 0.
        h = model.context() #es el h vacío del principio
        optim.zero_grad()
        for i in range( len( words[:-1])): #todas las palabras menos la última, porque es el símbolo que clasifica 
            z = torch.tensor( words[i+1] ).view(1)
            x0 = one_hot( words[i], N)
            x1, h = model.forward( x0, h)
            error += costf( x1, z)
        y = model.predict( h)
        error += costf( y, torch.tensor(label))
        error.backward()
        optim.step()
        E += error.item()
        if b%100 == 0:
            print( t, b, error.item())
    print( E)

