import torch
import reviewer


reviews = reviewer.load( "imdb_labelled.txt")
wl, wd = reviewer.vocabulary( reviews)
seqs = reviewer.sequence( reviews, wd)

P = len(reviews)
N = len(wl)

B = 50
L = 75      # Review Max. Len.


# genero la matriz para hacer los batchs
pads = torch.LongTensor( P, L)
for i in range(P):
    seq = seqs[i][0]
    M = len(seq)
    pads[i] = 0
    if M<L:
        pads[i,-M:] = torch.tensor(seq)
    else:
        pads[i] = torch.tensor(seq[:L])


# genero la matriz para hacer los batchs: parecido pero para labels
labels = torch.tensor([ label for words,label in reviews], dtype=torch.float)



# separo el último cuarto para validación
trn_inputs =   pads[:-N//4]
trn_target = labels[:-N//4]

tst_inputs =   pads[-N//4:]
tst_target = labels[-N//4:]



from torch.utils.data import TensorDataset, DataLoader

trn_data = TensorDataset( trn_inputs, trn_target)
tst_data = TensorDataset( tst_inputs, tst_target)

trn_load = DataLoader( trn_data, shuffle=True, batch_size=B)
tst_load = DataLoader( tst_data, shuffle=True, batch_size=B)



class Sentiment( torch.nn.Module):
    def __init__( _, vocab, embed, context, output=1):
        super().__init__()
        _.isize = vocab
        _.esize = embed
        _.hsize = context
        _.osize = output #por defecto es 1: sigmoidea prob que sea positv
        _.Wemb = torch.nn.Embedding(_.isize, _.esize)
        _.Wrnn = torch.nn.RNN(_.esize, _.hsize, batch_first=True)
        _.Wrnn = torch.nn.GRU(_.esize, _.hsize, batch_first=True) #GRU es un tipo de LSTM
        _.Wo = torch.nn.Linear(_.hsize, _.osize)

    def forward( _, xi): #el input son todos las words de la reseña directamente
        embout = _.Wemb(xi)
        rnnouthides, rnnouthideultimo = _.Wrnn(embout)
        y = torch.sigmoid(_.Wo(rnnouthideultimo)) #¿es igual que .sigmoid()?
        return y.squeeze() #saca todas las dimensiones 1 extras y dejame un vector


model = Sentiment( N, 50, 100, 1)
optim = torch.optim.Adam( model.parameters())
costf = torch.nn.MSELoss()



T = 30
model.train()
for t in range(T):
    E = 0.
    for words, label in trn_load:
        optim.zero_grad()
        
        y = model(words)
        
        error = costf(y, label)
        error.backward()
        optim.step()

        E += error.item()
    print( t, E)


model.eval()
with torch.no_grad():
    r, t = 0, 0
    for words, label in tst_load:
        senti = model(words)
        r += ( (senti>=0.5)==(label==1) ).sum().item()
        t += len(label)
print( "Accuracy:", 100*r/t)
