import pandas as pd
import numpy as np
import torch
from torch.nn import functional as F
from sklearn import preprocessing
from torch.autograd import Variable

test = pd.read_csv('test-A/in.tsv', sep = "\t", header = None)
train = pd.read_csv('train/train.tsv', sep = '\t', header = None)
dev = pd.read_csv('dev-0/in.tsv', sep = '\t', header = None)

def preprocess(set):
    vowels = ['a', 'e', 'i', 'o', 'u', 'a', 'e', 'y']
    var1 = []
    var2 = []
    var4 = []
    var5 = []
    var6 = []
    var7 = []
    varbls = []

    for row in set.iloc[:,-2]:
        row = str(row)
        if (row[-5:] == 'kopia'):
            var4.append(1)
        else:
            var4.append(0)
        if (row[:1] == 'y') or (row[:4] == 'cent') or (row[:2] in ['ae', 'ah', 'ai', 'aj', 'bc', 'bd', 'bf', 'bg', 'bh', 'bk', 'ck', 'dc', 'dd', 'ez', 'fó', 'fp', 'fq',  'hn', 'hm', 'ie', 'ig', 'ii', 'ik', 'il', 'tz', 'tv', 'uy']):
            var6.append(0)
        else:
            var6.append(1)
        var7.append([len(row)])

    for row in set.iloc[:,-2]:
        vw = 0
        cs = 0
        row = str(row)
        for i in list(row):
            if i in vowels:
                vw += 1
                var5.append(1)
            elif (i in ['q', 'x', 'ü']):
                var5.append(0)
                cs += 1
            else:
                var5.append(1)
                cs += 1
        ratio = vw/(cs + vw)        
        var1.append( min(3*(ratio), 1) - max(0, 1.5*(ratio - 0.333)) )

    for i in set.iloc[:,-1]:
        if (i < 165):
            var2.append([i])
        elif (i > 2000):
            var2.append([1])
        else:
            var2.append([165])
    mm_scaler = preprocessing.MinMaxScaler()
    var2 = mm_scaler.fit_transform(var2)
    var7 = mm_scaler.fit_transform(var7)

    for i in range(len(set.iloc[:,-1])):
        varbls.append([var1[i], var2[i], var4[i], var5[i], var6[i], var7[i]])

    x1 = torch.tensor(var1, dtype=torch.float) # how "normal" is ratio of vowels to consonants
    x2 = torch.tensor(var2, dtype=torch.float) # frequency
    x4 = torch.tensor(var4, dtype=torch.float) # ends with "kopia"
    x5 = torch.tensor(var5, dtype=torch.float) # lacks non-Polish letters
    x6 = torch.tensor(var6, dtype=torch.float) # lacks some weird groups of letters
    x7 = torch.tensor(var7, dtype=torch.float) # length of words
    xData = torch.tensor(varbls)
    
    return xData

var3 = []
for row in train[0]:
    if (row == 0):
        var3.append([0])
    elif (row == 1):
        var3.append([1])
yData = torch.tensor(train[0], dtype = torch.double)

input_size = 6
num_epochs = 10000
learning_rate = 0.002

W1 = torch.rand([6,16],dtype=torch.double, requires_grad=True)
b1 = torch.rand(16,dtype=torch.float, requires_grad=True)
W2  = torch.rand(16,dtype=torch.double, requires_grad=True)
b2  = torch.rand(1,dtype=torch.float, requires_grad=True)
W3  = torch.rand(16,dtype=torch.double, requires_grad=True)
b3  = torch.rand(1,dtype=torch.float, requires_grad=True)

def model(x):
    h = torch.relu(x @ W1 + b1)
    y = torch.sigmoid(h @ W3 + b3)
    return y
criterion = torch.nn.MSELoss(reduction='sum')
                            
for epoch in range(num_epochs):
    y_pred = model(preprocess(train))
    loss = (criterion(y_pred, yData))
    print(loss)
    loss.backward()
    if (loss.item() < 42560):
        learning_rate = 0.0001
    with torch.no_grad():
        W1 = W1 - learning_rate * W1.grad
        b1 = b1 - learning_rate * b1.grad
        W3 = W3 - learning_rate * W3.grad
        b3 = b3 - learning_rate * b3.grad
    W1.requires_grad_(True)
    b1.requires_grad_(True)
    W3.requires_grad_(True)
    b3.requires_grad_(True)
    
    if (loss.item() < 1610):
        break;
file = open("dev-0/out.tsv","w+")
new_x = preprocess(dev)  
y_pred = model(new_x)

for j in range(0, len(y_pred)):
    if y_pred[j].data.item() < 0.2:
        file.write("0\n")
    else:
        file.write("1\n")

file.close()

file = open("test-A/out.tsv","w+")
new_x = preprocess(test)  
y_pred = model(new_x)

for j in range(0, len(y_pred)):
    if y_pred[j].data.item() < 0.2:
        file.write("0\n")
    else:
        file.write("1\n")
        
file.close()