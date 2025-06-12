import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import seaborn as sns
from torchinfo import summary
from torcheval.metrics.aggregation.auc import AUC
from sklearn.model_selection import train_test_split
from torchmetrics import AUROC
from models.logreg import LogReg

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


n_sample=1000

def calculate_auc(yp, y):
    auroc = AUROC(task="binary")
    return(auroc(torch.sigmoid(yp), y))

#Generate data

df = pd.DataFrame(np.random.randint(0,n_sample,size=(n_sample, 4)), columns=list('ABCD'))
df = df.assign(y=np.random.randint(0,2,size=(n_sample, 1)))
df = df.assign(X1=np.random.normal(10*df.loc[:,'y'], 2*(df.loc[:,'y']+1)))
#sns.kdeplot(data=df, x="X1", hue="y")

x_data = torch.tensor(df.drop('y', axis=1).values, dtype=torch.float32)
y_data = torch.tensor(df.loc[:, 'y'], dtype=torch.long)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, shuffle=True)



model = LogReg(x_train.shape[1])
loss_function = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

summary(model, input_size=x_train.shape)

n_epoch=10000
train_losses=[]
test_losses=[]
test_aucs=[]
train_aucs=[]

for epoch in range(n_epoch):
    model.train()
    # Forward propagation (predicting train data) #a
    train_preds = model(x_train).squeeze()
    train_loss  = loss_function(train_preds, y_train.float())#.float().unsqueeze(1))
    
    # Predicting test data #b
    with torch.no_grad():
        test_preds = model(x_test).squeeze()
        test_loss  = loss_function(test_preds, y_test.float())#.float().unsqueeze(1))
        
    # Calculate accuracy #c
    train_auc = calculate_auc(train_preds, y_train)
    test_auc  = calculate_auc(test_preds, y_test)
    
    # Backward propagation #d
    optimizer.zero_grad()
    train_loss.backward()

    # Gradient descent step #e
    optimizer.step()
    model.eval() 

    # Store training history #f
    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())
    train_aucs.append(train_auc.item())
    test_aucs.append(test_auc.item())
    
    # Print training data #g
    if epoch%100==0:
        print(f'Epoch: {epoch} \t|' \
            f' Train loss: {np.round(train_loss.item(),3)} \t|' \
            f' Test loss: {np.round(test_loss.item(),3)} \t|' \
            f' Train auc: {np.round(train_auc.item(),2)} \t|' \
            f' Test auc: {np.round(test_auc.item(),2)}')