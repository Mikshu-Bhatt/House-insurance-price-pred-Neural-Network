import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
import bisect
import torch.nn.init as init
import matplotlib.pyplot as plt
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader, Dataset,TensorDataset
df = pd.read_csv('/home/scai/mtech/aib232067/ML/realtor-data.zip.csv')
df = df.drop(columns=['prev_sold_date'])
print(0)
map_list = []
for colmn in df.columns:
    if df[colmn].dtype != 'int64' and df[colmn].dtype != 'float64':
        price_list = []
        lst = []
        mapp = {}
        for val in df[colmn].unique():
            # print(val)
            mean_val = df[df[colmn] == val]['price'].mean()
            index = bisect.bisect_left(price_list,mean_val)
            price_list.insert(index,mean_val)
            lst.insert(index,val)
        for c,val in enumerate(lst):
            mapp[val] = c + 1
        map_list.append(mapp)


        
c = 0
for colmn in df.columns:
    if df[colmn].dtype != 'int64' and df[colmn].dtype != 'float64':
        df[colmn] = df[colmn].map(map_list[c])
        c+=1
print(1)
df.dropna(inplace=True)
df = df.sample(frac=1.0, random_state=42)

X = df[['status', 'bed', 'bath', 'acre_lot', 'city', 'state', 'zip_code',
       'house_size']]


X = torch.tensor(X.values, dtype=torch.float32)

y = df['price']
y = torch.tensor(y.values, dtype=torch.float32)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
l = int(len(X)*0.8)
X_train = X[:l]
X_test = X[l:]
y_train = y[:l]
y_test = y[l:]
# train_loader_X = DataLoader(X_train, batch_size=500, shuffle=False)
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)
# train_loader_y = DataLoader(y_train, batch_size=500, shuffle=False)
# val_loader_X = DataLoader(X_test, batch_size=500, shuffle=False)
# val_loader_y = DataLoader(y_test, batch_size=500, shuffle=False)

class nn_model1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(nn_model1,self).__init__()
        self.mlp_layers = nn.ModuleList([ 
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim,32),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(32,output_dim)
            ])
        for c,layers in enumerate(self.mlp_layers):
            # print(layers,c)
            if c%3 == 0:
                init.xavier_uniform_(layers.weight, gain=1)

    def forward(self,x):
        for layers in self.mlp_layers:
            # print(layers)
            x = layers(x)
        return x
    
ip_dim = len(df.columns) - 1
model1 = nn_model1(ip_dim,64,1)
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model1.parameters(), lr=0.02)

l = 100
l_test = 100
min_t = np.Inf
min_val = np.Inf
t_loss = []
v_loss = []
for epoch in range(150):
    model1.train()

    loss0 = 0
    loss1 = 0
    
    for data_point,label in train_loader:
        # print(data_point)
        output = model1(data_point)
        loss0 += torch.sqrt(criterion(torch.t(output),label))
    t_loss.append(loss0.item()/l)
    optimizer.zero_grad()
    loss0.backward()
    optimizer.step()
    
    model1.eval()
    with torch.no_grad():
        for data_point,label in test_loader:
        # print(data_point)
            output = model1(data_point)
            loss1 += torch.sqrt(criterion(torch.t(output),label))
        v_loss.append(loss1.item()/l)
    if (loss0/l < min_t) and (loss1/l < min_val):
        best_model = model1
        min_t = loss0/l
        min_val = loss1/l
        print(f'updated model')
    print(f'train_loss for {epoch} is ---> {loss0/l} and val loss is {loss1/l_test}')
    if loss0/l < 0.00001:
            
            print(epoch)
            break
    
print(t_loss,v_loss)
plt.figure(figsize=(8, 6))
plt.plot(t_loss, label='Training Loss')
plt.plot(v_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('small_batch')
plt.legend()
plt.show()
plt.savefig('small_batch.png')


torch.save(best_model.state_dict(), '/home/scai/mtech/aib232067/ML/best_model_small_batch.pth')

