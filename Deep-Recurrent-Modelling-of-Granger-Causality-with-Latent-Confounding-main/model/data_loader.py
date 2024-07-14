import torch
from sklearn.preprocessing import StandardScaler
import torch.utils.data as data_utils
import numpy as np

def create_sequence():
    y=torch.tensor([1,2,3,4,5]).float().view(1,5)
    p=torch.tensor([6,7,8,9,10]).float().view(1,5)
    p=torch.cat((p,p)).view(2,-1)
    x=torch.tensor([11,12,13,14,15]).float().view(1,5)
    data = torch.cat((y,x,p))
    return torch.transpose(data,0,1)


def create_inout_sequences(z, p, x, y, seq_len, train_size,\
                           val_size, test_size, batch_size):
    
    z=torch.tensor(z).view(1,-1)
    p=torch.tensor(p).view(1,-1)
    x=torch.tensor(x).view(1,-1)
    y=torch.tensor(y).view(1,-1)
    
    data = torch.cat((y,x,p))
    data = torch.transpose(data,0,1)
    
    data = np.array(data)
    scaler = StandardScaler()
    scaler = scaler.fit(data[:-(test_size+val_size)])
    data = scaler.transform(data)
    
    # CONCAT DATA IN ORDER OF Y,X,P
    # size of full data must be divisible by seq_length
    # batch size is how many sequences per batch
    
    
    def make_seq(data, seq_len):
        x = []
        y = []
        for i in range(len(data)//seq_len-1):
            i = i*seq_len
            seq = data[i:i+seq_len]
            label = data[i+2:i+seq_len+1] 
            x.append(seq)
            y.append(label)
            
        return torch.tensor(x), torch.tensor(y)
    

    x_train,y_train = make_seq(data[:train_size],seq_len)    
    x_val,y_val = make_seq(data[train_size:train_size+val_size],seq_len)   
    x_test,y_test = make_seq(data[train_size+val_size:],seq_len)   
     
    train_dataset = data_utils.TensorDataset(x_train, y_train)
    train_loader = data_utils.DataLoader(train_dataset,\
                                    batch_size=batch_size, shuffle=False)
    
    val_dataset = data_utils.TensorDataset(x_val, y_val)
    val_loader = data_utils.DataLoader(val_dataset,\
                                batch_size=len(val_dataset), shuffle=False)

    test_dataset = data_utils.TensorDataset(x_test, y_test)
    test_loader = data_utils.DataLoader(test_dataset,\
                                batch_size=len(test_dataset), shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler

 