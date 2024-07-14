import math
import numpy as np
import pandas as pd


total_length=1000

def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    
def generate(sig_to_noise): 
    np.random.seed(0)
    z = [1,1,1,1]
    x=[1,1,1,1]
    #y=[3,3,3,3]
    p = []
    n = [3,3,3,3]
    t1 = [3,3,3,3]
    for i in range(4,total_length+4):
        z.append(math.tanh(z[i-1]+np.random.normal(0,0.01)))
        p.append(z[i]**2+np.random.normal(0,0.05))
        x.append(sigmoid(z[i-2])+np.random.normal(0,0.01))
    
        #term1 = sigmoid(z[i-4])
        #term2 = sigmoid(x[i-2])
        term1 = z[i-4]*z[i-3]
        term2 = x[i-2]*x[i-1]
        t1.append(term1+term2)
        n.append(np.random.normal(0,1))
        #noise = np.random.normal(0,1)
        #alpha = (abs(term1+term2)/sig_to_noise)/abs(noise)
        #y.append(term1+term2+alpha*noise)

    alpha = np.mean(t1)/sig_to_noise
    y = np.add(t1,np.multiply(n,alpha)).tolist()
    x=x[-total_length:]
    y=y[-total_length:]
    p=p[-total_length:]
    z=z[-total_length:]
    df=pd.DataFrame({"x":x,"y":y,"p":p,"z":z})
    return df
#df.to_csv("dataset2_snr26.csv")



