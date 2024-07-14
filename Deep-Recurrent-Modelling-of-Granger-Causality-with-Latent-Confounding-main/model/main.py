import torch
import os
from torch import optim
import pandas as pd


os.chdir("C:/Users/tonyz/Desktop/Granger/model")
import gruvae
import train
from data_loader import create_inout_sequences
from functions import t_test
from generate_data import generate


data = pd.read_csv("river.csv")[:1000]
z = data["Kempten"].tolist()
p = data["Lenggries"].tolist()
x = data["Kempten"].tolist()
y = data["Dillingen"].tolist()
  
def obtain_errors(model,trials, train_loader,\
                  val_loader, test_loader, scaler, lam=0,\
                  model_type="full", res="No"):
    torch.manual_seed(0)
    mses = []
    mses_nox = []
    vae_trainer = train.AutoEncoderTrainer(model, optim.Adam,\
                    train_loader, val_loader,\
                    test_loader, scaler, 0.001, lam, model_type)
    for i in range(trials):
        _,mse,mse_nox,_ = vae_trainer.train_val_test_iter(test_loader,\
                                                          "test", res)
        mses.append(mse)
        mses_nox.append(mse_nox)
    return mses,mses_nox


def run_model(z, p, x, y, seq=20):
    torch.manual_seed(0)
    model=gruvae.GRUVAE(1,1,5,5,5,5,2,5,10,3,0.3)
    train_loader, val_loader, test_loader, scaler\
        = create_inout_sequences(z, p, x, y, seq, 800, 100, 100, 10)
    vae_trainer = train.AutoEncoderTrainer(model, optim.Adam,\
                            train_loader, val_loader,\
                            test_loader, scaler, lr=0.001, lam=0.01)
    _,_,model = vae_trainer.train_and_evaluate(50)
    return train_loader, val_loader, test_loader, model, scaler


train_loader, val_loader, test_loader, model, scaler = run_model(z, p, x, y)
full_mses, full_mses_res = obtain_errors(model, 50, train_loader,\
                          val_loader, test_loader, scaler)  
        
print(t_test([i.tolist() for i in full_mses],\
             [i.tolist() for i in full_mses_res],alternative='greater'))
             
def calc_pvalues(sig_to_noise_low="default", sig_to_noise_upp="default"):
    
    if sig_to_noise_low !="default":
        data_low = generate(sig_to_noise_low)
        z_low = data_low["z"].tolist()
        p_low = data_low["p"].tolist()
        x_low = data_low["x"].tolist()
        y_low = data_low["y"].tolist()
        
        train_loader, val_loader, test_loader, model, scaler\
            = run_model(z_low, p_low, x_low, y_low)
        full_mses, full_mses_res = obtain_errors(model, 50, train_loader,\
            val_loader, test_loader, scaler)  
        p_value_low = t_test([i.tolist() for i in full_mses],\
            [i.tolist() for i in full_mses_res],alternative='less')
    
    if sig_to_noise_upp !="default":
        data_upp = generate(sig_to_noise_upp)
        z_upp = data_upp["z"].tolist()
        p_upp = data_upp["p"].tolist()
        x_upp = data_upp["x"].tolist()
        y_upp = data_upp["y"].tolist()
   
        train_loader, val_loader, test_loader, model, scaler\
            = run_model(z_upp, p_upp, x_upp, y_upp)
        full_mses, full_mses_res = obtain_errors(model, 50, train_loader,\
            val_loader, test_loader, scaler)  
        p_value_upp = t_test([i.tolist() for i in full_mses],\
            [i.tolist() for i in full_mses_res],alternative='less')
    
    if  sig_to_noise_low!="default" and sig_to_noise_upp!="default":   
        return p_value_low, p_value_upp
    elif sig_to_noise_low!="default" and sig_to_noise_upp=="default":
        return p_value_low
    elif sig_to_noise_upp!="default" and sig_to_noise_low=="default":
        return p_value_upp
    else:
        pass

    
def vary_alpha(trials, sig_to_noise_low, sig_to_noise_upp, type="bisection",\
               step=0.1):
    p_value_low, p_value_upp = calc_pvalues(sig_to_noise_low,sig_to_noise_upp)    
    print(p_value_low, p_value_upp)
    
    if p_value_low<0.05 and p_value_upp<0.05:
        print("enter new bounds")
        return None
    elif p_value_low>0.05 and p_value_upp>0.05:
        print("enter new bounds")
        return None
    c=0
    if type=="bisection":
        prev_low = 0
        prev_p = 0
        for i in range(trials):
            print(f"\tsig_to_noise_low: {sig_to_noise_low}, p_low: {p_value_low}, sig_to_noise_upp: {sig_to_noise_upp}, p_upp: {p_value_upp}")            
            prev_low = sig_to_noise_low
            prev_p = p_value_low
            if p_value_low>0.05:
                sig_to_noise_low = (sig_to_noise_upp+sig_to_noise_low)/2
                p_value_low = calc_pvalues(sig_to_noise_low=sig_to_noise_low)
            if p_value_low<0.05:
                sig_to_noise_upp = sig_to_noise_low
                sig_to_noise_low = prev_low
                p_value_upp = p_value_low
                p_value_low = prev_p
    else:
        while c==0:
            sig_to_noise_low+=step
            p_value_low = calc_pvalues(sig_to_noise_low=sig_to_noise_low)
            print(f"\tsig_to_noise_low: {sig_to_noise_low}, p_low: {p_value_low}, sig_to_noise_upp: {sig_to_noise_upp}, p_upp: {p_value_upp}")
            if p_value_low<0.05:       
                break


def vary_seq(seqs, z, p, x, y):
    for i in seqs:
        train_loader, val_loader, test_loader, model, scaler\
            = run_model(z, p, x, y, i)
        full_mses, full_mses_res = obtain_errors(model, 50, train_loader,\
                          val_loader, test_loader, scaler) 
        p_val = t_test([i.tolist() for i in full_mses],\
             [i.tolist() for i in full_mses_res],alternative='less')  
        print(f"\tlength: {i}, p_val: {p_val}")

if __name__=="__main__":
    #vary_alpha(1,5, type="e", step=1)
    #vary_alpha(3,4, type="e", step=0.1)
    #vary_alpha(3,3.1, type="e", step=0.01) #3.05
    
    #vary_alpha(8,0.5,10)
    vary_seq([4,6,8,10,12,14,16], z, p, x, y)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

