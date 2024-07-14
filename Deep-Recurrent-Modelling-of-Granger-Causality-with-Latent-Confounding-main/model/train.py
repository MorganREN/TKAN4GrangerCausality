import torch
from functions import loss_function
import torch.nn as nn
import numpy as np
import copy

class AutoEncoderTrainer:
    """AutoEncoder Training class."""
    def __init__(self, model, optimizer, train_loader, val_loader,\
                 test_loader, scaler, lr=0.001, lam=0, model_type="full"): 
        """Initialization."""
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "cpu")
        self.lr = lr
        self.optimizer = optimizer(self.model.parameters(), self.lr)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.model_type = model_type
        self.scaler = scaler
        self.lam = lam
        
    def train_val_test_iter(self, d, mode="train", res="No"):
        """one training epoch"""
        mse = nn.MSELoss()
        def operate():
            l1=0
            #meansqs = 0
            for idx, (data,label) in enumerate(d):
                meansq = 0
                meansq_nox = 0
                data = data.to(self.device)
                label = label.to(self.device)
                if mode == "train:":
                    self.optimizer.zero_grad()
                       
                ys = data[:,:,0:1].permute(0,2,1)
                xs = data[:,:,1:2].permute(0,2,1)
                ps = data[:,:,2:].permute(0,2,1)

                y = label[:,:,0:1].permute(0,2,1)
                x = label[:,:,1:2].permute(0,2,1)
                p = label[:,:,2:].permute(0,2,1)
        
                l2=0
                

                for i in range(len(y)):
                    if self.model_type == "full":
                        _,z_prior,z_post,_,_,_,_,x_hats,\
                            y_hats,p_hats,y_nox_hats = self.model(ys[i],ps[i],xs[i])
                        if res!="No":
                            _,_,_,_,_,_,_,_,\
                            y_hats,_ = self.model(ys[i],ps[i],xs[i],res="yes")
                        l2 += loss_function(x[i],y_hats,y_nox_hats,\
                            y[i],p_hats,p[i],z_prior,z_post,x_hats)/len(y_hats)

                    
                    else:
                        _,z_prior,z_post,_,_,_,_,x_hats,\
                            y_hats,p_hats = self.model(ys[i],ps[i])
                        l2 += loss_function(x[i],y_hats,\
                            y[i],p_hats,p[i],z_prior,z_post)/len(y_hats)
                    meansq += mse(torch.stack(y_hats).view(len(y_hats),-1),\
                                 y[i].view(len(y_hats),-1))/len(y_hats)
                    meansq_nox += mse(torch.stack(y_nox_hats).view(len(y_nox_hats),-1),\
                                 y[i].view(len(y_nox_hats),-1))/len(y_nox_hats)
                #l2 += self.lam*(torch.norm(self.model.y_emission2.layers[0][0].weight.transpose(1,0)[0])\
                 #               +torch.norm(self.model.y_emission2.layers[0][0].weight.transpose(1,0)[1:]))
                #sb = self.model.y_emission2.layers[0][0].weight.transpose(1,0)
                #l2 = l2/len(y)
                #l2 += self.lam*self.model.n_units_mlp**0.5*sum([torch.norm(sb[i])/torch.norm(sb[i])**2 for i in range(len(sb))])
                if mode == "train":
                    l2.backward()
                    self.optimizer.step()  
                l1 += l2
                #meansqs += meansq
            copymodel = copy.deepcopy(self.model)
            return l1/len(d), meansq, meansq_nox, copymodel
        
        if mode == "train":
            self.model.train()
            self.model.to(self.device)
            l,mse,mse_nox,model = operate()
        else:
            self.model.eval()  
            with torch.no_grad():
                l,mse,mse_nox,model = operate()
        return l,mse,mse_nox,model
        
        """
        copymodel = copy.deepcopy(self.model)
        """
            
    def train_and_evaluate(self, epochs):
        val_losses = []
        mses = []
        mses_nox = []
        models = []
        for epoch in range(epochs):
            train_loss,_,_,_ = self.train_val_test_iter(self.train_loader)
            val_loss,vmse,vmse_nox,model = self.train_val_test_iter(self.val_loader, "val")
            test_loss,mse,mse_nox,_ = self.train_val_test_iter(self.test_loader, "test")
            val_losses.append(val_loss)
            mses.append(mse)
            mses_nox.append(mse_nox)
            models.append(model)
            print(f"\tEpoch: {epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Test MSE: {mse:.4f}, Test MSE nox: {mse_nox: .4f}")
            #if epoch>3:
             #   if val_loss > val_losses[-2] and val_losses[-2] > val_losses[-3]:
              #      return val_losses[epoch-2].item(), mses[epoch-2].item(), mses_nox[epoch-2].item(), models[epoch-2]
                    
            
        idx = np.argmin(val_losses)
        return val_losses[idx].item(), mses[idx].item(), models[idx]
        

                

            
        
        
        

    
    
    
    
  