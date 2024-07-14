import torch
import torch.nn as nn
from multi_layer_perceptron import MLP
from functions import sample_from_normal

class GRUVAE(nn.Module):

    def __init__(self, x_bool, p_size, gru_x_size, gru_y_size,\
                 gru_z_size, gru_p_size, z_size,\
                     inf_gru_size, n_units_mlp, n_mlp, dropout=None):
        super().__init__()

        # Attributes
        self.x_bool = x_bool
        self.p_size = p_size
        self.gru_x_size = gru_x_size 
        self.gru_y_size = gru_y_size
        self.gru_z_size = gru_z_size
        self.gru_p_size = gru_p_size
        self.inf_gru_size = inf_gru_size
        self.z_size = z_size
        self.n_units_mlp = n_units_mlp #number of hidden nodes in mlp
        self.n_mlp = n_mlp #number of mlp layers
        self.dropout = dropout # dropout probability of mlp
        
        # Prior of Z
        self.gru_z_prior = nn.GRU(self.z_size, self.gru_z_size, 1) #rnn for z
        self.z_prior = MLP(self.gru_z_size, self.n_units_mlp, self.z_size*2,\
                           self.n_mlp, dropout = self.dropout) #mlp for z prior
        
        # RNNs for x, y and p
        self.gru_x = nn.GRU(1, self.gru_x_size, 1)
        self.gru_y = nn.GRU(1, self.gru_y_size, 1)
        self.gru_p = nn.GRU(self.p_size, self.gru_p_size, 1)
        
        # Posterior of Z
        #self.gru_z_post = nn.GRU(self.p_size+self.x_bool+1,\
         #                        self.inf_gru_size, 1)
        self.gru_z_post = nn.GRU(self.p_size,\
                                 self.inf_gru_size, 1)
        
        self.z_post = nn.Linear(self.inf_gru_size, self.z_size * 2)
                
        # emission model        
        #self.y_emission = MLP(self.gru_z_size+self.gru_y_size+self.x_bool*self.gru_x_size, self.n_units_mlp,\
         #                         1, self.n_mlp, dropout = self.dropout)
            
        #self.y_emission2 = MLP(self.gru_z_size+self.gru_y_size, self.n_units_mlp,\
         #                         1, self.n_mlp, dropout = self.dropout)
        
        self.y_emission = MLP(self.gru_z_size+self.gru_y_size, self.n_units_mlp,\
                                  1, self.n_mlp, dropout = self.dropout)
            
        self.y_emission2 = MLP(1+self.gru_x_size*self.x_bool, self.n_units_mlp,\
                                  1, self.n_mlp, dropout = self.dropout)
            
        self.x_emission = MLP(self.gru_z_size+self.gru_x_size,\
                              self.n_units_mlp, 1, self.n_mlp,\
                                  dropout = self.dropout)
        self.p_emission = MLP(self.gru_z_size+self.gru_p_size,\
                              self.n_units_mlp, self.p_size,\
                                  self.n_mlp, dropout = self.dropout)
        self.softplus = nn.Softplus()
        

    def z_inference(self, g_t):
        # given g_t return z posterior
        z_mean_var = self.z_post(g_t)
        z_mean_var = torch.chunk(z_mean_var, 2, dim=-1)
        z_std = self.softplus(z_mean_var[1]) 
        z = sample_from_normal(z_mean_var[0],z_std)
        return z, (z_mean_var[0],z_std)
    
    
    def reconstruct(self, h_z, h_y, h_p, h_x=None, res="No"):
        # map hidden states to target variables
        p = self.p_emission(torch.cat((h_z,h_p),1))
        if h_x is not None:
            x = self.x_emission(torch.cat((h_z,h_x),1))
            if res!="No":
                h_x = torch.zeros_like(h_x)
            y_nox = self.y_emission(torch.cat((h_z,h_y),1))
            y = self.y_emission2(torch.cat((y_nox,h_x),1))
            #y = self.y_emission2(torch.cat((y,h_x),1))
            return (y, x, p, y_nox)
        else:
            y = self.y_emission(torch.cat((h_z,h_y),1))
            #y = self.y_emission2(y)
            return (y, p)
    

    def run_GRUVAE(self, y, p, x=None, res="No"):

        zs, z_post, z_prior = [], [], [] 
        x_hats, y_hats, p_hats, y_nox_hats = [], [], [],[]
        self.gru_y.flatten_parameters()
        self.gru_p.flatten_parameters()
        h_y = self.gru_y(y.view(y.size()[-1],-1,1))[0]
        h_p = self.gru_p(p.reshape(p.size()[-1],-1,self.p_size))[0]
        
        if x is not None:
            self.gru_x.flatten_parameters()
            h_x = self.gru_x(x.view(x.size()[-1],1,1))[0]
            combined = torch.cat((x,y,p))
        else:
            combined = torch.cat((y,p))
            h_x = None

        #g_t = self.gru_z_post(combined.view(combined.size()[-1],1,-1))[0]
        self.gru_z_post.flatten_parameters()
        g_t = self.gru_z_post(p.view(p.size()[-1],1,-1))[0]    
        h_z_prev = torch.zeros(self.gru_z_size).view(1,-1).to("cuda")
        

        for t in range(1,y.size()[-1]):

            # paramterise prior z distribution
            zt_params = self.z_prior(h_z_prev)   
            zt_params = torch.chunk(zt_params, 2, dim=-1)
            z_prior.append((zt_params[0],self.softplus(zt_params[1])))
            
            # paramterise posterior z distribution 
            z, z_posterior = self.z_inference(g_t[t])
            z_post.append(z_posterior)
            zs.append(z) 
            self.gru_z_prior.flatten_parameters()
            h_z = self.gru_z_prior(torch.cat(zs).view(len(zs),-1,\
                                                         self.z_size))[0]

            
            # compute reconstructions
            if x is not None:
                x_hats.append(self.reconstruct(h_z[-1], h_y[t],\
                                               h_p[t], h_x[t],res)[1]) 
                y_hats.append(self.reconstruct(h_z[-1], h_y[t],\
                                               h_p[t], h_x[t],res)[0])
                p_hats.append(self.reconstruct(h_z[-1], h_y[t],\
                                               h_p[t], h_x[t],res)[-2])
                y_nox_hats.append(self.reconstruct(h_z[-1], h_y[t],\
                                               h_p[t], h_x[t],res)[-1])
            else:
                y_hats.append(self.reconstruct(h_z[-1],\
                                               h_y[t], h_p[t],h_x,res)[0])
                p_hats.append(self.reconstruct(h_z[-1], h_y[t],\
                                               h_p[t],h_x,res)[-1])
            h_z_prev = h_z[-1]
        return zs, z_prior, z_post, h_z, h_x, h_y, h_p, x_hats, y_hats, p_hats, y_nox_hats
      
    def forward(self, y, p, x=None, res="No"):
        zs, z_prior, z_post, h_z, h_x, h_y, h_p, x_hats,\
            y_hats, p_hats, y_nox_hats = self.run_GRUVAE(y, p, x, res)
        return zs, z_prior, z_post, h_z, h_x, h_y, h_p, x_hats, y_hats, p_hats, y_nox_hats
        

# input of shape dimesnion, seq-len
#gruvae=GRUVAE(1,1,3,3,3,3,3,3,3,3,0.1)
#y=torch.tensor([1,2,3,4,5]).float().view(1,5)
#p=torch.tensor([6,7,8,9,10]).float().view(1,5)
#p=torch.cat((p,p)).view(2,-1)
#x=torch.tensor([11,12,13,14,15]).float().view(1,5)
#gruvae.run_GRUVAE(y,p,x)
#zs, z_prior, z_post, h_z, h_x, h_y, h_p, x_hats, y_hats, p_hats=gruvae.run_GRUVAE(y,p,x)
#gruvae(y,p,x)



#y = val_loader.dataset.tensors[0].view(-1,3)[:,0].view(-1,1,5)
#x = val_loader.dataset.tensors[0].view(-1,3)[:,1].view(-1,1,5)
#p = val_loader.dataset.tensors[0].view(-1,3)[:,2].view(-1,1,5)
#gruvae(y,p,x)


