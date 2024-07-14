import torch
import torch.distributions as distribution
from scipy.stats import ttest_ind 
import numpy as np

def init_network_weights(model):
    if type(model) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(model.weight)
        if model.bias is not None:
            model.bias.data.fill_(0.0)
            
def normal_distribution(mu, var):

    #var = torch.exp(0.5*logvar)
    normal = distribution.Normal(mu, var)
    return normal


def sample_from_normal(mu, var):
    normal = normal_distribution(mu, var)
    sample = normal.rsample()
    return sample

def loss_function(x, y_hat, y_nox_hat, y, p_hat, p, z_prior, z_posterior,\
                  x_hat=None):
    var = 1.0
    batch_loss = 0
    x=x.view(x.size()[-1],-1)
    y=y.view(y.size()[-1],-1)
    p=p.view(p.size()[-1],-1)
    
    for i in range(len(x)):

        if x_hat is not None:
            """
            batch_loss+=(distribution.Normal(x_hat[i], var).\
                log_prob(x[i]).sum()\
                    + distribution.Normal(y_hat[i], var).\
                log_prob(y[i]).sum()\
                    + distribution.Normal(p_hat[i], var).\
                log_prob(p[i]).sum()\
                    + distribution.Normal(y_nox_hat[i], var).\
                log_prob(y[i]).sum())
            """
            batch_loss +=distribution.Normal(y_hat[i], var).\
                log_prob(y[i]).sum()+distribution.Normal(y_nox_hat[i], var).\
                log_prob(y[i]).sum()
        else:
            batch_loss+=(distribution.Normal(y_hat[i], var).\
                log_prob(y[i]).sum()\
                    + distribution.Normal(p_hat[i], var).\
                log_prob(p[i]).sum())
        
        prior_mean,prior_std = z_prior[i]
        post_mean,post_std = z_posterior[i]        
                
        prior = normal_distribution(prior_mean,prior_std)
        posterior = normal_distribution(post_mean,post_std)
        KL = distribution.kl_divergence(posterior,prior).sum()
        #batch_loss-=KL
    return -batch_loss

def t_test(x,y,alternative='both-sided'):
    _, double_p = ttest_ind(x,y,equal_var = False)
    if alternative == 'both-sided':
        pval = double_p
    elif alternative == 'greater':
        if np.mean(x) > np.mean(y):
            pval = double_p/2.
        else:
            pval = 1.0 - double_p/2.
    elif alternative == 'less':
        if np.mean(x) < np.mean(y):
            pval = double_p/2.
        else:
            pval = 1.0 - double_p/2.
    return pval

                
                
