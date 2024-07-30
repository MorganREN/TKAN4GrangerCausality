from kan import *

class cKAN(nn.Module):
    def __init__(self, num_series, lag, hidden, prun_th, grid=3, k=3, seed=42):
        '''
        cKAN model with one KAN per time series

        Args:
          num_series: dimensionality of multivariate time series.
          lag: number of previous time points to use in prediction.
          hidden: list of number of hidden units per layer.
          prun_th: threshold for the prunning.
        '''
        super(cKAN, self).__init__()
        self.p = num_series
        self.lag = lag

        # set up KANs
        self.networks = nn.ModuleList([
            KAN(hidden, grid=grid, k=k, seed=seed, auto_save=False) for _ in range(num_series)
        ])

    def forward(self, x):
        '''
        Forward pass of the cKAN model

        Args:
          x: input tensor of shape (batch_size, num_series, lag)

        Returns:
          y: output tensor of shape (batch_size, num_series)
        '''
        # get the output of each KAN
        y = torch.cat([network(x) for network in self.networks], dim=2)
        return y
    
    def GC(self, threshold):
        '''
        Extract Granger causality matrix from the KANs

        Args:
          threshold: threshold for the prunning.

        Returns:
            GC: Granger causality matrix of shape (num_series, num_series)
        '''
        scores = self.get_scores()
        GC = np.zeros((self.p, self.p))


    def get_scores(self):
        '''
        Get the score of the first layer

        Returns:
            output: Score of each node of the first layer
        '''
        output = []
        for i in range(self.p):
            scores = self.networks[i].node_scores[0].view(self.p, self.lag).sum(dim=1).detach().numpy()
            output.append(scores)

        output = np.array(output)
        return output
    

def rearrange_data(X, num_series, lag):
    '''
    Rearrange the time series, to form a data with lags

    Input:
        X: tensor with shape (1, num_series, T)
        num_series: number of series
        lag: lag
    Output:
        array: tensor with shape (num_series * lag, T-lag)
    '''
    li = []
    for i in range(num_series):
        for j in range(lag):
            li.append(X[0, :, i].detach().numpy()[j:-lag+j])

    # transfer li to tensor
    array = torch.tensor(np.array(li), dtype=torch.float32).T

    return array

def create_dataset(X, Y, device='cpu'):
    '''
    Create a dataset to satisfy the KAN requirement
    '''
    dataset = {}
    dataset['train_input'] = X.to(device)
    dataset['test_input'] = X.to(device)
    dataset['train_label'] = Y.to(device)
    dataset['test_label'] = Y.to(device)

    return dataset


def train_model_ckan(ckan, array, max_iter=20, opt='LBFGS', lamb=0.001, device='cpu'):
    '''
    train the ckan model

    Input:
        ckan: component kan with lots of kan models
        X: time series
        opt: optimizer
        lambd: coefficient for the regularization
        device: used for gpu acceleration
    Output:
        output: the loss of each kan
    '''
    lag = ckan.lag
    num_series = ckan.p
    T = array.shape[1]

    X = rearrange_data(array, num_series, lag)

    train_loss_list = []

    for i in range(num_series):
        Y = array[:, :, i][0, :T-lag].reshape(1, T-lag).T
        dataset = create_dataset(X, Y, device)
        loss_list = ckan.networks[i].fit(dataset, opt=opt, steps=max_iter, lamb=lamb)
        train_loss_list.append(loss_list['train_loss'])
        # print(loss_list)

    temp = [np.array(train_loss_list[i]) for i in range(num_series)]
    output = np.array(temp).T

    return output



