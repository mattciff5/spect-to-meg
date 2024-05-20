import torch
from torch import nn
import torch.nn.functional as F
import seaborn as sns
import tqdm as tqdm

class Ridge:
    def __init__(self, alpha = 0, fit_intercept = True, device=None):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.device = device
       
    def fit(self, X: torch.tensor, y: torch.tensor) -> None:
        X = X.rename(None).to(self.device)
        y = y.rename(None).to(self.device)
        if X.dtype != y.dtype:
            y = y.to(X.dtype)
        assert X.shape[0] == y.shape[0], "Number of X and y rows don't match"
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1, device=self.device), X], dim = 1)
        lhs = X.T @ X 
        rhs = X.T @ y
        if self.alpha == 0:
            self.w, _ = torch.lstsq(rhs, lhs)
        else:
            ridge = self.alpha*torch.eye(lhs.shape[0], device=self.device)
            self.w, _ = torch.lstsq(rhs, lhs + ridge)
            
    def predict(self, X: torch.tensor) -> None:
        X = X.rename(None).to(self.device)
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1, device=self.device), X], dim = 1)
        return X @ self.w
    


class LinearRegression():
    """"My custom regression class"""   
    def __init__(self, C=0.1,lr=1e-3, penalty=None, n_iter=10000, fit_intercept=True):
        self.C = C
        self.lr = lr
        self.history = []
        self.penalty = penalty 
        self.n_iter = n_iter
        self.fit_intercept= fit_intercept
        self.device = torch.device("cuda:{}".format(1) if torch.cuda.is_available() else 'cpu')
        return
    
    def add_intercept(self,x):
        a = torch.ones(x.size()[0],1,device=self.device)
        return torch.cat((x,a),axis=1)
    
    def linreg(self, x):
        """"Linear regression function"""  
        return x @ self.w + self.b           # matrix multiply inputs (x) with the transposed weights (w) and add the intercept (b)
    
    def loss(self, y, y_hat):
        """"Calculate loss"""  
        mse = torch.mean((y-y_hat)**2)            # mean squared error
        
        if self.penalty == 'l2':
            penalty = self.C * torch.sum(self.w**2) # lambda multiplied by the sum of squared weights 
        
        if self.penalty == 'l1':
            penalty = self.C * torch.sum(torch.abs(self.w))    # lambda multiplied by the sum of weights 
        
        if self.penalty == None:
            penalty = 0 
        
        return  mse + penalty 
    
    def cast_to_tensor(self, x):
        # return torch.tensor(x, device=self.device).float()
        return x.to(self.device).float()
        
    def fit(self,x,y):
        """"Fit model"""  
        x = self.cast_to_tensor(x)
        y = self.cast_to_tensor(y)
        
        if self.fit_intercept:
            x = self.add_intercept(x)
        
        self.w = torch.randn((x.size()[1],y.size()[1]), requires_grad=True, device=self.device) #instantiate weights
        self.b = torch.randn(y.size()[1], requires_grad=True, device=self.device)           #instantiate bias

        # print(self.w.shape)
        # print(self.b.shape)
        
        for i in tqdm.trange(self.n_iter):
            y_hat = self.linreg(x)   # make predictions
            # print(y_hat.shape)
            loss = self.loss(y,y_hat) # calculate loss function
            loss.backward()           # backprop
            
            with torch.no_grad(): 
                self.w -= self.w.grad * self.lr #update weights
                self.b -= self.b.grad * self.lr #update bias
                self.w.grad.zero_()
                self.b.grad.zero_()
            
            self.history.append(loss.item())
            
    def predict(self, x):
        """"Predict"""  
        x = self.cast_to_tensor(x)
        
        if self.fit_intercept:
            x = self.add_intercept(x)
        
        return self.linreg(x).detach().cpu().numpy()
    

    
    def plot_history(self):
        """"Plot loss function over time"""  
        return sns.lineplot(x=[i+1 for i in range(len(self.history))],y=self.history).set(xlabel='Iteration', ylabel='Loss',title='History')