import torch
# t https://github.com/ZixuanJiang/pre-rmsnorm-transformer.



class LayerNorm:
    def __init__(self, d, epsilon=1e-5):
        self.d = d
        self.epsilon = epsilon

    def forward(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(var + self.epsilon)


class RMSNorm:
    ### Recenter before passing into blocks
    ### RMSNorm before postprocess 
    def __init__(self, d, epsilon=1e-5):
        self.d = d
        self.epsilon = epsilon

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.epsilon)
        return x / rms


class CRMSNorm:
    ## 
    ## CRMSNorm before preprocess
    def __init__(self, d, epsilon=1e-5):
        self.d = d
        self.epsilon = epsilon

    def forward(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        var = torch.mean(x**2, dim=-1, keepdim=True) - mu**2
        return (x - mu) / torch.sqrt(var + self.epsilon)


if __name__ == "__main__":
    import torch
    
    d = 10
    epilson = 0.01

    x = torch.rand(10)  # converts the 1-element tensor to a Python float

    ones = torch.ones(10)

    print(ones.shape)
    
    mu = lambda x: ones.T @ x / d 


    out = (x - mu(x) @ ones )/torch.sqrt( torch.norm(x)^2 / d - mu(x)^2 + epilson) 