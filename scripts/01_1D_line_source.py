def main():
   

    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt

    # ---- reproducibility / device ----
    import random
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    def true_value(x):
        d=0.5
        sigma_a=0.4
        s=1
        L=np.sqrt(d/sigma_a)
        exp=torch.exp(-x/L)
        func=(s*L)/(2*d)*exp
        return func

    torch.manual_seed(123)
    x=torch.linspace(1,10,100).view(-1,1)
    result=true_value(x)
    print(result)
    plt.plot(result)


    class FCN(nn.Module):
        "Defines a fully-connected network in PyTorch"
        def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
            super().__init__()
            activation = nn.Tanh
            self.fcs = nn.Sequential(*[
                            nn.Linear(N_INPUT, N_HIDDEN),
                            activation()])
            self.fch = nn.Sequential(*[
                            nn.Sequential(*[
                                nn.Linear(N_HIDDEN, N_HIDDEN),
                                activation()]) for _ in range(N_LAYERS-1)])
            self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        def forward(self, x):
            x = self.fcs(x)
            x = self.fch(x)
            x = self.fce(x)
            return x

    torch.manual_seed(123)

    pinn= FCN(1,1,32,6)

    x_boundary=torch.tensor(0.).view(-1,1).requires_grad_(True)
    x_train= torch.linspace(0,100,200).view(-1,1).requires_grad_(True)
    d=0.5
    sigma_a=0.4
    s=1
    L=np.sqrt(d/sigma_a)
    optimiser=torch.optim.Adam(pinn.parameters(),lr=1e-4)
    for i in range(30001):
        optimiser.zero_grad()
        u=pinn(x_boundary)
        dudx=torch.autograd.grad(u,x_boundary,torch.ones_like(u),create_graph=True)[0]
        loss1=torch.squeeze((u-1))**2

        u=pinn(x_train)
        dudx=torch.autograd.grad(u,x_train,torch.ones_like(u),create_graph=True)[0]
        d2udx2=torch.autograd.grad(dudx,x_train,torch.ones_like(u),create_graph=True)[0]

        loss2=torch.mean((d2udx2-(u/(L**2)))**2)

        loss=loss1+ loss2
        loss.backward()
        optimiser.step()

        if i%5000 ==0:
            u=pinn(x).detach()
            print(loss1,loss2)
            plt.plot(x,u,label="PINN",color="tab:red",alpha=0.6)
            plt.plot(x,result,label="Exact Solution",color="tab:green",alpha=0.6)
            plt.show()


if __name__ == '__main__':
    main()
