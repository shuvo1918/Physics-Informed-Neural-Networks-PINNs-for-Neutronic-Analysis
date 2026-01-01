def main():
  

    import torch
    import os
    import torch.autograd as autograd         # computation graph
    from torch import Tensor                  # tensor node in the computation graph
    import torch.nn as nn                     # neural networks
    import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
    import matplotlib                   #This is for using AGG backend in order to prevent failure of memory while creating the images
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.ticker
    import numpy as np
    import time
    from pyDOE import lhs         #Latin Hypercube Sampling
    import scipy.io
    import pandas as pd

    import random
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)




    class FCN(nn.Module):
        "Defines a fully-connected network in PyTorch"
        def __init__(self,N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
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

    def materials(x):
        d_blanket=torch.add(torch.zeros_like(x),0.4)
        d_core = torch.add(torch.zeros_like(x),0.5)
        s_blanket= torch.add(torch.zeros_like(x),0.0)
        s_core= torch.add(torch.zeros_like(x),1.0)
        sigma_a_blanket=torch.add(torch.zeros_like(x),0.2)
        sigma_a_core =torch.add(torch.zeros_like(x),0.4)
        # sigma_s_blanket= torch.add(torch.zeros_like(x),0.094853)
        # sigma_s_core= torch.add(torch.zeros_like(x),0.089302)
        region= x<=5
        d=torch.where(region,d_core,d_blanket)
        s=torch.where(region,s_core,s_blanket)
        sigma_a=torch.where(region,sigma_a_core,sigma_a_blanket)


        return d, s, sigma_a

    N_b=500
    N_f=1000

    x_left=torch.tensor(0.,requires_grad=True).view(-1,1).to(device)
    x_right=torch.tensor(10.,requires_grad=True).view(-1,1).to(device)
    x_physics_1=np.linspace(0,10,N_f)
    x_physics_add=np.linspace(3,7,500)
    x_physics=np.concatenate((x_physics_1,x_physics_add))

    x_physics=torch.tensor(x_physics,dtype=torch.float32,requires_grad=True).view(-1,1).to(device)
    d,s,sigma_a=materials(x_physics)
    d=d.to(device)
    s=s.to(device)
    sigma_a=sigma_a.to(device)

    test=torch.linspace(0,10,100).to(device).view(-1,1)

    ref=pd.read_csv('./shuvo_fixed.csv')

    torch.manual_seed(123)

    pinn= FCN(1,1,32,4).to(device)
    loss_history = []
    optimiser= torch.optim.Adam(pinn.parameters(),lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=1000, gamma=0.9)
    plot_dir = 'Dis_as1D_32_3_1500_3_7_points_loss100'
    os.makedirs(plot_dir, exist_ok=True)
    error_df = pd.DataFrame(columns=['steps','Loss_phy','Loss','MSE', 'Relative_Error', 'Relative_Percentage_Error'])

    for i in range(15001):
        optimiser.zero_grad()
        u=pinn(x_left)
        dudx=torch.autograd.grad(u,x_left,torch.ones_like(u),create_graph=True)[0]
        loss_lb=torch.mean((((-0.5)*dudx)-torch.zeros_like(dudx))**2)
        u=pinn(x_right)
        dudx=torch.autograd.grad(u,x_right,torch.ones_like(u),create_graph=True)[0]
        loss_rb=torch.mean(((.5*u)+(0.4*dudx))**2)
        u=pinn(x_physics)
        dudx=torch.autograd.grad(u,x_physics,torch.ones_like(u),create_graph=True)[0]
        du2dx2=torch.autograd.grad(dudx,x_physics,torch.ones_like(dudx),create_graph=True)[0]
        loss_phy=torch.mean((-s-(d*du2dx2)+(sigma_a*u))**2)
        loss=loss_lb+loss_rb+loss_phy
        loss.backward()
        optimiser.step()
        if i % 200 == 0:
            plt.clf()  # Clear the current figure
            phi = pinn(test).detach().cpu()
            phid = phi.view(-1, 1)
            test_cpu = test.cpu()
            plt.plot(test_cpu[:, 0], phid, label="Pinn Solution", color="tab:red", alpha=0.6)
            plt.plot(test_cpu[:, 0], ref, label="Reference Solution", color="tab:green", alpha=0.6)
            plt.title(f'No of steps{i}')
            plt.xlabel("x(cm)")
            plt.ylabel("Flux value")
            plt.legend()
            plt.savefig(os.path.join(plot_dir, f'plot_epoch_heat{i}.png'))
            plt.show()
            print(loss_lb,loss_rb,loss_phy)

            # Calculate MSE
            phid=phid.numpy()
            mse_error = ((phid - ref) ** 2).mean()

            # Calculate the relative error
            relative_error = np.sqrt(mse_error) / np.abs(ref).mean()

            # Calculate the relative percentage error
            relative_percentage_error = (np.linalg.norm(phid - ref) / np.linalg.norm(ref)) * 100

            # Append the errors to the DataFrame
            error_df = error_df.append({'steps':i,
                                        'Loss_phy':loss_phy.detach().cpu().numpy(),
                                        'Loss':loss.detach().cpu().numpy(),
            'MSE': mse_error,
            'Relative_Error': relative_error,
            'Relative_Percentage_Error': relative_percentage_error
            }, ignore_index=True)
    error_df.to_csv(f'pinn_errors{plot_dir}.csv', index=False)
    print("Errors saved to pinn_errors.csv")

    print(test_cpu.shape)

    error_df = pd.DataFrame(columns=['MSE', 'Relative_Error', 'Relative_Percentage_Error'])


    # Calculate MSE
    phid=phid.numpy()
    mse_error = ((phid - ref) ** 2).mean()

    # Calculate the relative error
    relative_error = np.sqrt(mse_error) / np.abs(ref).mean()

    # Calculate the relative percentage error
    relative_percentage_error = (np.linalg.norm(phid - ref) / np.linalg.norm(ref)) * 100

    # Append the errors to the DataFrame
    error_df = error_df.append({
    'MSE': mse_error,
    'Relative_Error': relative_error,
    'Relative_Percentage_Error': relative_percentage_error
    }, ignore_index=True)
    error_df.to_csv('pinn_errors.csv', index=False)
    print("Errors saved to pinn_errors.csv")


if __name__ == '__main__':
    main()
