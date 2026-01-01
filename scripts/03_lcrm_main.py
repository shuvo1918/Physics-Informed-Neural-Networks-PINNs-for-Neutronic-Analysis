def main():
    """LCRM Main (2D PINN)

    Plain-python version of the original `LCRM Main.py` used in the thesis work.
    Kept close to the original style.
    """


    import torch
    import os
    import torch.autograd as autograd         # computation graph
    from torch import Tensor                  # tensor node in the computation graph
    import torch.nn as nn                     # neural networks
    import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
    import matplotlib
    matplotlib.use('Agg')                     #This is for using AGG backend in order to prevent failure of memory while creating the images
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.ticker

    import numpy as np
    import time
    from pyDOE import lhs         #Latin Hypercube Sampling
    import scipy.io

    #Set default dtype to float32
    torch.set_default_dtype(torch.float)

    #PyTorch random number generator
    torch.manual_seed(1234)

    # Random number generators in other libraries
    np.random.seed(1234)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)

    if device == 'cuda':
        print(torch.cuda.get_device_name())
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
    def materials(x,y):
        d_blanket=torch.add(torch.zeros_like(x),2.094999864)
        d_core = torch.add(torch.zeros_like(x),2.200801092)
        s_blanket= torch.add(torch.zeros_like(x),0.00214231)
        s_core= torch.add(torch.zeros_like(x),0.01048083)
        sigma_a_blanket=torch.add(torch.zeros_like(x),0.064256)
        sigma_a_core =torch.add(torch.zeros_like(x),0.062158)
        sigma_s_blanket= torch.add(torch.zeros_like(x),0.094853)
        sigma_s_core= torch.add(torch.zeros_like(x),0.089302)
        c1= ((x >= 10) & (x <= 30)) & ((y >= 10) & (y <= 30))
        c2= (x >= 70) & (x <= 90) & (y >= 70) & (y <= 90)
        region= c1|c2
        d=torch.where(region,d_core,d_blanket)
        s=torch.where(region,s_core,s_blanket)
        sigma_a=torch.where(region,sigma_a_core,sigma_a_blanket)
        sigma_s=torch.where(region,sigma_s_core,sigma_a_blanket)

        return d, s, sigma_a,sigma_s

    N_b=1000
    N_f=5000
    Test_point=1000

    X_lb = np.array([0.0, 0.0]) + np.array([100.0, 0.0]) * lhs(2,  int(N_b))
    X_ub = np.array([0.0, 100.0]) + np.array([100.0, 0.0]) * lhs(2,  int(N_b))
    Y_lb = np.array([0.0, 0.0]) + np.array([0.0, 100.0]) * lhs(2,  int(N_b))
    Y_rb = np.array([100.0, 0.0]) + np.array([0.0, 100.0]) * lhs(2,  int(N_b))
    X_f  = np.array([0.0,0.0])+ np.array([100.0,100.0])*lhs(2,int(N_f))
    X_test = np.array([0.0,0.0])+ np.array([100.0,100.0])*lhs(2,int(Test_point))


    torch.manual_seed(123)

    pinn= FCN(2,1,40,8).to(device)
    lower=torch.tensor(X_lb, dtype=torch.float32, requires_grad=True).to(device)
    upper=torch.tensor(X_ub, dtype=torch.float32, requires_grad=True).to(device)
    left = torch.tensor(Y_lb, dtype=torch.float32, requires_grad=True).to(device)
    right=torch.tensor(Y_rb, dtype=torch.float32, requires_grad=True).to(device)
    train=torch.tensor(X_f, dtype=torch.float32, requires_grad=True).to(device)
    d,s,sig_a,sig_s=materials(train[:,0],train[:,1])
    d=d.to(device)
    s=s.to(device)
    sig_a=sig_a.to(device)
    sig_s=sig_s.to(device)
    #test=torch.tensor(X_test,dtype=torch.float32)
    x=np.linspace(0,100,100)
    y=np.linspace(0,100,100)
    X,Y= np.meshgrid(x,y)
    test= torch.tensor(np.vstack([X.flatten(), Y.flatten()]).T, dtype=torch.float32).cpu()
    test=test.to(device)
    optimiser= torch.optim.Adam(pinn.parameters(),lr=1e-3)
    plot_dir = '40_8_1000_5000_1000'
    os.makedirs(plot_dir, exist_ok=True)

    for i in range(34001):
        optimiser.zero_grad()

        u=pinn(lower)
        dudy=torch.autograd.grad(u,lower,torch.ones_like(u),create_graph=True)[0]
        loss_low=torch.mean((.5*u-(2.094999864*dudy[:,[1]]))**2)
        u=pinn(upper)
        dudy=torch.autograd.grad(u,upper,torch.ones_like(u),create_graph=True)[0]
        loss_ub=torch.mean((.5*u+(2.094999864*dudy[:,[1]]))**2)
        u=pinn(left)
        dudx=torch.autograd.grad(u,left,torch.ones_like(u),create_graph=True)[0]
        loss_lb=torch.mean((.5*u-(2.094999864*dudx[:,[0]]))**2)
        u=pinn(right)
        dudx=torch.autograd.grad(u,right,torch.ones_like(u),create_graph=True)[0]
        loss_rb=torch.mean((.5*u+(2.094999864*dudx[:,[0]]))**2)
        #define Physics loss
        # Compute the gradient (first derivatives) of u with respect to x and y
        u=pinn(train)
        grad_u= torch.autograd.grad(u,train,torch.ones_like(u),create_graph=True)[0]
        grad_u_x = grad_u[:,0]
        grad_u_y = grad_u[:,1]

    # Now we need to compute the second derivatives
        grad_u_xx=torch.autograd.grad(grad_u_x, train, grad_outputs=torch.ones_like(grad_u_x), create_graph=True)[0]
        grad_u_yy=torch.autograd.grad(grad_u_y, train, grad_outputs=torch.ones_like(grad_u_y), create_graph=True)[0]
        d2u_dx2 =grad_u_xx[:,0]   # Second derivative with respect to x
        d2u_dy2 = grad_u_yy[:,1]  # Second derivative with respect to y
        loss_phy=torch.mean((s+(d*d2u_dx2+d*d2u_dy2)-(sig_a*u))**2)
        loss=loss_low+loss_ub+loss_lb+loss_rb+loss_phy
        loss.backward()
        optimiser.step()

        if i %100==0:
            #print(test.shape)
            phi=pinn(test).detach().cpu()
            phid=phi.view(X.shape)
            print(loss_low,loss_ub, loss_lb , loss_rb,loss_phy,loss)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            #phi=phi.reshape(len(x),len(x))
            surf = ax.plot_surface(X,Y,phid,cmap='viridis')
            # Add labels and title
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Flux')
            ax.set_title(f'Training Step{i}')
    

            # Add colorbar
            fig.colorbar(surf, ax=ax, label='Flux')
            fig.savefig(os.path.join(plot_dir, f'plot_epoch_{i}.png'))
        

            # Show the plot
            #plt.show()
            plt.figure(figsize=(10, 8))
            plt.contourf(X,Y,phid, levels=100)
            plt.colorbar()
            #plt.title('Neutron Flux')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Training Step {i}')
            plt.savefig(os.path.join(plot_dir, f'plot_epoch_heat{i}.png'))
            #plt.show()
            plt.close()


if __name__ == '__main__':
    main()
