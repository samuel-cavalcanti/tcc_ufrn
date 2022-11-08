import torch
from torch import nn
import numpy as np

from matplotlib import pyplot as plt


class AlphaLayer(nn.Module):

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()

        
        hidden_nodes = 2 
        bias = True
        self._alpha_layer= nn.Linear(input_dim, hidden_nodes) #[a,b] + [c,d].T
        self.cos_layer =  nn.Linear(hidden_nodes, hidden_nodes,bias=bias)
        self.sin_layer =  nn.Linear(hidden_nodes, output_dim,bias=bias)


    def __forward_alpha(self,theta:torch.Tensor)->torch.Tensor:
        """
            W_i*cos(K*theta + B)*W_j*sin(K*theta + B) 
        """
        alpha= self._alpha_layer(theta)
        """
            W_i*cos(alpha)*W_j*sin(alpha) + b_i*b_j
        """        
        return  self.cos_layer(torch.cos(alpha) ) * self.sin_layer(torch.sin(alpha))

       
    def forward(self, x:torch.Tensor):
        return self.__forward_alpha(x)
       

def main() -> None:
    device = torch.device('cpu')

    with open('dataset_test.npy', 'rb') as f:
        x = np.load(f).astype(np.float32)
        y = np.load(f).astype(np.float32)

    # y = y[:,0].reshape((-1,1))
    kinematic = AlphaLayer(x.shape[1], y.shape[1]).to(device)

    optimizer = torch.optim.RMSprop(
        kinematic.parameters(), lr=1.0e-3)  # 0.000025

    reduction = 'sum'
    mse_loss = nn.MSELoss(reduction=reduction)
    mae_loss = nn.L1Loss(reduction=reduction)

    history = {
        'MSE_train': [],
        'MAE_train': [],

        'MSE_test': [],
        'MAE_test': [],
    }

    size_train = int(x.shape[0]*0.6)

    x_train = x[:size_train]
    x_test = x[size_train:]
    y_train = y[:size_train]
    y_test = y[size_train:]

    x_train = torch.from_numpy(x_train)
    x_test = torch.from_numpy(x_test)

    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)

    batch_size = 500
    number_of_batch = x_train.shape[0] // 500
    rest_of_batch = x_train.shape[0] % 500

    def get_batch_index(batch_index: int):
        if batch_index < rest_of_batch:
            begin = batch_index*(batch_size + 1)
            end = begin + batch_size + 1
        else:
            begin = batch_index*batch_size + rest_of_batch
            end = begin + batch_size

        return begin, end

    for epoch in range(10_000):

        # for i in range(number_of_batch):
        #     begin,end  = get_batch_index(i)
        #     x_train[begin:end]
        pred = kinematic.forward(x_train)
        loss: torch.Tensor = mse_loss(pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            y_pred_train = kinematic.forward(x_train)
            loss_mse_train = mse_loss(y_pred_train, y_train).item()
            loss_mae_train = mae_loss(y_pred_train, y_train).item()

            if loss_mse_train < 10:
                optimizer.param_groups[0]['lr'] =  2.5e-5

            # if loss_mse_train < 0.1:
            #     optimizer.param_groups[0]['lr'] =  2.5e-7

            

            y_pred_test = kinematic.forward(x_test)
            loss_mse_test = mse_loss(y_pred_test, y_test).item()
            loss_mae_test = mae_loss(y_pred_test, y_test).item()

            history['MSE_test'].append(loss_mse_test)
            history['MAE_test'].append(loss_mae_test)

            history['MSE_train'].append(loss_mse_train)
            history['MAE_train'].append(loss_mae_train)

            print(
                f"Train MSE {loss_mse_train:0.4f} Train MAE: {loss_mae_train:0.4f} Test MSE {loss_mse_test:0.4f} MAE {loss_mae_test:0.4f}  epoch: {epoch}")

    y_pred_test: torch.Tensor = kinematic.forward(x_test)

    pred = y_pred_test.detach().numpy()

    true = y_test.detach().numpy()

    print('Pred: ', pred[:10])
    print('True: ', true[:10])

    time_in_epochs = np.arange(len(history['MAE_test']))*10

    # plt.figure(0)
    # plt.plot(time_in_epochs, history['MSE_train'], label='TRAIN MSE')
    # plt.plot(time_in_epochs, history['MSE_test'], 'r-', label='TEST MSE')
    # plt.legend()
    # plt.figure(1)
    # plt.plot(time_in_epochs, history['MAE_train'], label='TRAIN MAE')
    # plt.plot(time_in_epochs, history['MAE_test'], 'r-', label='TEST MAE')
    # plt.legend()

    plt.figure(2)

    plt.scatter(y[:, 0], y[:, 1], label='True')
    plt.scatter(pred[:, 0], pred[:, 1], s=0.5, c='r', label='Predict')
    plt.show()


if __name__ == '__main__':
    main()
