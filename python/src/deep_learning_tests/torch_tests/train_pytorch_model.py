import torch
from torch import nn
import numpy as np
from datasets import load_memory_dataset
from constants import KINEMATIC_TORCH_PATH
from matplotlib import pyplot as plt
from test_torch_model_sim_data import FourierLayer


class RobotDynamicModel(nn.Module):

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()

        hidden_nodes = 16
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_nodes),
            nn.ELU(),
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.ELU(),
            nn.Linear(hidden_nodes, 4),
            nn.ELU(),
            nn.Linear(4, output_dim),
        )

    def forward(self, x):
        return self.layers.forward(x)



def main() -> None:
    device = torch.device('cpu')

    x, y = load_memory_dataset()

    theta = x[:, -1]
    x_theta = np.cos(theta).reshape((-1,1))
    y_theta = np.sin(theta).reshape((-1,1))

    vx = x[:, 0].reshape((-1,1))
    vy = x[:, 1].reshape((-1,1))
    vtheta = x[:, 2].reshape((-1,1))

    
 
 
    x = np.hstack((vx, vy, vtheta, x_theta, y_theta))

   

    index = np.arange(x.shape[0])
    np.random.shuffle(index)
    x = x[index]
    y = y[index]

    train_size = int(x.shape[0]*0.60)

    x_train = x[:train_size]
    y_train = y[:train_size]

    x_test = x[train_size:]
    y_test = y[train_size:]

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    kinematic = FourierLayer(x.shape[1], y.shape[1]).to(device)

    # kinematic.load_state_dict(torch.load(KINEMATIC_TORCH_PATH))

    optimizer = torch.optim.RMSprop(
        kinematic.parameters(), lr=0.1e-3)  # 0.000025

    reduction = 'sum'
    mse_loss = nn.MSELoss(reduction=reduction)
    mae_loss = nn.L1Loss(reduction=reduction)

    history = {
        'MSE_train': [],
        'MAE_train': [],

        'MSE_test': [],
        'MAE_test': [],
    }

    epoch = 0
    min_loss = 20
    while min_loss > 0.01 and epoch < 1_000_000:
        pred = kinematic.forward(x_train)
        loss: torch.Tensor = mse_loss(pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:

            loss_mse_train = loss.item()
            loss_mae_train = mae_loss(pred, y_train)
            min_loss = loss_mse_train
            if loss_mse_train < 28:
                optimizer.param_groups[0]['lr'] =  2.5e-7
            # elif loss_mse_train < 100:
            #     optimizer.param_groups[0]['lr'] =  1e-04
            # elif loss_mse_train < 500:
            #     optimizer.param_groups[0]['lr'] = 1e-02 #0.000025
                
            # elif loss_mse_train < 10_000:
            #     optimizer.param_groups[0]['lr'] = 1e-01

            #history['MSE_train'].append(loss_mse_train)
            #history['MAE_train'].append(loss_mae_train)

            y_pred_test = kinematic.forward(x_test)
            loss_mse_test = mse_loss(y_pred_test, y_test).item()
            loss_mae_test = mae_loss(y_pred_test, y_test).item()

            #history['MSE_test'].append(loss_mse_test)
            #history['MAE_test'].append(loss_mae_test)

            print(
                f"Train MSE {loss_mse_train:0.4f} Train MAE: {loss_mae_train:0.4f} Test MSE {loss_mse_test:0.4f} MAE {loss_mae_test:0.4f}  epoch: {epoch}")
    
        epoch +=1
    # torch.save(kinematic.state_dict(), 'kinematic.pth')

    y_pred_test: torch.Tensor = kinematic.forward(x_test)

    pred = y_pred_test.detach().numpy()

    true = y_test.detach().numpy()

    print('Pred: ', pred[:10])
    print('True: ', true[:10])

    # time_in_epochs = np.arange(len(history['MAE_test']))*10

    # plt.figure(0)
    # plt.plot(time_in_epochs, history['MSE_train'], label='TRAIN MSE')
    # plt.plot(time_in_epochs, history['MSE_test'], 'r-', label='TEST MSE')
    # plt.legend()
    # plt.figure(1)
    # plt.plot(time_in_epochs, history['MAE_train'], label='TRAIN MAE')
    # plt.plot(time_in_epochs, history['MAE_test'], 'r-', label='TEST MAE')
    # plt.legend()
    # plt.show()

    print(kinematic)


if __name__ == '__main__':
    main()
