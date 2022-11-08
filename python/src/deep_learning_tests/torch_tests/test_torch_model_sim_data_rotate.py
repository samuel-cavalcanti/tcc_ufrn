import torch
from torch import nn
import numpy as np

from matplotlib import pyplot as plt
from datasets import load_sim_data_rotate
from constants import Array32
from tqdm import tqdm
from .kinematic_model import ORI_MODEL_PATH,load_orientation






def to_kinematic_data(data_t1, data_t2) -> tuple[Array32, Array32]:
    """
        x = linear velocity x linear velocity y, angular velocity 
        y = wheel left velocity, wheel right velocity

        return (x,y)
    """
    dt = data_t2[:, 0] - data_t1[:, 0]
    ds = data_t2[:, 1:3] - data_t1[:, 1:3]
    theta_1 = data_t1[:, 3]

    dtheta = data_t2[:, 3] - theta_1

    dt = dt.reshape((-1, 1))
    dtheta = dtheta.reshape((-1, 1))
    theta_1 = theta_1.reshape((-1, 1))
    vel_lin = ds/dt
    vel_ang = dtheta/dt

    for i in range(len(vel_lin)):
        theta = theta_1[i, 0]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        vel_lin_x = vel_lin[i, 0]
        vel_lin_y = vel_lin[i, 1]

        vel_lin[i, 0] = vel_lin_x*cos_theta + vel_lin_y*sin_theta
        vel_lin[i, 1] = -vel_lin_x*sin_theta + vel_lin_y*cos_theta

    kinematic_input = np.hstack((vel_lin, vel_ang))
    kinematic_output = data_t1[:, -2:]

    return kinematic_input, kinematic_output


def main() -> None:

    sim_t1, sim_t2 = load_sim_data_rotate()

    x, y = to_kinematic_data(data_t1=sim_t1, data_t2=sim_t2)

    x = x[:, -1].reshape((-1,1))

    index = np.arange(x.shape[0])
    np.random.shuffle(index)
    x = x[index]
    y = y[index]

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

    ori_model = load_orientation()

    print(ori_model)

    optimizer = torch.optim.RMSprop(
        ori_model.parameters(), lr=1.0e-4)  # 0.000025

    reduction = 'sum'
    mse_loss = nn.MSELoss(reduction=reduction)
    mae_loss = nn.L1Loss(reduction=reduction)

    pbar = tqdm(range(700_000))
    for epoch in pbar:

        pred = ori_model.forward(x_train)
        loss: torch.Tensor = mse_loss(pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:

            y_pred_train = ori_model.forward(x_train)
            loss_mse_train = mse_loss(y_pred_train, y_train).item()
            loss_mae_train = mae_loss(y_pred_train, y_train).item()

            y_pred_test = ori_model.forward(x_test)
            loss_mse_test = mse_loss(y_pred_test, y_test).item()
            loss_mae_test = mae_loss(y_pred_test, y_test).item()

            history['MSE_test'].append(loss_mse_test)
            history['MAE_test'].append(loss_mae_test)

            history['MSE_train'].append(loss_mse_train)
            history['MAE_train'].append(loss_mae_train)

            description = f"Train MSE {loss_mse_train:0.4f} Train MAE: {loss_mae_train:0.4f} Test MSE {loss_mse_test:0.4f} MAE {loss_mae_test:0.4f}  epoch: {epoch}"

            pbar.set_description(description)

    torch.save(ori_model.state_dict(), ORI_MODEL_PATH)

    y_pred_test = ori_model.forward(x_test)

    pred = y_pred_test.detach().numpy()

    true = y_test.detach().numpy()

    print('Pred: ', pred[:10])
    print('True: ', true[:10])

    time_in_epochs = np.arange(len(history['MAE_test']))*10  # type: ignore

    plt.figure(0)
    plt.plot(time_in_epochs, history['MSE_train'], label='TRAIN MSE')
    plt.plot(time_in_epochs, history['MSE_test'], 'r-', label='TEST MSE')
    plt.legend()
    plt.figure(1)
    plt.plot(time_in_epochs, history['MAE_train'], label='TRAIN MAE')
    plt.plot(time_in_epochs, history['MAE_test'], 'r-', label='TEST MAE')
    plt.legend()

    plt.figure(2)
    plt.title('Pred x True')
    plt.scatter(y[:, 0], y[:, 1], label='True')
    plt.scatter(pred[:, 0], pred[:, 1], s=0.5, c='r', label='Predict')
    plt.legend()
    plt.show()
