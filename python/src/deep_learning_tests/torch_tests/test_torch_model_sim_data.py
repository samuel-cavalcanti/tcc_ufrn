import torch
from torch import nn
import numpy as np
from constants import Array32
from matplotlib import pyplot as plt
import datasets

from tqdm import tqdm

from . import kinematic_model


def randomize_and_split_data(x: Array32, y: Array32, split_percentage: float) -> tuple[tuple[Array32, Array32], tuple[Array32, Array32]]:
    index = np.arange(x.shape[0])
    np.random.shuffle(index)
    x = x[index]
    y = y[index]

    size_train = int(x.shape[0]*split_percentage)

    x_train = x[:size_train]
    x_test = x[size_train:]
    y_train = y[:size_train]
    y_test = y[size_train:]

    return (x_train, y_train), (x_test, y_test)


def plot_train(history: dict[str, list[float]], plot_name: str):
    time_in_epochs = np.arange(len(history['MAE_test']), dtype=np.float32)*10
    plt.figure(1)
    plt.plot(time_in_epochs, history['MSE_train'], label='TRAIN MSE')
    plt.plot(time_in_epochs, history['MSE_test'], 'r-', label='TEST MSE')
    plt.legend()
    plt.savefig(f'plots/{plot_name}_mse.pdf')
    plt.figure(2)
    plt.plot(time_in_epochs, history['MAE_train'], label='TRAIN MAE')
    plt.plot(time_in_epochs, history['MAE_test'], 'r-', label='TEST MAE')
    plt.legend()
    plt.savefig(f'plots/{plot_name}_mae.pdf')
    # plt.show()


def train_model(model: nn.Module, train: tuple[torch.Tensor, torch.Tensor], test: tuple[torch.Tensor, torch.Tensor], epochs: int) -> dict[str, list[float]]:
    history = {
        'MSE_train': [],
        'MAE_train': [],

        'MSE_test': [],
        'MAE_test': [],
    }
    x_train, y_train = train
    x_test, y_test = test

    optimizer = torch.optim.RMSprop(model.parameters(), lr=1.0e-4)
    reduction = 'mean'
    mse_loss = nn.MSELoss(reduction=reduction)
    mae_loss = nn.L1Loss(reduction=reduction)
    pbar = tqdm(iterable=range(epochs))
    last_loss = -2

    for epoch in pbar:
        pred: torch.Tensor = model.forward(x_train)
        loss = mse_loss.forward(pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:

            y_pred_train: torch.Tensor = model.forward(x_train)
            loss_mse_train = mse_loss.forward(y_pred_train, y_train).item()
            loss_mae_train = mae_loss.forward(y_pred_train, y_train).item()

            y_pred_test = model.forward(x_test)
            loss_mse_test = mse_loss.forward(y_pred_test, y_test).item()
            loss_mae_test = mae_loss.forward(y_pred_test, y_test).item()

            history['MSE_test'].append(loss_mse_test)
            history['MSE_train'].append(loss_mse_train)

            history['MAE_test'].append(loss_mae_test)
            history['MAE_train'].append(loss_mae_train)

            pbar.set_description(
                f"Train MSE {loss_mse_train:0.4f} Train MAE: {loss_mae_train:0.4f} Test MSE {loss_mse_test:0.4f} MAE {loss_mae_test:0.4f}  epoch: {epoch}")
            if last_loss == loss_mse_train:
                break
            last_loss = loss_mse_train

    return history


def train_kinematic_model() -> None:

    sim_t1, sim_t2 = datasets.load_sim_data_dataset()

    x, y = datasets.pre_processing_sim_data(data_t1=sim_t1, data_t2=sim_t2)

    orthogonal_motion = np.zeros((x.shape[0], 2), dtype=np.float32)
    y = np.hstack((y, orthogonal_motion))

    (x_train, y_train), (x_test, y_test) = randomize_and_split_data(x, y, 0.6)

    x_train = torch.from_numpy(x_train)
    x_test = torch.from_numpy(x_test)

    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)

    kinematic = kinematic_model.load_kinematic(load_weights=False)

    history = train_model(model=kinematic,
                          train=(x_train, y_train),
                          test=(x_test, y_test),
                          epochs=300_000)

    for name, param in kinematic.named_parameters():
        print(f'{name}:{param.data}')

    kinematic.save_model()

    plot_train(history, 'kinematic')


def train_linear_model() -> None:

    sim_t1, sim_t2 = datasets.load_sim_data_dataset()

    x, y = datasets.pre_processing_sim_data(data_t1=sim_t1, data_t2=sim_t2)

    (x_train, y_train), (x_test, y_test) = randomize_and_split_data(x, y, 0.6)

    x_train = torch.from_numpy(x_train)
    x_test = torch.from_numpy(x_test)

    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)

    linear_model = kinematic_model.load_linear(load_weights=False)

    history = train_model(model=linear_model,
                          train=(x_train, y_train),
                          test=(x_test, y_test),
                          epochs=300_000)

    for name, param in linear_model.named_parameters():
        print(f'{name}:{param.data}')

    kinematic_model.save_linear(linear_model)

    plot_train(history, 'linear')


def evaluate_models(models: dict[str, nn.Module], test: tuple[torch.Tensor, torch.Tensor]) -> None:

    x_test, y_test = test
    reduction = 'mean'
    mse_loss = nn.MSELoss(reduction=reduction)

    for model_name, model in models.items():
        y_pred = model.forward(x_test)
        error = mse_loss.forward(y_pred, y_test)
        print(f'model: {model_name} error: {error.item():0.4f}')


def evaluate() -> None:
    kinematic = kinematic_model.load_kinematic()
    linear = kinematic_model.load_linear()

   
    analytic = kinematic_model.load_analytic()

    

    linear_kinematic = kinematic.build_linear()

    models: dict[str, nn.Module] = {
        'kinematic': linear_kinematic,
        'linear': linear,
        'analytic': analytic
    }

    x, y  = datasets.load_sim_5_dataset()




    _, (x_test, y_test) = randomize_and_split_data(x, y, 0.6)

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    evaluate_models(models, test=(x_test, y_test))

    print('analytic matrix', analytic.weight)
    print('kinematic matrix', linear_kinematic.weight)
    print('linear matrix', linear.weight)
