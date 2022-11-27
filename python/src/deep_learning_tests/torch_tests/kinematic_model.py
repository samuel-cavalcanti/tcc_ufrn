import torch
from torch import nn
import numpy as np

KINEMATIC_MODEL_PATH = 'models/kinematic.tch'
KINEMATIC_MODEL_JIT_PATH = 'models/kinematic.pt'

LINEAR_MODEL_PATH = 'models/linear.tch'
LINEAR_MODEL_JIT_PATH = 'models/linear.pt'


class Kinematic(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.alpha = nn.parameter.Parameter(torch.empty((1, 2)))
        self.beta = nn.parameter.Parameter(torch.empty((1, 2)))
        self.l = nn.parameter.Parameter(torch.empty((1, 2)))
        self.r = nn.parameter.Parameter(torch.empty((1, 2)))

        nn.init.kaiming_uniform_(self.alpha, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.beta, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.l, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.r, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        gamma = self.alpha + self.beta

        sin_gamma = gamma.sin()
        cos_gamma = gamma.cos()
        l_pred_phi = self.l * self.beta.cos()/self.r

        """
            1/r[sin ( α + β ), - cos ( α + β ), - l cos(β)] = ϕ
        """
        weight_phi = torch.cat(
            ((sin_gamma/self.r).T, (-cos_gamma/self.r).T, -l_pred_phi.T), 1)

        # print(f'sin_gamma/self.r:{sin_gamma/self.r}')
        # print(f'-cos_gamma/self.r:{-cos_gamma/self.r}')
        # print(f'l_pred_phi:{l_pred_phi}\n')
        # print(f'weight_phi:{weight_phi}\n')
       

        """
            [ cos ( α + β ), sin ( α + β ), l sin β ] = 0
        """
        l_pred_zero = self.l*self.beta.sin()
        weights_0 = torch.cat((cos_gamma.T, sin_gamma.T, l_pred_zero.T), 1)

        # print(f'cos_gamma:{cos_gamma}')
        # print(f'sin_gamma:{sin_gamma}')
        # print(f'l_pred_zero:{l_pred_zero}\n')
        # print(f'weights_0:{weights_0}\n')
       
        w = torch.cat((weight_phi, weights_0))

        return nn.functional.linear(x, w, bias=None)

    @torch.jit.ignore  # type: ignore
    def save_model(self) -> None:
        torch.save(self.state_dict(), KINEMATIC_MODEL_PATH)

    @torch.jit.ignore  # type: ignore
    def to_jit_script(self) -> None:
        model = self.build_linear()
        sm = torch.jit.script(model)  # type: ignore
        sm.save(KINEMATIC_MODEL_JIT_PATH)  # type: ignore

    @torch.jit.ignore  # type: ignore
    def build_linear(self) -> nn.Linear:
        gamma = self.alpha + self.beta

        sin_gamma = gamma.sin()/self.r
        cos_gamma = gamma.cos()/self.r
        l_pred = self.l * self.beta.cos()/self.r

        weight = torch.cat((sin_gamma.T, -cos_gamma.T, -l_pred.T), 1)
        linear = nn.Linear(3, 2, bias=False)
        linear.weight.data.copy_(weight)
        return linear


def load_kinematic(load_weights: bool = True) -> Kinematic:

    model = Kinematic()
   
    if load_weights:
        model.load_state_dict(torch.load(KINEMATIC_MODEL_PATH))
    return model


def load_linear(load_weights: bool = True) -> nn.Linear:

    model = nn.Linear(3, 2, bias=False)
    if load_weights:
        model.load_state_dict(torch.load(LINEAR_MODEL_PATH))
    return model


def load_analytic() -> nn.Linear:
    r = 0.0825
    l = 0.291285
    analytic_kinematic = torch.tensor(
        [
            [1/r, 0, l/r],
            [1/r, 0, -l/r],
        ])
    analytic = nn.Linear(3, 2, bias=False)

    if analytic._parameters['weight'] is None:
        raise Exception('weight parameter is None')
    analytic._parameters['weight'].data.copy_(analytic_kinematic)

    return analytic
  

def save_linear(model: nn.Linear) -> None:
    torch.save(model.state_dict(), LINEAR_MODEL_PATH)
