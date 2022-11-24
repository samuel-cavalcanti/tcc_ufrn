import torch
from torch import nn
import numpy as np

ORI_MODEL_PATH = 'test_ori.tch'
KINEMATIC_MODEL_PATH = 'test.tch'


class Kinematic(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.alpha = nn.parameter.Parameter(torch.empty((1, 2)))
        self.beta = nn.parameter.Parameter(torch.empty((1, 2)))
        self.l = nn.parameter.Parameter(torch.empty((1, 2)))
        self.r = nn.parameter.Parameter(torch.empty((1, 2)))
        # self.linear = nn.Linear(3, 2, bias=False)

        nn.init.kaiming_uniform_(self.alpha, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.beta, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.l, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.r, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        gamma = self.alpha + self.beta

        sin_gamma = gamma.sin()
        cos_gamma = gamma.cos()
        l_pred_phi = -self.l * self.beta.cos()/self.r
        

        """
            1/r[sin ( α + β ), - cos ( α + β ), - l cos(β)] = ϕ
        """
        weights = torch.cat(((sin_gamma/self.r).T, (-cos_gamma/self.r).T, l_pred_phi.T), 1)


        """
            [ cos ( α + β ), sin ( α + β ), l sin β ] = 0
        """
        l_pred_zero =self.l*self.beta.sin()
        weights_2 = torch.cat((cos_gamma.T,sin_gamma.T,l_pred_zero.T),1 )


      

        w = torch.cat((weights,weights_2))





        return nn.functional.linear(x, w, None)

    @torch.jit.ignore  # type: ignore
    def save_model(self) -> None:
        torch.save(self.state_dict(), KINEMATIC_MODEL_PATH)

    @torch.jit.ignore  # type: ignore
    def to_jit_script(self) -> None:
        sm = torch.jit.script(self)  # type: ignore
        sm.save('kinematic.pt')  # type: ignore


def load_kinematic() -> Kinematic:

    model = Kinematic()
    """
        solução analítica da cinemática

        weight = torch.tensor(
        [
            [1/0.0825, 0, 0.30145/0.0825],
            [1/0.0825, 0, -0.30145/0.0825],
        ])

        for param in model.parameters():
            param.data.copy_(weight)
    """

    model.load_state_dict(torch.load(KINEMATIC_MODEL_PATH))
    for name,param in model.named_parameters():
        print(f'{name}:{param.data}')
    return model
