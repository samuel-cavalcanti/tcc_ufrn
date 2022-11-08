import torch
from torch import nn


ORI_MODEL_PATH = 'test_ori.tch'
TRANSLATE_MODEL_PATH = 'test.tch'


class TranslateModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.w = nn.Linear(2, 2, bias=True)

    def forward(self, vel: torch.Tensor) -> torch.Tensor:

        return self.w.forward(vel)


class OrientationLayer(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.w = nn.Linear(1, 2)
        self.w_2 = nn.Linear(2, 2)

    def forward(self, vel_ang: torch.Tensor) -> torch.Tensor:
        alpha = self.w.forward(vel_ang)

        return self.w_2.forward(alpha.sin())


class Kinematic(nn.Module):

    
    # linear: nn.Module
    # angular: nn.Module
    
   
    def __init__(self, ori: nn.Module, trans: nn.Module) -> None:
        super().__init__()
        self.linear = trans
        self.angular = ori

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        linear_vel = x[:, :2]
        angular_vel = x[:, 2].reshape((-1, 1))

        linear_vel_wheel = self.linear.forward(linear_vel)
        angular_vel_wheel = self.angular.forward(angular_vel)

        return (linear_vel_wheel + angular_vel_wheel)/2.0

    @torch.jit.ignore # type: ignore
    def save_model(self) -> None:
        torch.save(self.linear.state_dict(), TRANSLATE_MODEL_PATH)
        torch.save(self.angular.state_dict(), ORI_MODEL_PATH)

    @torch.jit.ignore # type: ignore
    def to_jit_script(self) -> None:
        sm = torch.jit.script(self) # type: ignore
        sm.save('kinematic.pt')  # type: ignore


def load_orientation() -> nn.Module:
    device = torch.device('cpu')
    angular = OrientationLayer().to(device)
    angular.load_state_dict(torch.load(ORI_MODEL_PATH))
    return angular


def load_translate() -> nn.Module:
    device = torch.device('cpu')
    linear = TranslateModel().to(device)
    linear.load_state_dict(torch.load(TRANSLATE_MODEL_PATH))
    return linear


def load_kinematic() -> Kinematic:
    # device = torch.device('cpu')
    linear = TranslateModel()  # .to(device)
    angular = OrientationLayer()  # .to(device)

    linear.load_state_dict(torch.load(TRANSLATE_MODEL_PATH))
    angular.load_state_dict(torch.load(ORI_MODEL_PATH))

    return Kinematic(angular, linear)  # .to(device)
