import torch

class DNQModel(torch.nn.Module):

    def __init__(self, height: int, width: int, outputs: int, device: torch.device) -> None:

        self.__device = device
        super().__init__()

        kernel = 5
        stride = 2
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=kernel, stride=stride)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(
            16, 16*2, kernel_size=kernel, stride=stride)
        self.bn2 = torch.nn.BatchNorm2d(16*2)
        self.conv3 = torch.nn.Conv2d(
            16*2, 16*2, kernel_size=kernel, stride=stride)
        self.bn3 = torch.nn.BatchNorm2d(16*2)

        conv_w = self.__conv2d_to_linear_size(width, kernel, stride)
        conv_w = self.__conv2d_to_linear_size(conv_w, kernel, stride)
        conv_w = self.__conv2d_to_linear_size(conv_w, kernel, stride)

        conv_h = self.__conv2d_to_linear_size(height, kernel, stride)
        conv_h = self.__conv2d_to_linear_size(conv_h, kernel, stride)
        conv_h = self.__conv2d_to_linear_size(conv_h, kernel, stride)

        linear_input_size = conv_w * conv_h * 32

        self.head = torch.nn.Linear(linear_input_size, outputs)

    @staticmethod
    def __conv2d_to_linear_size(size: int, kernel_size: int, stride: int) -> int:
        """
            Number of Linear input connections depends on output of conv2d layers
            and therefore the input image size, so compute it.
        """
        return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, x: torch.Tensor):
        x = x.to(self.__device)

        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.nn.functional.relu(x)

        return self.head(x.view(x.size(0), -1))
