import torchvision.transforms as T
import torch
import numpy  as np




resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=T.InterpolationMode.BICUBIC),
                    T.ToTensor()])  # type: ignore


def get_cart_location(screen_width: int, world_width: float, cart_world_pos_x: float) -> int:
    scale = screen_width / world_width

    return int(cart_world_pos_x * scale + screen_width / 2.0)  # MIDDLE OF CART


def image_to_tensor(image: np.ndarray, world_width: float, cart_world_pos_x: float) -> torch.Tensor:

    screen = image.transpose((2, 0, 1))

    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]

    view_width = int(screen_width * 0.6)

    cart_location = get_cart_location(
        screen_width, world_width, cart_world_pos_x)

    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0)




