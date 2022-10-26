import gym
import numpy as np
import cv2


class CartPole:
    __env: gym.Env
    __env_id = 'CartPole-v1'

    def __init__(self) -> None:
        self.__env = gym.make(self.__env_id,
                              render_mode='rgb_array').unwrapped
        self.__env.reset()
        cv2.namedWindow(self.__env_id)

    def reset(self) -> None:
        self.__env.reset()

    def close(self) -> None:
        self.__env.render()
        self.__env.close()

    def get_image(self) -> np.ndarray:

        buffer: np.ndarray = self.__env.render()   # type: ignore
        cv2.imshow(self.__env_id, cv2.cvtColor(buffer, cv2.COLOR_RGB2BGR))
        cv2.waitKey(30)
        return buffer

    def get_world_width(self) -> float:
        return self.__env.x_threshold * 2  # type: ignore

    def get_state(self) -> np.ndarray:
        return self.__env.state   # type: ignore

    def get_actions(self) -> np.ndarray:
        """
        ### Action Space
            The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
            of the fixed force the cart is pushed with.
            | Num | Action                 |
            |-----|------------------------|
            | 0   | Push cart to the left  |
            | 1   | Push cart to the right |
        """
        return np.ascontiguousarray([0, 1], dtype=np.int32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:

        return self.__env.step(action)

    def random_action(self) -> int:
        return self.__env.action_space.sample()
