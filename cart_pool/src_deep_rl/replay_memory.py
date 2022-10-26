from collections import deque
import random
from transition import Transition


class ReplayMemory:
    __memory: deque

    def __init__(self, capacity: int) -> None:
        self.__memory = deque([], capacity)

    def push(self, transition: Transition) -> None:
        self.__memory.append(transition)

    def rand_sample(self, bash_size: int) -> list[Transition]:
        return random.sample(self.__memory, bash_size)

    def __len__(self) -> int:
        return len(self.__memory)
