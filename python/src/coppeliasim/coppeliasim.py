from dataclasses import dataclass
from typing import Any
from .zmqRemoteApi import RemoteAPIClient
import random


@dataclass
class Motors:
    left: int
    right: int


@dataclass
class Robot:
    handle: int
    motors: Motors


@dataclass
class Targets:
    handle: int
    targets: list[list[float]]
    current_pos_index: int


TARGET_POS = [
    [1.810547113418579, -1.9869998693466187, 0.2530002295970917],
    [-0.15445302426815033, -1.960999608039856, 0.2530002295970917],
    [-0.14445288479328156, -0.9789999127388, 0.2530002295970917],
    [-0.1354527771472931, 0.96200031042099, 0.2530002295970917],
    [-0.144452765583992, 1.948000192642212, 0.2530002295970917],
    [1.7815473079681396, 1.9449999332427979, 0.2530002295970917],
    [
        1.7695471048355103,
        -0.034000199288129807,
        0.2530002295970917,
    ],
    [
        0.8573786616325378,
        -0.014516029506921768,
        0.2530002295970917,
    ],
    [
        -0.15445293486118317,
        -0.00700022280216217,
        0.2530002295970917,
    ],
    [-2.0844528675079346, -0.9789997935295105, 0.2530002295970917],
    [-2.0794517993927, -1.9540002346038818, 0.2530002295970917],
    [-2.093451738357544, 0.006999874487519264, 0.2530002295970917],
    [-2.0964515209198, 1.9259999990463257, 0.2530002295970917],
    [-1.089451551437378, 1.9279992580413818, 0.2530002295970917],
    [-2.0504512786865234, 0.9569994807243347, 0.2530002295970917],
    [0.8035488128662109, 1.9239994287490845, 0.2530002295970917],
    [1.8005484342575073, 0.9539993405342102, 0.2530002295970917],
    [1.7905484437942505, -0.9860007762908936, 0.2530002295970917],
]

INI_POS = [0.0, 0.0, +6.4145e-01]
INIT_ORI = [0.002015680753128251, -0.03632379239250599, 0.0]


class Coppeliasim:

    __client: RemoteAPIClient
    __sim: Any
    __robot: Robot
    __targets: Targets

    def __init__(self) -> None:
        self.__client = RemoteAPIClient()
        self.__sim = self.__client.getObject('sim')

        self.__robot = self.__connect_to_robot()

        self.__targets = self.__connect_to_targets()

    def __connect_to_robot(self) -> Robot:

        robot_handle = self.__sim.getObject('./cantoneira_dobrada')

        motors = Motors(
            left=self.__sim.getObject('./DynamicLeftJoint'),
            right=self.__sim.getObject('./DynamicRightJoint'),
        )

        return Robot(robot_handle, motors)

    def __connect_to_targets(self) -> Targets:

        target = self.__sim.getObject('./target')

        return Targets(handle=target, targets=TARGET_POS, current_pos_index=8)

    def get_target_distance(self) -> list[float]:
        relative_pos: list[float] = self.__sim.getObjectPosition(
            self.__targets.handle, self.__robot.handle)
        return relative_pos

    def get_target_pos(self) -> list[float]:
        return self.__sim.getObjectPosition(
            self.__targets.handle, -1)

    def get_robot_pos(self) -> list[float]:
        return self.__sim.getObjectPosition(self.__robot.handle, -1)

    def reset_robot_pos(self) -> None:
        self.__sim.setObjectPosition(
            self.__robot.handle, -1, INI_POS)
        self.__sim.setObjectOrientation(
            self.__robot.handle, -1,INIT_ORI)

    def set_robot_random_orientation(self):

        self.__sim.setObjectOrientation(
            self.__robot.handle, -1, [INIT_ORI[0], INIT_ORI[1], random.uniform(-3.14159265359, 3.14159265359)])

    def get_orientation_z(self) -> float:
        robot_euler_angles: list[float] = self.__sim.getObjectOrientation(
            self.__robot.handle, -1)

        return robot_euler_angles[2]

    def get_time(self) -> float:

        return self.__sim.getSimulationTime()

    def set_Velocity(self, left: float, right: float) -> None:

        self.__sim.setJointTargetVelocity(self.__robot.motors.left, left)
        self.__sim.setJointTargetVelocity(self.__robot.motors.right, right)

    def start_simulation(self) -> None:
        self.__sim.startSimulation()

    def stop_simulation(self) -> None:
        self.__sim.stopSimulation()

    def enable_step(self) -> None:
        self.__client.setStepping()

    def step(self) -> None:
        self.__client.step()

    def move_target(self) -> None:

        new_index = random.randrange(0, len(self.__targets.targets))

        self.__targets.current_pos_index = new_index

        self.__sim.setObjectPosition(
            self.__targets.handle, -1, self.__targets.targets[self.__targets.current_pos_index])

        pass
