
from coppeliasim import Coppeliasim
import numpy as np

from frederico_controller import FredericoController, PID
from deep_learning_tests.torch_tests.kinematic_model import load_kinematic
from constants import Array32
import torch


def create_federico_controller() -> FredericoController:
    k_p = 0.05145
    k_i = k_p / 2.9079
    k_d = k_p * 0.085
    """Constantes Kp Ki Kd adquiridas experimentalmente """
    pid_position = PID(k_p, k_i, k_d, set_point=0)
    k_p = 0.2
    k_i = 0.015
    k_d = k_p * 0.00  # 474
    pid_orientation = PID(k_p, k_d, k_i, set_point=0)
    controller = FredericoController(position_controller=pid_position,
                                     orientation_controller=pid_orientation)

    return controller


def exploration_rate(n: int, min_rate=0.1) -> float:
    """Decaying exploration rate"""
    return max(min_rate, min(1, 1.0 - np.log10((n + 1) / 25)))


def is_goal(target_pos) -> bool:
    thirty_centimeters = 0.3
    distance = np.sqrt(target_pos[0]**2 + target_pos[1]**2)
    return distance < thirty_centimeters


def main():

    sim = Coppeliasim()
    controller = create_federico_controller()
    kinematic = load_kinematic().float()
    # torch.jit.script(kinematic).save('kinematic_module.pt')

    def run_sim_n_steps(n: int = 5):
        for _ in range(n):
            sim.step()

    def reset_simulation():
        sim.move_target()
        sim.reset_robot_pos()
        sim.set_Velocity(0.0, 0.0)
        sim.set_robot_random_orientation()
        run_sim_n_steps()

    sim.enable_step()
    sim.start_simulation()
    # reset_simulation()

    def read_simulation_data() -> tuple[list[float], float, float]:
        """return: robot pos[x,y,z], time , robot orientation in radians """
        theta = sim.get_orientation_z()
        robot_pos = sim.get_robot_pos()
        time = sim.get_time()

        return robot_pos, time, theta

    robot_pos = np.zeros((3), dtype=np.float32)
    target_pos = np.zeros((2), dtype=np.float32)

    kinematic_input = np.zeros((3), dtype=np.float32)
    limit = [0.16133267,0.04307239, np.deg2rad(20.0)]
    max_iterations = 10_000

    for _ in range(1, max_iterations+1):

        s_1, _, theta_1 = read_simulation_data()

        target_pos_sim = sim.get_target_pos()

        robot_pos[:2] = s_1[:2]
        robot_pos[2] = theta_1
        target_pos[:] = target_pos_sim[:2]

        vel_robot = controller.step(current=robot_pos, desired_pos=target_pos)

        cos_theta = np.cos(theta_1)
        sin_theta = np.sin(theta_1)

        """
           q_r = R(theta)q
            
            |vx_r|  =   | cos(theta) sen(theta)| |vx|
            |vy_r|  =   |-sin(theta) cos(theta)| |vy|

            mudando a referência da velocidade do referencial global (q) para o referencial do robô (q_r)
        """
        kinematic_input[0] = vel_robot[0]*cos_theta + vel_robot[1]*sin_theta
        kinematic_input[1] = -vel_robot[0]*sin_theta + vel_robot[1]*cos_theta
        kinematic_input[2] = vel_robot[2]

        for i in range(3):
            if np.abs(kinematic_input[i]) >limit[i]:
                kinematic_input[i]=kinematic_input[i]/ kinematic_input[i] *limit[i]

        distance_to_target = robot_pos[:2] - target_pos
        norm_distance = np.linalg.norm(distance_to_target)
        angle = np.arctan2(distance_to_target[1], distance_to_target[0])

        print(
            f'vel distance to target {norm_distance:0.4f} angle: {np.rad2deg(angle):0.4f}  ')
        print(
            f'vel x:{vel_robot[0]:0.4f} vel y:{vel_robot[1]:0.4f} theta {np.rad2deg(vel_robot[2]):0.4} ')
        print(
            f'kinematic x:{kinematic_input[0]:0.4f} kinematic y:{kinematic_input[1]:0.4f} kinematic {vel_robot[2]:0.4} ')

        vel_wheels: torch.Tensor = kinematic.forward(
            torch.from_numpy(kinematic_input).float().reshape((1, -1)))[0]

        vel_wheels = vel_wheels.detach().numpy()

        print('vel_wheels', vel_wheels)

        sim.set_Velocity(float(vel_wheels[0]), float(vel_wheels[1]))
        run_sim_n_steps()

    sim.stop_simulation()


if __name__ == '__main__':
    main()
