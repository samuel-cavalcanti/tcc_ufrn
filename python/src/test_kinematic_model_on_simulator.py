
from coppeliasim import Coppeliasim
import numpy as np

from frederico_controller import FredericoController, PID
from deep_learning_tests.torch_tests import kinematic_model
from constants import Array32
import datasets
import torch
import time
from tqdm import tqdm
def create_federico_controller() -> FredericoController:
    k_p = 0.05145
    k_i = k_p / 2.9079
    k_d = k_p * 0.085
    """Constantes Kp Ki Kd adquiridas experimentalmente """
    pid_position = PID(k_p, k_i, k_d, set_point=0)
    k_p = 0.4
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

    models = [kinematic_model.load_analytic(),
              kinematic_model.load_linear(),
              kinematic_model.load_kinematic().build_linear()]

    dataset_names = [datasets.ERROR_T_ANALYTIC_PATH,
                     datasets.ERROR_T_LINEAR_PATH,
                     datasets.ERROR_T_KINEMATIC_PATH]
       

    for model,dataset_name in zip(models,dataset_names):
        sim = Coppeliasim()
        sim.enable_step()
        sim.start_simulation()
        distances,angles,sim_times =run_simulation(sim,model.float())
        sim.stop_simulation()
        time.sleep(5)
        del sim

        x = np.array(sim_times, dtype=np.float32)
        d = np.array(distances, dtype=np.float32).reshape(-1, 1)
        a = np.array(angles, dtype=np.float32).reshape(-1, 1)
        y = np.hstack((d, a))
        datasets.save_dataset(x, y, dataset_name)



def run_simulation(sim:Coppeliasim,model:torch.nn.Module)->tuple[list[float],list[float],list[float]]:
    max_iterations = 10_000
    distances: list[float] = list()
    sim_times: list[float] = list()
    angles: list[float] = list()
    robot_pos = np.zeros((3), dtype=np.float32)
    target_pos = np.zeros((2), dtype=np.float32)
    controller = create_federico_controller()
    for name, param in model.named_parameters():
        print(f'{name}:{param.data}')
    kinematic_input = np.zeros((3), dtype=np.float32)
    def read_simulation_data() -> tuple[list[float], float, float]:
        """return: robot pos[x,y,z], time , robot orientation in radians """
        theta = sim.get_orientation_z()
        robot_pos = sim.get_robot_pos()
        time = sim.get_time()

        return robot_pos, time, theta
    

    def run_sim_n_steps(n: int = 6):
        """ 
            cada step são 5 milissegundos
            então n steps são n*5ms, n=6 => 30ms
        """
        for _ in range(n):
            sim.step()
    
    pbar = tqdm(range(1, max_iterations+1))
    for _ in pbar:

        s_1, _, theta_1 = read_simulation_data()

        target_pos_sim = sim.get_target_pos()

        robot_pos[:2] = s_1[:2]
        robot_pos[2] = theta_1
        target_pos[:] = target_pos_sim[:2]

        vel_robot = controller.step(current=robot_pos, desired_pos=target_pos)

        cos_theta = np.cos(theta_1)
        sin_theta = np.sin(theta_1)
        pbar.set_description(f'sin(theta)= {sin_theta}, theta: {np.rad2deg(theta_1)}, vel robot: {vel_robot}')

        """
        q_r = R(theta)q
            
            |vx_r|  =   | cos(theta) sen(theta)| |vx|
            |vy_r|  =   |-sin(theta) cos(theta)| |vy|

            mudando a referência da velocidade do referencial global (q) para o referencial do robô (q_r)
        """
        kinematic_input[0] = vel_robot[0]*cos_theta + vel_robot[1]*sin_theta
        kinematic_input[1] = -vel_robot[0]*sin_theta + vel_robot[1]*cos_theta
        kinematic_input[2] = vel_robot[2]

        distance_to_target = robot_pos[:2] - target_pos
        angle = np.arctan2(distance_to_target[1], distance_to_target[0])

        vel_wheels: torch.Tensor = model.forward(
            torch.from_numpy(kinematic_input).float().reshape((1, -1)))[0]

        vel_wheels = vel_wheels.detach().numpy()

        sim.set_Velocity(float(vel_wheels[0]), float(vel_wheels[1]))

        sim_time = sim.get_time()
        distance = float(np.linalg.norm(distance_to_target))
        distances.append(distance)
        angles.append(angle)
        sim_times.append(sim_time)

        if is_goal(distance_to_target):
            break

        run_sim_n_steps()

    return distances,angles,sim_times


if __name__ == '__main__':
    main()
