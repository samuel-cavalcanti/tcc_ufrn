from coppeliasim import Coppeliasim
import numpy as np
from tqdm import tqdm

import datasets


N_SAMPLES = 10_000
VEL_WHEEL_MAX = 5.0


def main():

    sim = Coppeliasim()

    def run_sim_n_steps(n: int = 10):
        """ 
            cada step são 5 milissegundos
            então n steps são n*5ms, n=6 => 30ms
        """
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
    reset_simulation()

    
    sim_t_1_data: list[list[float]] = list()
    sim_t_2_data: list[list[float]] = list()

    kinematic_outputs: list[list[float]] = list()
    kinematic_inputs: list[list[float]] = list()

    def read_simulation_data() -> tuple[list[float], float, float]:
        """return: robot pos[x,y,z], time , robot orientation in radians """
        theta = sim.get_orientation_z()
        robot_pos = sim.get_robot_pos()
        time = sim.get_time()

        return robot_pos, time, theta

    pbar = tqdm(iterable=range(1, N_SAMPLES+1))
    for iterations in pbar:

        if iterations % 15 == 0:
            reset_simulation()

        s_1, t_1, theta_1 = read_simulation_data()
        vel_wheels = np.random.sample((2)) * VEL_WHEEL_MAX

        # print('robot pos 1:', s_1[:2])
        sim.set_Velocity(
            left=float(vel_wheels[0]),
            right=float(vel_wheels[1]))  # type: ignore
        run_sim_n_steps()
        vel_lin, vel_ang = sim.get_robot_velocity()

    
        s_2, t_2, theta_2 = read_simulation_data()
        vel_lin, vel_ang = sim.get_robot_velocity()
        theta = sim.get_orientation_z()
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        kinematic_input = [0.0, 0.0, 0.0]
        kinematic_input[0] = vel_lin[0]*cos_theta + vel_lin[1]*sin_theta
        kinematic_input[1] = -vel_lin[0]*sin_theta + vel_lin[1]*cos_theta
        kinematic_input[2] = vel_ang[2]
        pbar.set_description(f'vel lin: [{kinematic_input[0]:0.4f},{kinematic_input[1]:0.4f}] vel ang: {kinematic_input[2]:0.4f}')

        vel_wheels = vel_wheels.tolist()
        sim_t_1 = list()
        sim_t_2 = list()

        sim_t_1.append(t_1)
        sim_t_1.extend(s_1[:-1])
        sim_t_1.append(theta_1)
        sim_t_1.extend(vel_wheels)

        sim_t_2.append(t_2)
        sim_t_2.extend(s_2[:-1])
        sim_t_2.append(theta_2)

        sim_t_1_data.append(sim_t_1)
        sim_t_2_data.append(sim_t_2)

        kinematic_outputs.append(vel_wheels)
        kinematic_inputs.append(kinematic_input)

    sim.stop_simulation()

    x = np.array(kinematic_inputs, dtype=np.float32)
    y = np.array(kinematic_outputs, dtype=np.float32)

    datasets.save_dataset(x, y, datasets.SIM_DATA_5_PATH)
