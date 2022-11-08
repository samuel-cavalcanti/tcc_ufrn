from coppeliasim import Coppeliasim
import numpy as np
from tqdm import tqdm

from datasets import SIM_DATA_PATH,SIM_DATA_ROTATE_PATH


def main():

    sim = Coppeliasim()

    def run_sim_10_steps():
        for _ in range(10):
            sim.step()

    def reset_simulation():
        sim.move_target()
        sim.reset_robot_pos()
        sim.set_Velocity(0.0, 0.0)
        sim.set_robot_random_orientation()
        run_sim_10_steps()

    sim.enable_step()
    sim.start_simulation()
    reset_simulation()

    iterations = 0
    max_iterations = 2_000

    sim_t_1_data: list[list[float]] = list()
    sim_t_2_data: list[list[float]] = list()

    def read_simulation_data() -> tuple[list[float], float, float]:
        """return: robot pos[x,y,z], time , robot orientation in radians """
        theta = sim.get_orientation_z()
        robot_pos = sim.get_robot_pos()
        time = sim.get_time()

        return robot_pos, time, theta

    for iterations in tqdm(iterable=range(1, max_iterations+1)):  

        if iterations % 18 == 0:
            reset_simulation()

        s_1, t_1, theta_1 = read_simulation_data()
        vel_wheels = np.random.sample((2)) * 2
        if iterations < 1_000:
            vel_wheels[0] = 0.0
        elif iterations < 2_000:
            vel_wheels[1] = 0.0

        print('robot pos 1:', s_1[:2])
        sim.set_Velocity(float(vel_wheels[0]), float(vel_wheels[1]))  # type: ignore
        run_sim_10_steps()
        s_2, t_2, theta_2 = read_simulation_data()
        print(f'robot pos 2: {s_2[:2]} theta {theta_1} wheels {vel_wheels}')
        print('vel:', np.array(s_2)/(t_2-t_1) - np.array(s_1)/(t_2-t_1))
       
        sim_t_1 = list()
        sim_t_2 = list()

        sim_t_1.append(t_1)
        sim_t_1.extend(s_1[:-1])
        sim_t_1.append(theta_1)
        sim_t_1.extend(vel_wheels.tolist())

        sim_t_2.append(t_2)
        sim_t_2.extend(s_2[:-1])
        sim_t_2.append(theta_2)

        sim_t_1_data.append(sim_t_1)
        sim_t_2_data.append(sim_t_2)

    sim.stop_simulation()

    sim_t_1_data = np.array(sim_t_1_data, dtype=np.float32)  # type: ignore
    sim_t_2_data = np.array(sim_t_2_data, dtype=np.float32)  # type: ignore

    with open(SIM_DATA_ROTATE_PATH, 'wb') as f:
        np.save(f, sim_t_1_data)  # type: ignore
        np.save(f, sim_t_2_data)  # type: ignore

