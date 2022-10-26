
from pathlib import Path
from typing import Any
import tensorflow as tf
from coppeliasim import Coppeliasim
import numpy as np
import random
import time
from frederico_controller import FredericoController, PID

Array32 = np.ndarray[Any, np.dtype[np.float32]]

KINEMATIC_PATH = 'dense_model_kinematic/weighs.h5'
CONTROLLER_PATH = 'dense_model_controller/weighs.h5'
ENVIRONMENT_PATH = 'dense_model_environment/weighs.h5'


def create_federico_controller() -> FredericoController:
    k_p = 0.1145
    k_i = k_p / 1.9079
    k_d = k_p * 0.085
    """Constantes Kp Ki Kd adquiridas experimentalmente """
    pid_position = PID(k_p, k_i, k_d, set_point=0)
    k_p = 0.4
    k_i = 0.15
    k_d = k_p * 0.0474
    pid_orientation = PID(k_p, k_d, k_i, set_point=0)
    controller = FredericoController(position_controller=pid_position,
                                     orientation_controller=pid_orientation)

    return controller


def main():

    federico_controller = create_federico_controller()

    controller: tf.keras.Model = tf.keras.models.Sequential([
        # tf.keras.layers.Input(shape=(1,5)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),

        # tf.keras.layers.Dense(2, bias_initializer="ones"),
        tf.keras.layers.Dense(3)
    ])
    kinematic: tf.keras.Model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(4, "elu"),
        tf.keras.layers.Dense(2),
    ])

    environment: tf.keras.Model = tf.keras.models.Sequential([
        # tf.keras.layers.Input(shape=(1,5)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),

        # tf.keras.layers.Dense(2, bias_initializer="ones"),
        tf.keras.layers.Dense(5)
    ])

    def load_model(model: tf.keras.Model, path: str):
        if Path(path).exists():
            model.load_weights(path)

    def compile_model_with_mse_error(model: tf.keras.Model, lr: float = 0.001):
        model.compile(tf.keras.optimizers.RMSprop(learning_rate=lr),
                      metrics=["mae"],
                      loss=tf.keras.losses.mse)

    load_model(kinematic, KINEMATIC_PATH)
    load_model(controller, CONTROLLER_PATH)
    load_model(environment, ENVIRONMENT_PATH)

    compile_model_with_mse_error(kinematic)
    compile_model_with_mse_error(controller, lr=0.0005)
    compile_model_with_mse_error(environment)

    sim = Coppeliasim()

    sim.enable_step()
    sim.start_simulation()

    def is_goal(target_pos) -> bool:
        thirty_centimeters = 0.3
        distance = np.sqrt(target_pos[0]**2 + target_pos[1]**2)
        return distance < thirty_centimeters

    def memory_to_numpy(memory: list[tuple[Array32, Array32]]) -> tuple[Array32, Array32]:
        x = list()
        y = list()
       
        for i, o in memory:
            x.append(i)
            y.append(o)
           

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
       

        return x, y

    iterations = 0
    max_iterations = 10_000
    arrived_times = 0
    memory_kinematic = list()
    memory_controller = list()
    simulation_data = list()
    batch_size = 16

    for iterations in range(1, max_iterations+1):

        if iterations % 18 == 0:
            sim.move_target()
            sim.reset_robot_pos()
            sim.set_robot_random_orientation()

        theta_1 = sim.get_orientation_z()
        s_1 = np.array(sim.get_robot_pos()[:2], dtype=np.float32)
        t_1 = sim.get_time()
        target_pos_1 = np.array(sim.get_target_pos()[
                                :2], dtype=np.float32)

        # desired_pos = np.array(sim.get_target_pos()[:2])  # np.array(
        #     [0.0, 0.0, target_pos_1[0], target_pos_1[1], theta_1], dtype=np.float32)
        # controller_vel = controller.predict(desired_pos.reshape((1, -1)))[0]

        # if np.random.random() > 2: #>  exploration_rate(iterations):
        #     print('pred!')

        #     vel_wheels = kinematic.predict(
        #         input_kinematic_controller.reshape((1, -1)))[0][:2]

        #     print(
        #         f'vel controller {input_kinematic_controller} -> vel wheels {vel_wheels} ')
        # else:
        vel_wheels = np.random.sample((2)).astype(np.float32)

        sim.set_Velocity(2*float(vel_wheels[0]), 2 * float(vel_wheels[1]))

        simulation_state_t1 = [t_1, s_1[0], s_1[1], theta_1,
                               target_pos_1[0], target_pos_1[1], vel_wheels[0], vel_wheels[1]]

        simulation_data.append(simulation_state_t1)

        for _ in range(10):
            sim.step()

        target_pos_2 = np.array(sim.get_target_pos()[
                                :2], dtype=np.float32)
        t_2 = sim.get_time()
        s_2 = np.array(sim.get_robot_pos()[:2], dtype=np.float32)
        theta_2 = sim.get_orientation_z()
        simulation_state_t2 = [t_2, s_2[0], s_2[1], theta_2,
                               target_pos_2[0], target_pos_2[1], vel_wheels[0], vel_wheels[1]]

        simulation_data.append(simulation_state_t2)
        dt = t_2 - t_1
        ds = s_2 - s_1
        dtheta = theta_2 - theta_1

        dv = ds/dt
        domega = dtheta/dt

        input_kinematic = np.array(
            [dv[0], dv[1], domega, theta_1], dtype=np.float32)

        input_controller = np.array(
            [target_pos_2[0], target_pos_2[1], target_pos_1[0], target_pos_1[1], theta_1], dtype=np.float32)

        output_kinematic = np.array([vel_wheels[0], vel_wheels[1]],
                                    dtype=np.float32)
        output_controller = input_kinematic.copy()

        print(f'target distance {np.linalg.norm(target_pos_2)}')
        print(
            f'input controller:{input_controller} output controller {output_controller}')
        print(
            f'input kinematic:{input_kinematic} output kinematic {output_kinematic}')
        print(f'iterations: {iterations} arrived times: {arrived_times}')

        print(
            f'pred kinematic:{input_kinematic} output kinematic {vel_wheels}')

        memory_kinematic.append(
            (input_kinematic, output_kinematic))
        # memory_controller.append(
        #     (input_controller, output_controller, desired_pos))

        # if is_goal(target_pos_2):
        #     arrived_times += 1
        #     sim.move_target()

        # if len(memory_kinematic) >= 10_000:
        #     [memory_kinematic.pop(1) for _ in range(4_000)]

        if len(memory_kinematic) >= batch_size:
            continue

            mini_batch = batch_size // 2
            batch = random.sample(memory_kinematic, batch_size)

            x, y = memory_to_numpy(batch)

            kinematic.fit(x, y.reshape((-1)), batch_size=mini_batch,
                          validation_split=0.2,)

            batch = random.sample(memory_controller, batch_size)

    sim.stop_simulation()

    with open('memory.npy', 'wb') as f:
        x, y = memory_to_numpy(memory_kinematic)
        np.save(f, x)
        np.save(f, y)
        
        print('saved kinematic memory data!')

    with open('simulation.npy','wb') as f:
        np.save(f, np.array(simulation_data,dtype=np.float32))
        print('saved simulation data!')

def exploration_rate(n: int, min_rate=0.1) -> float:
    """Decaying exploration rate"""
    return max(min_rate, min(1, 1.0 - np.log10((n + 1) / 25)))


def manual_controller() -> tuple[float, float]:
    number = int(input('digite um número de 1 9\n'))

    match number:
        case 1:
            return 1, -1
        case 2:
            return -1, -1
        case 3:
            return -1, 1
        case 4:
            return 1, 0
        case 5:
            return 0, 0
        case 6:
            return 0, 1
        case 7:
            return 1, 0.5
        case 8:
            return 1, 1
        case 9:
            return 0.5, 1

    raise IndexError


if __name__ == '__main__':
    main()


"""

# match np.random.randint(0,4):
            #     case 0:
            #         vel_x = 1.0
            #         vel_y = 0.0
            #     case 1:
            #         vel_x = 0.0
            #         vel_y = 1.0
            #     case 2:
            #         vel_x = 0.0
            #         vel_y = -1.0

            #     case 3:
            #         vel_x = -1.0
            #         vel_y = 0.0
            #     case _:
            #         raise ValueError('error on rand int')


# angular_vel = 0.0
            # velocity = np.array([vel_y,vel_x,angular_vel]).reshape((1, -1))


   # space_l = np.linspace(0,1)
            # space_r = np.linspace(0,1)
            # j = iterations % len(space_l)
            # i = iterations // len(space_l)
            #  (space_l[i],space_r[j])

"""
