from constants import Array32
import numpy as np



SIM_DATA_PATH = 'simulation_data/sim_data.npy'
SIM_DATA_ROTATE_PATH ='simulation_data/sim_data_rotate.npy'
MEMORY_PATH = 'simulation_data/memory.npy'
SIMULATION_DATA_PATH = 'simulation_data/simulation.npy'
DATA_CSV_PATH ='simulation_data/data.csv'


def load_memory_dataset() -> tuple[Array32, Array32]:
    """
    x = linear velocity, angular velocity, robot orientation theta
        [vx,vy,dtheta/dt,robot_theta]

    y = wheel left velocity in rads/s, wheel right velocity in rads/s
        [wheel_left,wheel_right]
    """
    with open(MEMORY_PATH, 'rb') as f:
        x = np.load(f)
        y = np.load(f)

    return x, y


def load_simulation_dataset() -> Array32:
    """
        simulation time,robot position,robot orientation, wheel velocity
        [time,robot_x,robot_y,robot_theta,wheel_left,wheel_right]

    """
    with open(SIMULATION_DATA_PATH, 'rb') as f:
        s = np.load(f)

    return s


def load_sim_data_dataset() -> tuple[Array32, Array32]:
    """
      x = simulation time,robot position,robot orientation, wheel velocity
        [time,robot_x,robot_y,robot_theta,wheel_left,wheel_right]

       y = simulation time,robot position,robot orientation,
         [time,robot_x,robot_y,robot_theta]
    """
    with open(SIM_DATA_PATH, 'rb') as f:
        x = np.load(f)
        y = np.load(f)

    x_r,y_r = load_sim_data_rotate()


    x = np.vstack((x,x_r))
    y = np.vstack((y,y_r))


    return x, y

def load_sim_data_rotate()->tuple[Array32,Array32]:
    """
      x = simulation time,robot position,robot orientation, wheel velocity
        [time,robot_x,robot_y,robot_theta,wheel_left,wheel_right]

       y = simulation time,robot position,robot orientation,
         [time,robot_x,robot_y,robot_theta]
    """
    with open(SIM_DATA_ROTATE_PATH, 'rb') as f:
        x_r = np.load(f)
        y_r = np.load(f)

        return x_r,y_r

def load_data_csv() -> Array32:
    """
        target pos relative to robot, robot wheel velocity
        [target_x,target_y,theta,wheel_left,wheel_right]
    """
    return np.loadtxt(DATA_CSV_PATH, skiprows=1, delimiter=',')  # type: ignore
