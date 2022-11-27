from constants import Array32
import numpy as np


SIM_DATA_PATH = 'simulation_data/sim_data.npy'
SIM_DATA_5_PATH = 'simulation_data/sim_data_amp_5.npy'
SIM_DATA_ROTATE_PATH = 'simulation_data/sim_data_rotate.npy'
SIM_DATA_30MS_PATH = 'simulation_data/sim_data_30ms.npy'
MEMORY_PATH = 'simulation_data/memory.npy'
SIMULATION_DATA_PATH = 'simulation_data/simulation.npy'
DATA_CSV_PATH = 'simulation_data/data.csv'
ERROR_T_KINEMATIC_PATH = 'simulation_data/error_t_kinematic.npy'
ERROR_T_LINEAR_PATH = 'simulation_data/error_t_linear.npy'
ERROR_T_ANALYTIC_PATH = 'simulation_data/error_t_analytic.npy'

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


def save_dataset(x: Array32, y: Array32, path: str) -> None:
    with open(path, 'wb') as f:
        np.save(f, x)  # type: ignore
        np.save(f, y)  # type: ignore


def load_error_t_dataset(path:str) -> tuple[Array32, Array32]:
    """
        x = time in seconds
        y = distance in meters, angle in rads

    """
    if path.find('error_t') == -1:
        raise Exception('Must be a error_t file')
    with open(ERROR_T_KINEMATIC_PATH, 'rb') as f:
        x = np.load(f)
        y = np.load(f)

    return x, y


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

    x_r, y_r = load_sim_data_rotate()

    x = np.vstack((x, x_r))
    y = np.vstack((y, y_r))

    return x, y


def load_sim_5_dataset()->tuple[Array32, Array32]:
    """
      x = simulation time,robot position,robot orientation, wheel velocity
        [time,robot_x,robot_y,robot_theta,wheel_left,wheel_right]

       y = simulation time,robot position,robot orientation,
         [time,robot_x,robot_y,robot_theta]
    """

    with open(SIM_DATA_5_PATH, 'rb') as f:
        x = np.load(f)
        y = np.load(f)


    return x, y

def load_sim_data_rotate() -> tuple[Array32, Array32]:
    """
      x = simulation time,robot position,robot orientation, wheel velocity
        [time,robot_x,robot_y,robot_theta,wheel_left,wheel_right]

       y = simulation time,robot position,robot orientation,
         [time,robot_x,robot_y,robot_theta]
    """
    with open(SIM_DATA_ROTATE_PATH, 'rb') as f:
        x_r = np.load(f)
        y_r = np.load(f)

        return x_r, y_r


def load_data_csv() -> Array32:
    """
        target pos relative to robot, robot wheel velocity
        [target_x,target_y,theta,wheel_left,wheel_right]
    """
    return np.loadtxt(DATA_CSV_PATH, skiprows=1, delimiter=',')  # type: ignore


def pre_processing_sim_data(data_t1: Array32, data_t2: Array32) -> tuple[Array32, Array32]:
    """
        data_t1 = simulation time,robot position,robot orientation, wheel velocity
        [time,robot_x,robot_y,robot_theta,wheel_left,wheel_right]

       data_t2 = simulation time,robot position,robot orientation,
         [time,robot_x,robot_y,robot_theta]

        x  = robot linear velocity,robot angular velocity
        y =  wheel_left,wheel_right

        return x,y
    """
    dt = data_t2[:, 0] - data_t1[:, 0]
    ds = data_t2[:, 1:3] - data_t1[:, 1:3]  # type: ignore
    theta_1 = data_t1[:, 3]
    theta_2 = data_t2[:, 3]

    dtheta = theta_2 - theta_1

    bigger_pi = dtheta > np.pi
    smaller_pi = dtheta < -np.pi

    dtheta[bigger_pi] = dtheta[bigger_pi] - 2*np.pi
    dtheta[smaller_pi] = 2*np.pi + dtheta[smaller_pi]

    dt = dt.reshape((-1, 1))
    dtheta = dtheta.reshape((-1, 1))
    theta_1 = theta_1.reshape((-1, 1))
    vel_lin = ds/dt
    vel_ang = dtheta/dt

    for i in range(len(vel_lin)):
        theta = theta_1[i, 0]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        vel_lin_x = vel_lin[i, 0]
        vel_lin_y = vel_lin[i, 1]

        vel_lin[i, 0] = vel_lin_x*cos_theta + vel_lin_y*sin_theta
        vel_lin[i, 1] = -vel_lin_x*sin_theta + vel_lin_y*cos_theta

    kinematic_input = np.hstack((vel_lin, vel_ang))

    kinematic_output = data_t1[:, -2:]

    return kinematic_input, kinematic_output
