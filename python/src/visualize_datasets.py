from datasets import load_simulation_dataset, load_sim_data_dataset, load_memory_dataset
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm


def theta_comp(theta_sim_data, theta_simulation, theta_memory):

    plt_next_fig()
    plt.title("Comparação entre os ângulos coletados em simulações diferentes")
    plt.scatter(np.cos(theta_sim_data), np.sin(
        theta_sim_data), label='sim data theta')
    plt.scatter(np.cos(theta_simulation), np.sin(theta_simulation),
                s=5, c='r', label='simulation theta')

    plt.scatter(np.cos(theta_memory), np.sin(theta_memory),
                s=1, c='g', label='simulation theta')
    plt.ylim((-np.pi, np.pi))
    plt.legend()


def plt_next_fig() -> None:
    plt.figure(len(plt.get_fignums()) + 1)


def test_to_kinematic_data():

    t_1 = np.array([0.0, 10, 10, 1.0, 0.0, 0.0]).reshape((1, -1))
    t_2 = np.array([0.5, 5, 5, 5]).reshape((1, -1))

    dt = 0.5
    vel_lin = np.array([5-10, 5-10]) / dt
    vel_ang = (5-1)/dt

    k = to_kinematic_data(t_1, t_2)[0]

    assert k[0] == vel_lin[0]
    assert k[1] == vel_lin[1]
    assert k[2] == vel_ang
    assert k[3] == 1.0
    assert k[4] == 0.0
    assert k[5] == 0.0

    print('to_kinematic_data pass')


def to_kinematic_data(data_t1, data_t2):
    dt = data_t2[:, 0] - data_t1[:, 0]
    ds = data_t2[:, 1:3] - data_t1[:, 1:3]
    theta_1 = data_t1[:, 3]

    dtheta = data_t2[:, 3] - theta_1

    dt = dt.reshape((-1, 1))
    dtheta = dtheta.reshape((-1, 1))
    theta_1 = theta_1.reshape((-1, 1))
    vel_lin = ds/dt
    vel_ang = dtheta/dt

    # for i in range(len(vel_lin)):
    #     theta = theta_1[i, 0]
    #     r_0 = np.array([np.cos(theta), np.sin(theta), -
    #                     np.sin(theta), np.cos(theta)]).reshape((2, 2))

    #     vel_lin[i, :] = r_0 @ vel_lin[i]

    # kinematic_input = np.hstack((vel_lin, vel_ang))

    kinematic_input = np.hstack((vel_lin, vel_ang, theta_1))
    kinematic_output = data_t1[:, -2:]

    kinematic_data = np.hstack((kinematic_input, kinematic_output))

    return kinematic_data


def com_vel(vel_sim_data, vel_simulation, vel_memory):
    plt_next_fig()
    plt.title("Comparação entre os velocidades coletados em simulações diferentes")
    plt.scatter(vel_sim_data[:, 0], vel_sim_data[:, 1], label='sim data vel',s=10)
    plt.scatter(vel_simulation[:, 0], vel_simulation[:,
                1], s=5, c='r', label='simulation vel')
    plt.scatter(vel_memory[:, 0], vel_memory[:, 1],
                s=1, c='g', label='memory vel')
    plt.legend()

    plt_next_fig()
    plt.plot(vel_sim_data[:, 0], 'o', label='sim data vel x')
    plt.legend()


def com_vel_wheel(vel_wheel_sim_data, vel_wheel_simulation, vel_wheel_memory):
    plt_next_fig()
    plt.title("Comparação entre os velocidades das rodas coletados em simulações diferentes")
    plt.scatter(
        vel_wheel_sim_data[:, 0], vel_wheel_sim_data[:, 1], label='sim data vel wheel',s=10)
    plt.scatter(vel_wheel_simulation[:, 0], vel_wheel_simulation[:,
                1], s=5, c='r', label='simulation vel wheel')
    plt.scatter(vel_wheel_memory[:, 0], vel_wheel_memory[:,
                1], s=1, c='g', label='memory vel wheel')
    plt.legend()

    pass


def diff_vel_dtheta(data_sim_dtheta, sim_dtheta, memory_dtheta):
    plt_next_fig()
    plt.scatter(np.cos(sim_dtheta), np.sin(sim_dtheta),
                c='g', label='simulation dtheta/dt ',s=10)
    plt.scatter(np.cos(data_sim_dtheta), np.sin(data_sim_dtheta),
                s=5, c='c', label='sim data dtheta/dt')
    plt.scatter(np.cos(memory_dtheta), np.sin(memory_dtheta),
                s=1, c='b', label='memory dtheta/dt')
    plt.legend()


def main():
    simulation_dataset = load_simulation_dataset()
    sim_data_t1, sim_data_t2 = load_sim_data_dataset()
    x_memory, y_memory = load_memory_dataset()

    simulation_dataset[:, -2:] = simulation_dataset[:, -2:]*2.0  # type: ignore
    y_memory = y_memory*2.0
    size = sim_data_t1.shape[0]
    theta_comp(theta_sim_data=sim_data_t1[:size, 3],
               theta_simulation=simulation_dataset[:size, 3],
               theta_memory=x_memory[:size,-1])

    kinematic_sim_data = to_kinematic_data(
        data_t1=sim_data_t1, data_t2=sim_data_t2)

    index_t1 = np.arange(0, simulation_dataset.shape[0], 2)
    index_t2 = np.arange(1, simulation_dataset.shape[0], 2)

    simulation_t1 = simulation_dataset[index_t1]
    simulation_t2 = simulation_dataset[index_t2]

    kinematic_simulation = to_kinematic_data(simulation_t1, simulation_t2)

    kinematic_memory = np.hstack((x_memory, y_memory))

    def find_nearest_sample(sample, kinematic_data):
        robot_vel_s1 = sample[:-2]
        min_distance = np.inf
        near_sample = kinematic_data[0]
        for s in kinematic_data:
            robot_vel_s2 = s[:-2]
            distance = np.linalg.norm(robot_vel_s1-robot_vel_s2)  # +sample[3]
            if distance < min_distance:
                min_distance = distance
                near_sample = s

        return near_sample

    com_vel(kinematic_sim_data[:, :2],
            kinematic_simulation[:, :2], kinematic_memory[:, :2])

    diff_vel_dtheta(
        kinematic_sim_data[:, 2], kinematic_simulation[:, 2], kinematic_memory[:, 2])
    com_vel_wheel(
        kinematic_sim_data[:, -2:], kinematic_simulation[:, -2:], kinematic_memory[:, -2:])

    plt.show()
    # near_samples = list()
    # for sample in tqdm(kinematic_simulation):
    #     near_sample = find_nearest_sample(sample, kinematic_sim_data)
    #     near_samples.append(near_sample)

    # near_samples = np.array(near_samples, dtype=np.float32)

    # with open('near_samples_2.npy', 'wb') as f:
    #     np.save(f,near_samples)
    #     # near_samples= np.load(f)

    # [print(s) for s in kinematic_simulation[:10]]

    # print('-'*100)
    # [print(s) for s in near_samples[:10]]


if __name__ == '__main__':
    # test_to_kinematic_data()
    main()


"""
    # def by_distance(sample_a):
    #     robot_pos = sample_a[1:3]
    #     return np.linalg.norm(robot_pos)

    # sorted_simulation = sorted(simulation_dataset,key=by_distance)      # type: ignore
    # sorted_sim_data_t2 = sorted(sim_data_t2,key=by_distance)      # type: ignore

    # sorted_simulation = np.array(sorted_simulation,dtype=np.float32)

    # sorted_sim_data_t2 = np.array(sorted_sim_data_t2,dtype=np.float32)

"""
