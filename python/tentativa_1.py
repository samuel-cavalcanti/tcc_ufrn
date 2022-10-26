
from coppeliasim import Coppeliasim
import numpy as np
import time
from matplotlib import pyplot as plt


def discrete_to_linear_continuos(x: int, limit: float, space: int) -> float:
    space = space - 1
    a = limit * 2.0 / space

    y = a * x - limit

    return y


def linear_continuos_to_discrete(y: float, limit: float, space: int) -> int:
    space = space - 1
    a = limit * 2.0 / space
    x = (y + limit) / a

    return round(x)


def plot_init():
    plt.ion()


def plot_close():
    plt.ioff()
    plt.show()


def plot_reward(rewards: list[float]):
    fig = plt.figure(1)
    plt.clf()

    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)


def main() -> None:

    dim = 50
    ac_dim = 15
    q_table = np.ones((dim, dim, dim, ac_dim, ac_dim)) * np.inf

    def policy(state: tuple[int, int, int]) -> tuple[int, int]:
        """Choosing action based on epsilon-greedy policy"""
        index_id = np.argmax(q_table[state])
        return index_id // q_table[state].shape[0], index_id % q_table[state].shape[0]

    def new_q_value(reward: float,  new_state: tuple, discount_factor=1) -> float:
        """Temporal difference for updating Q-value of state-action pair"""
        future_optimal_value = q_table[new_state][policy(new_state)]
        learned_value = reward + discount_factor * future_optimal_value
        return learned_value

    # Adaptive learning of Learning Rate
    def learning_rate(n: int, min_rate=0.01) -> float:
        """Decaying learning rate"""
        return max(min_rate, min(1.0, 1.0 - np.log10((n + 1) / 25)))

    def exploration_rate(n: int, min_rate=0.1) -> float:
        """Decaying exploration rate"""
        return max(min_rate, min(1, 1.0 - np.log10((n + 1) / 25)))

    sim = Coppeliasim()
    sim.enable_step()

    def train(episode: int) -> tuple[bool, float]:
        target_pos = sim.get_target_distance()
        z_ori = sim.get_orientation_z()
        current_state = (
            linear_continuos_to_discrete(target_pos[0], 5.8, q_table.shape[0]),
            linear_continuos_to_discrete(target_pos[1], 5.8, q_table.shape[1]),
            linear_continuos_to_discrete(z_ori, np.pi, q_table.shape[2]),
        )

        if np.random.random() > exploration_rate(episode):
            action = policy(current_state)
        else:
            action = np.random.random_integers(
                low=0, high=q_table.shape[3]-1, size=2)

        # action_in = input('digite uma ação: ')
        # action_in = action_in.replace(' ','')
        # left, right = action_in.split(',')
        # action = (int(left),int(right))
       


        left, right = action
        left = discrete_to_linear_continuos(left, 2.0, q_table.shape[3])
        right = discrete_to_linear_continuos(right, 2.0, q_table.shape[4])
        sim.set_Velocity(left, right)
        for _ in range(5):
            sim.step()

        target_pos = sim.get_target_distance()
        z_ori = sim.get_orientation_z()
        new_state = (
            linear_continuos_to_discrete(target_pos[0], 5.8, q_table.shape[0]),
            linear_continuos_to_discrete(target_pos[1], 5.8, q_table.shape[1]),
            linear_continuos_to_discrete(z_ori, np.pi, q_table.shape[2]),
        )

        distance = np.sqrt(target_pos[0]**2 + target_pos[1]**2)

        reward = 1/distance

        lr = learning_rate(episode)
        learn_t_value = new_q_value(reward, new_state)
        old_value = q_table[current_state][action]
        q_table[current_state][action] = (1-lr)*old_value + lr*learn_t_value

        message =\
            f"""
target pos: {target_pos}
Orientation: {z_ori}
current state{current_state} new state: {new_state}
distance = {distance}
reward = {reward}
current_episode {episode}
action [{left} , {right}]
"""
        print(message)
        thirty_centimeters = 0.3

        is_goal = distance < thirty_centimeters

        return is_goal, reward

    n_episodes = 10000
    thirty_seconds = 30
    reward_per_episode = list()
    sim.start_simulation()
    for e in range(n_episodes):

        done = False
        
        accumulative_reward = 0
        sim_start_time = sim.get_time()

        while done is False:
            try:
                is_goal, reward = train(episode=e)
                accumulative_reward += reward
                sim_time = sim.get_time()
                delta_t = sim_time - sim_start_time
                time_expired = delta_t > thirty_seconds
                done = is_goal or time_expired
                if is_goal:
                    sim.move_target()
            except KeyboardInterrupt:
                print('finishing training')
                np.save("table.npy", q_table)
                np.savetxt("rewards.csv", np.array(reward_per_episode), delimiter=',')
                exit(1)
                

        reward_per_episode.append(accumulative_reward)
        plot_reward(reward_per_episode)

        sim.set_Velocity(0.0, 0.0)
    sim.stop_simulation()
    np.save("table.npy", q_table)
    np.savetxt("rewards.csv", np.array(reward_per_episode), delimiter=',')


if __name__ == '__main__':
    main()
