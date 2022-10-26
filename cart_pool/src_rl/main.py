import gym
import numpy as np
from cart_pole import CartPole
from matplotlib import pyplot


def main():
    environment = CartPole()
    init_plot()
    no_buckets = (1, 1, 6, 3)
    no_actions = len(environment.get_actions())  # type: ignore
    state_bounds = [
        [-4.8, 4.8],
        [-0.5, 0.5],
        [-0.41887903, 0.41887903],
        [-0.8726646259971648, 0.8726646259971648]
    ]

    action_index = len(no_buckets)
    q_value_table = np.zeros(no_buckets + (no_actions,))
    min_explore_rate = 0.01
    min_learning_rate = 0.1

    max_episodes = 1000
    max_time_steps = 250
    streak_to_end = 120
    solved_time = 199
    discount = 0.99
    no_streaks = 0
    time_steps = []
    for episode_no in range(max_episodes):
        explore_rate = select_explore_rate(episode_no, min_explore_rate)
        learning_rate = select_learning_rate(episode_no, min_learning_rate)
        environment.reset()
        state = environment.get_state()

        start_state_value = bucketize_state_value(
            state, state_bounds, no_buckets)
        previous_state_value = start_state_value

        for time_step in range(max_time_steps):
            environment.get_image()
            selected_action = select_action(
                previous_state_value, explore_rate, environment, q_value_table)
            state, reward, done, _, _ = environment.step(selected_action)
            state_value = bucketize_state_value(
                state, state_bounds, no_buckets)
            best_q_value = np.amax(q_value_table[state_value])
            q_value_table[previous_state_value + (selected_action,)] += learning_rate * (
                reward + discount * (best_q_value) - q_value_table[previous_state_value + (selected_action,)])

            if done:
                time_steps.append(time_step+1)
                plot_duration(time_steps)
                print(
                    f'Episode {episode_no} finished after {time_step} time steps')
                if time_step >= solved_time:
                    no_streaks += 1
                else:
                    no_streaks = 0
                    break
        if no_streaks > streak_to_end:
            break
    np.save('model.npy', q_value_table)

    pyplot.ioff()
    pyplot.show()


def select_action(state_value, explore_rate, environment: CartPole, q_value_table: np.ndarray) -> int:
    if np.random.random() < explore_rate:
        action = environment.random_action()
    else:
        action = np.argmax(q_value_table[state_value])
    return action  # type: ignore


def select_explore_rate(x, min_explore_rate):
    return max(min_explore_rate, min(1, 1.0 - np.log10((x+1)/25)))


def select_learning_rate(x, min_learning_rate):
    return max(min_learning_rate, min(0.5, 1.0 - np.log10((x+1)/25)))


def bucketize_state_value(state_value: np.ndarray, state_value_bounds, no_buckets):

    bucket_indexes = []
    for i in range(len(state_value)):

        if state_value[i] <= state_value_bounds[i][0]:
            bucket_index = 0
        elif state_value[i] >= state_value_bounds[i][1]:
            bucket_index = no_buckets[i] - 1
        else:
            bound_width = state_value_bounds[i][1] - state_value_bounds[i][0]
            offset = (no_buckets[i]-1)*state_value_bounds[i][0]/bound_width
            scaling = (no_buckets[i]-1)/bound_width
            bucket_index = int(round(scaling*state_value[i] - offset))

        bucket_indexes.append(bucket_index)
    return tuple(bucket_indexes)


def init_plot():
    pyplot.ion()


def plot_duration(durations: list[float]):
    pyplot.figure(2)
    pyplot.clf()

    pyplot.title('Training ...')
    pyplot.xlabel('Episode')
    pyplot.ylabel('Duration')
    pyplot.scatter(list(range(len(durations))), durations)

    # if len(durations_tensor) >= 100:
    #     means = durations_tensor.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means))
    #     pyplot.figure(1)
    #     pyplot.plot(means.numpy())

    pyplot.pause(0.001)


if __name__ == '__main__':
    main()
