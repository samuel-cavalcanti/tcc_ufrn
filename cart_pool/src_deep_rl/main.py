from itertools import count
from pathlib import Path
from matplotlib import pyplot
import torch
import numpy as np
from cart_pole import CartPole
import utils
from dnq_model import DNQModel
from replay_memory import ReplayMemory
from transition import Transition


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
NUMBER_OF_EPISODES = 10_000
MODEL_PATH = Path('models')
POLICY_MODEL_PATH = MODEL_PATH /'policy'
TARGET_MODEL_PATH = MODEL_PATH /'target'


def main() -> None:

    init_plot()

    cart_pole = CartPole()

    n_actions = cart_pole.get_actions().size

    def cart_pole_image_to_tensor() -> torch.Tensor:
        image = cart_pole.get_image()

        world_width = cart_pole.get_world_width()

        state = cart_pole.get_state()

        return utils.image_to_tensor(image, world_width, state[0])

    _, _, screen_height, screen_width = cart_pole_image_to_tensor().shape

    device = torch.device('cpu')

    policy_net = DNQModel(height=screen_height,
                          width=screen_width,
                          outputs=n_actions,
                          device=device).to(device)

    target_net = DNQModel(height=screen_height,
                          width=screen_width,
                          outputs=n_actions,
                          device=device).to(device)

    if POLICY_MODEL_PATH.exists():
        policy_net.load_state_dict(torch.load(POLICY_MODEL_PATH))
        policy_net.eval()

    if TARGET_MODEL_PATH.exists():
        target_net.load_state_dict(torch.load(TARGET_MODEL_PATH))
        target_net.eval()
    else:
         target_net.load_state_dict(policy_net.state_dict())

    optimizer = torch.optim.RMSprop(policy_net.parameters())

    memory = ReplayMemory(10_000)
  

    def select_action(state: torch.Tensor, steps: int):

        threshold = EPS_END + (EPS_START-EPS_END) * \
            np.exp(-steps/EPS_DECAY)

        if np.random.random() > threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[np.random.randint(0, n_actions)]], device=device, dtype=torch.long)

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return

        transitions = memory.rand_sample(BATCH_SIZE)
        no_final_mask = []
        non_final_next_states = []
        states = []
        actions = []
        rewards = []
        for t in transitions:
            is_not_final = t.next_state is not None
            if is_not_final:
                non_final_next_states.append(t.next_state)

            no_final_mask.append(is_not_final)
            states.append(t.state)
            actions.append(t.action)
            rewards.append(t.reward)

        no_final_mask = torch.tensor(no_final_mask)
        non_final_next_states = torch.cat(non_final_next_states)

        state_batch = torch.cat(states)
        actions_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, actions_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[no_final_mask] = target_net(
            non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values*GAMMA) + reward_batch

        # Compute Huber loss
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values,
                         expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            assert param.grad is not None
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    episode_durations = []
    steps = 0
    max_t = 20
    for i_episode in range(NUMBER_OF_EPISODES):
        cart_pole.reset()

        last_tensor = cart_pole_image_to_tensor()
        current_tensor = cart_pole_image_to_tensor()

        state = current_tensor - last_tensor

        for t in count():

            assert state is not None
            action = select_action(state, steps)
            steps += 1

            _, reward, done, _, _ = cart_pole.step(
                action.item())  # type: ignore

            reward = torch.tensor([reward], device=device)
            last_tensor = current_tensor
            current_tensor = cart_pole_image_to_tensor()

            if not done:
                next_state = current_tensor - last_tensor
            else:
                next_state = None

            # type: ignore
            memory.push(Transition(state, action, next_state, reward))
            print(f'time step: {t}')
            state = next_state

            optimize_model()

            if done:
                episode_durations.append(float(t + 1))

                plot_duration(episode_durations)
                break

            if i_episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

    print('Complete')
    cart_pole.close()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.save(target_net.state_dict(), MODEL_PATH / 'target.tch')
    torch.save(policy_net.state_dict(), MODEL_PATH / 'policy.tch')

    print('Please close the window to exit')
    pyplot.ioff()
    pyplot.show()


def init_plot():
    pyplot.ion()


def plot_duration(durations: list[float]):
    pyplot.figure(2)
    pyplot.clf()
    durations_tensor = torch.tensor(durations)
    pyplot.title('Training ...')
    pyplot.xlabel('Episode')
    pyplot.ylabel('Duration')
    pyplot.plot(durations_tensor.numpy())

    # if len(durations_tensor) >= 100:
    #     means = durations_tensor.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means))
    #     pyplot.figure(1)
    #     pyplot.plot(means.numpy())

    pyplot.pause(0.001)


if __name__ == '__main__':
    main()
