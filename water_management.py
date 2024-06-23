import time
from pprint import pprint
import h5py
import torch
from matplotlib import pyplot as plt
from torch import nn
from core.learners.mones import MONES
from datetime import datetime
import uuid
from examples.susquehanna_river_simulation import create_susquehanna_river_env


class Actor(nn.Module):
    def __init__(self, nS, nA, hidden=50):
        super(Actor, self).__init__()

        self.nA = nA
        self.fc1 = nn.Linear(nS, hidden)
        self.fc2 = nn.Linear(hidden, nA)

        nn.init.xavier_uniform_(self.fc1.weight, gain=1)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)

    def forward(self, state):

        # actor
        a = self.fc1(state)
        a = torch.tanh(a)
        a = self.fc2(a)
        return a


def train_agent(logdir):
    number_of_observations = 1
    number_of_actions = 4
    agent = MONES(
        create_susquehanna_river_env,
        Actor(number_of_observations, number_of_actions, hidden=50),
        n_population=5,
        n_runs=2,
        logdir=logdir,
    )
    timer = time.time()
    agent.train(10)
    print(f"Training took: {time.time() - timer} seconds")

    print("Logdir:", logdir)
    torch.save({"dist": agent.dist, "policy": agent.policy}, logdir + "checkpoint.pt")


def run_agent(logdir):
    # Load agent
    checkpoint = torch.load(logdir)
    print(checkpoint)
    agent = checkpoint["policy"]

    timesteps = 12
    env = create_susquehanna_river_env()
    obs, _ = env.reset(seed=2137)
    for _ in range(timesteps):
        action = agent.forward(torch.from_numpy(obs).float())
        action = action.detach().numpy().flatten()
        print("Action:")
        pprint(action)
        (
            final_observation,
            final_reward,
            final_terminated,
            final_truncated,
            final_info,
        ) = env.step(action)
        print("Reward:")
        pprint(final_reward)


def show_logs(logdir):
    with h5py.File(logdir, "r") as f:
        # Print all root level object names (aka keys)
        # these can be group or dataset names
        print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
        a_group_key = list(f.keys())[0]

        # get the object type for a_group_key: usually group or dataset
        print(type(f[a_group_key]))

        # If a_group_key is a group name,
        # this gets the object names in the group and returns as a list
        data = list(f[a_group_key])
        print(data)

        group = f["train"]
        print("Hypervolume:", group["hypervolume"][()])
        print("Indicator metric:", group["metric"][()])
        # print(group['returns']['ndarray'][()])
        # print(group['returns']['step'][()])

        plt.plot(group["hypervolume"][()][:, 0], group["hypervolume"][()][:, 1])
        plt.show()


if __name__ == "__main__":
    logdir = "runs/"
    logdir += datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + str(uuid.uuid4())[:4] + "/"

    train_agent(logdir)

    # Trained agent path
    # logdir = "runs/2024-05-20_14-25-12_bca4/checkpoint.pt"
    # run_agent(logdir)

    # Read log file
    # logdir = "runs/2024-05-21_23-21-58_1c89/log.h5"
    # show_logs(logdir)
