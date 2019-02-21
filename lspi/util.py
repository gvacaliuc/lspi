import gym

from .policy import Policy


def generate_sample_data(num_samples, env: gym.Env, policy: Policy):
    """
    Example Usage:
        data = generate_sample_data(1e3, env, RandomPolicy(env))
    """
    state = env.reset()

    data = []

    for _ in range(int(num_samples)):
        action = policy(state)
        nextstate, reward, done, _ = env.step(action)
        data.append((state, action, reward, nextstate))

        if done:
            state = env.reset()
        else:
            state = nextstate

    return data
