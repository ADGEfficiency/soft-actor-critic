class RandomPolicy():
    def __init__(self, env):
        self.env = env
        assert all(abs(self.env.action_space.low) == abs(self.env.action_space.high))

    def __call__(self, observation=None):
        unscaled = self.env.action_space.sample().reshape(1, *self.env.action_space.shape)
        scaled = unscaled / abs(self.env.action_space.high)
        return scaled, None, None


def make_random_policy(env):
    return RandomPolicy(env)
