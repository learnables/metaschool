
import torch
import cherry


class RandomPolicy(cherry.nn.Policy):

    def __init__(self, env):
        super(RandomPolicy, self).__init__()
        self.num_actions = env.action_space.n

    def forward(self, state):  # must return a density
        probs = torch.ones(self.num_actions) / self.num_actions
        density = cherry.distributions.Categorical(probs=probs)
        return density


class OptimalPolicy(cherry.nn.Policy):

    def __init__(self, env):
        super(OptimalPolicy, self).__init__()
        self.weight = torch.tensor([1, 0, 1, 0, 0, 0, 0, -1, 0, 1]).float()
        self.bias = torch.tensor([- 0.99 / env.max_screen_size]).float()

    def forward(self, state):
        preacts = torch.dot(self.weight, state) + self.bias
        acts = torch.tanh(preacts)
        logits = torch.tensor([-acts, acts]) * 1e5  # spiked distribution
        density = cherry.distributions.Categorical(logits=logits)
        return density


class MLPPolicy(cherry.nn.Policy):

    def __init__(self, input_size, output_size, hidden_size=128):
        super(MLPPolicy, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
        )

        self.actor = torch.nn.Linear(hidden_size, output_size)
        self.critic = torch.nn.Linear(hidden_size, 1)

    def value(self, state):
        return self.critic(self.features(state))

    def forward(self, state):
        logits = self.actor(self.features(state))
        density = cherry.distributions.Categorical(logits=logits)
        return density


class CNNPolicy(cherry.nn.Policy):

    def __init__(self, input_size, output_size):
        super(CNNPolicy, self).__init__()
        self.features = cherry.models.atari.NatureFeatures(1)
        self.actor = cherry.models.atari.NatureActor(512, output_size)
        self.critic = cherry.models.atari.NatureCritic(512)

        # replace ReLUs with GELUs
        for i, module in enumerate(self.features):
            if isinstance(module, torch.nn.ReLU):
                self.features[i] = torch.nn.GELU()

    def value(self, state):
        if state.ndim == 3:
            state = state.unsqueeze(1)
        return self.critic(self.features(state))

    def forward(self, state):
        if state.ndim == 3:
            state = state.unsqueeze(1)
        logits = self.actor(self.features(state))
        density = cherry.distributions.Categorical(logits=logits)
        return density