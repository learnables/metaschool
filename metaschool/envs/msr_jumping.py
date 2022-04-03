#!/usr/bin/env python

import torch
import gym


class DefaultColors:
    white = 1.0
    gray = 0.5
    black = 0.0


class MSRJumpEnv(gym.Env):

    """
    <a href="https://github.com/learnables/metaschool/blob/master/metaschool/envs/msr_jumping.py" class="source-link">[Source]</a>

    ## Description

    Bare bone re-implementation of the MSR Montreal Jumping task.

    ## References

    1. Tachet des Combes et al. 2018. "Learning Invariances for Policy Generalization" \
            ICLR 2018 Workshop Track

    ## Example

    ~~~python
    env = MSRJumpEnv(
        screen_width=64,
        screen_height=64,
        obstacle_position=25,
        agent_speed=2,
        jumping_height=15,
        obstacle_size=(10, 10),
        agent_size=(10, 10),
    )
    env.reset()
    while True:
        action = env.get_optimal_action()
        s, r, d, _ = env.step(action)
        if d: break
    print('Max achievable reward:', env.reward_range[1])
    ~~~

    """

    def __init__(
        self,
        floor_height=10,
        obstacle_position=20,
        obstacle_size=(9, 10),
        vision_observations=True,
        screen_width=84,
        screen_height=84,
        jumping_height=15,
        agent_speed=1,
        agent_size=(5, 10),
        colors=None,
        device=None,
        num_envs=1,
        max_episode_steps=600,
    ):
        """
        ## Arguments

        * `floor_height` (int, *optional*, default=10) - Height of the floor.
        * `obstacle_position` (int, *optional*, default=20) - X-position of the obstacle.
        * `obstacle_size` (tuple, *optional*, default=(9, 10)) - Dimensions of obstacle.
        * `vision_observations` (bool, *optional*, default=True) - Use vision observations or physical state.
        * `screen_width` (int, *optional*, default=84) - Witdth of the screen.
        * `screen_height` (int, *optional*, default=84) - Height of the screen
        * `jumping_height` (int, *optional*, default=15) - Height of agent's jump.
        * `agent_speed` (int, *optional*, default=1 - Speed of agent.
        * `agent_size` (tuple, *optional*, default=(5, 10)) - Dimensions of agent.
        * `colors` (class, *optional*, default=None) - Color values for visual observations.
        * `device` (torch.device, *optional*, default=None) - Device for observation tensors.
        * `num_envs` (int, *optional*, default=1) - Number of parallel environments (unsupported).
        * `max_episode_steps` (int, *optional*, default=600) - Horizon length.
        """
        super(MSRJumpEnv, self).__init__()
        assert jumping_height+1 >= agent_size[0] + obstacle_size[0], \
            'Task unsolvable: increase jumping height.'
        assert num_envs == 1, 'Multiple envs not supported, yet.'

        # dynamics
        self.vision_observations = vision_observations
        self.floor_height = floor_height
        self.obstacle_position = obstacle_position
        self.obstacle_size = obstacle_size
        self.colors = DefaultColors() if colors is None else colors
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.max_screen_size = max(self.screen_height, self.screen_width)

        # agent
        self.agent_size = agent_size
        self.agent_speed = agent_speed
        self.jumping_height = jumping_height
        self.device = device

        # properties
        self.num_envs = num_envs
        self.reward_range = (-1, self.screen_width + 2 - agent_size[0])
        self.action_space = gym.spaces.Discrete(2)
        if vision_observations:
            self.observation_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(num_envs, screen_height, screen_width),
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(num_envs, 10),
            )
        self._max_episode_steps = max_episode_steps

        # reset
        self.reset()

    def reset(self):
        self.agent_position = [0, self.floor_height]
        self.jumping = [False, 0]  # [mid-air, delta_y]
        self.done = False
        self._steps = 0
        self._base_observation = self._get_base_observation()
        return self.get_observation()

    def _step_dynamics(self, action):

        if not self.jumping[0] and action == 1:  # jump!
            self.jumping = [True, 1]

        # update x position
        self.agent_position[0] += self.agent_speed

        if self.jumping[0]:
            # update jump direction
            if self.agent_position[1] > self.floor_height + self.jumping_height:
                self.jumping[1] = -1

            # update y position
            self.agent_position[1] += self.jumping[1] * self.agent_speed

            # stop jump
            if self.agent_position[1] == self.floor_height:
                self.jumping = [False, 0]

    def step(self, action):
        # step dynamics
        self._step_dynamics(action)

        # decide termination
        timelimit = self._steps > self._max_episode_steps
        collision = \
            self.obstacle_position + self.obstacle_size[0] > self.agent_position[0] and \
            self.obstacle_position < self.agent_position[0] + self.agent_size[0] and \
            self.floor_height + self.obstacle_size[1] > self.agent_position[1] and \
            self.floor_height < self.agent_position[1] + self.agent_size[1]
        exited = self.screen_width < self.agent_position[0] + self.agent_size[0]
        done = timelimit or collision or exited

        # compute rewards
        if collision:
            reward = -1.0
        elif exited:
            reward = self.agent_speed + 1.0
        else:
            reward = self.agent_speed

        # get observation
        observation = self.get_observation()

        self._steps += 1
        return observation, reward, done, None

    def render(self, mode='human'):
        if mode == 'text':
            raise NotImplementedError()
        elif mode == 'human':
            raise NotImplementedError()
        elif mode == 'rgb_array':
            return self.get_observation(True) * 255.0

    def _get_base_observation(self, vision=None):
        if vision is None:
            vision = self.vision_observations
        if vision:
            # draw background
            frame = torch.ones(
                size=(self.screen_height, self.screen_width),
                dtype=torch.float32,
                device=self.device,
            ) * self.colors.black

            # draw obstacle
            frame[
                self.obstacle_position:self.obstacle_position+self.obstacle_size[0],
                self.floor_height:self.floor_height+self.obstacle_size[1],
            ].fill_(self.colors.gray)

            # draw screen outline
            frame[0:self.screen_height, 0].fill_(self.colors.white)
            frame[0:self.screen_height, self.screen_width-1].fill_(self.colors.white)
            frame[0, 0:self.screen_width].fill_(self.colors.white)
            frame[self.screen_height-1, 0:self.screen_width].fill_(self.colors.white)

            # draw floor
            frame[0:self.screen_width, self.floor_height].fill_(self.colors.white)
            return frame
        else:
            return torch.tensor([
                self.agent_position[0],
                self.agent_position[1],
                self.agent_size[0],
                self.agent_size[1],
                self.agent_speed,
                self.jumping_height,
                self.floor_height,
                self.obstacle_position,
                self.obstacle_size[0],
                self.obstacle_size[1],
            ], device=self.device) / self.max_screen_size

    def get_observation(self, vision=None):
        if vision is None:
            vision = self.vision_observations

        if self.vision_observations == vision:
            obs = self._base_observation.clone()
        else:
            obs = self._get_base_observation(vision)

        if vision:
            obs[
                self.agent_position[0]:self.agent_position[0]+self.agent_size[0],
                self.agent_position[1]:self.agent_position[1]+self.agent_size[1],
            ].fill_(self.colors.white)
            return obs.t().unsqueeze(0)
        else:
            obs[0] = self.agent_position[0] / self.max_screen_size
            obs[1] = self.agent_position[1] / self.max_screen_size
            return obs

    def get_optimal_action(self):
        """
        Returns the optimal action given current state of the world.
        """
        dist_to_obstacle = abs(self.obstacle_position - self.agent_position[0])
        jump_margin = self.agent_size[0] + self.obstacle_size[1]
        return int(dist_to_obstacle <= jump_margin - 1)


if __name__ == "__main__":

    # Test optimal action
    env = MSRJumpEnv(
        screen_width=64,
        screen_height=64,
        obstacle_position=25,
        agent_speed=2,
        #  jumping_height=15,
        #  obstacle_size=(10, 10),
        #  agent_size=(10, 10),
    )

    random_rewards = 0.0
    env.reset()
    while True:
        _, r, d, _ = env.step(env.action_space.sample())
        random_rewards += r
        if d:
            break
    print('Random rewards:', random_rewards)

    optimal_rewards = 0.0
    env.reset()
    while True:
        action = env.get_optimal_action()
        _, r, d, _ = env.step(action)
        #  print(action, env.agent_position)
        optimal_rewards += r
        if d:
            break
    print('Optimal rewards:', optimal_rewards)
    print('Max rewards:', env.reward_range[1])
