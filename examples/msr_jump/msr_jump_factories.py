# -*- coding=utf-8 -*-

import random
import numpy as np
import gym
import cherry
import metaschool as ms


class MSRJumpFactory(ms.EnvFactory):

    def __init__(
        self,
        possible_heights=None,
        possible_positions=None,
        vision_observations=False,
        screen_size=60,
        frame_stack=1,
        device=None,
    ):
        if possible_heights is None:
            possible_heights = [i * 5 for i in range(1, 10)]
        if possible_positions is None:
            possible_positions = [i * 5 for i in range(3, 10)]
        self.possible_heights = possible_heights
        self.possible_positions = possible_positions
        self.vision_observations = vision_observations
        self.screen_size = screen_size
        self.frame_stack = frame_stack
        self.device = device

    def make(self, config):
        env = ms.envs.MSRJumpEnv(
            vision_observations=self.vision_observations,
            screen_height=self.screen_size,
            screen_width=self.screen_size,
            device=self.device,
            **config,
        )
        if self.vision_observations:
            if self.frame_stack > 1:
                env = gym.wrappers.FrameStack(env, self.frame_stack)
                env = cherry.wrappers.StateLambda(
                    env=env,
                    fn=lambda s: np.array(s).transpose(1, 0, 2, 3),
                )
            else:
                env = cherry.wrappers.StateLambda(env, lambda s: s.unsqueeze(0))
        return env

    def sample(self):
        config = ms.TaskConfig(
            floor_height=random.choice(self.possible_heights),
            obstacle_position=random.choice(self.possible_positions),
        )
        return config

    def all_configs(self):
        for height in self.possible_heights:
            for position in self.possible_positions:
                config = ms.TaskConfig(
                    floor_height=height,
                    obstacle_position=position,
                )
                yield config


class JumpWrapperFactory(ms.WrapperFactory):

    def sample(self, env):
        return env, None

    def wrap(self, env, config):
        env = cherry.wrappers.Torch(env)
        env = cherry.wrappers.Runner(env)
        return env
