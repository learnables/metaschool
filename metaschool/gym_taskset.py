# -*- coding=utf-8 -*-

import gym

from collections import Iterable

from .base import WrapperFactoryList


class GymTaskset(object):

    #  TODO: Add an option to enumerate all possible tasks?

    """
    <a href="https://github.com/learnables/metaschool/blob/master/metaschool/gym_taskset.py" class="source-link">[Source]</a>

    ## Description

    A utility class to easily sample tasks and keeping track of previously seen tasks.

    ## Example

    ~~~python
    taskset = GymTaskset(
        env_factory=MyEnvFactory(),
        wrapper_factories=[
            MyWrapperFactory1(),
            MyWrapperFactory2(),
            MyWrapperFactory3(),
        ],
    )
    for iteration in range(num_iterations):
        train_task = taskset.sample()
        # ... train on train_task ...
        test_task = taskset.make_like(train_task)
        # ... test on freshly generated (and identical) test_task ...

    # enumerate sampled tasks
    for seen_task in iter(taskset):
        # ... process previously seen task ...
    ~~~

    """

    def __init__(self, env_factory, wrapper_factories=None):
        """
        ## Arguments

        * `env_factory` (EnvFactory) - An environment factory.
        * `wrapper_factories` (WrapperFactory, *optional*, default=None) - A list of wrapper factories.
        """
        self.env_factory = env_factory
        if wrapper_factories is None:
            wrapper_factories = []
        elif not isinstance(wrapper_factories, Iterable):
            wrapper_factories = [wrapper_factories, ]
        self.wrapper_factories = WrapperFactoryList(wrapper_factories)
        self._env_to_config = {}
        self._all_configs = []

    def sample(self) -> gym.Env:
        """
        ## Description

        Samples environment and wrapper configurations, and returns the instantiated
        wrapped environment.

        ## Returns

        * `env` (Env) - The wrapped environment with newly sampled configurations.
        """
        configs = []

        # sample env
        env_config = self.env_factory.sample()
        env = self.env_factory.make(env_config)
        configs.append(env_config)

        # sample wrappers
        for wrapper_factory in self.wrapper_factories:
            factory_config = wrapper_factory.sample(env)
            env = wrapper_factory.wrap(env, factory_config)
            configs.append(factory_config)

        # keep track of configs, envs
        self._env_to_config[id(env.unwrapped)] = len(self._all_configs)
        self._all_configs.append(configs)

        return env

    def make_like(self, env: gym.Env) -> gym.Env:
        """
        ## Description

        Given a previously sampled environment, instantiates a new copy based on its
        sampled (env and wrapper) configurations.

        Useful to have identical environment instances for training and testing.

        ## Arguments

        * `env` (Env) - The environment to copy.

        ## Returns

        * `env` (Env) - The newly created copy.
        """
        env_id = id(env.unwrapped)
        if env_id not in self._env_to_config:
            raise ValueError(
                'Env config not found in list of previous configs.'
            )
        return self[self._env_to_config[env_id]]

    def __getitem__(self, index: int) -> gym.Env:
        config = self._all_configs[index]
        env = self.env_factory.make(config[0])
        for wrapper_config, wrapper in zip(config[1:], self.wrapper_factories):
            env = wrapper.wrap(env, wrapper_config)
        return env

    def __len__(self):
        return len(self._all_configs)
