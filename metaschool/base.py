# -*- coding=utf-8 -*-

import gym

from typing import Sequence


class TaskConfig(dict):

    """
    <a href="https://github.com/learnables/metaschool/blob/master/metaschool/base.py" class="source-link">[Source]</a>

    ## Description

    Dictionary-like object to store task and wrapper configurations.

    ## Example

    ~~~python
    config = TaskConfig(seed=42, car='Volvo')
    config.location = 'Town2'
    config.weather = 'Cloudy'
    ~~~

    """

    def __setattr__(self, name, value):
        self[name] = value

    def __getattr__(self, name):
        if name in self:
            return self[name]
        return super(TaskConfig, self).__getattr__(name)


class EnvFactory(object):

    #  TODO: Add an option to enumerate the configs [with getitem?]

    """
    <a href="https://github.com/learnables/metaschool/blob/master/metaschool/base.py" class="source-link">[Source]</a>

    ## Description

    Base class to define how a task is created, and how its configuration is sampled.
    Task creation proceeds in two steps:

    1. Sampling of the random parameters of the task (see `sample()`).
    2. Instantiation of the task given chosen parameters (see `make()`).

    In a single-task setting, this is typically done with one large function (typically named `make_env`).

    ## Example

    ~~~python
    class DrivingFactory(EnvFactory):

        def make(self, config):
            env = gym.make('DrivingEnv-v0', **config)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env

        def sample(self):
            config = TaskConfig(location='Town2', weather='Sunny')
            config.color = random.choice(['Volvo', 'Mercedes', 'Audi', 'Hummer'])
            return config

    factory = DrivingFactory()
    train_task = factory.make(factory.sample())
    test_task = factory.make(factory.sample())
    ~~~

    """

    def make(self, config: TaskConfig) -> gym.Env:
        """
        ## Description

        Defines how to create a task, given a configuration.

        Use this method to wrap your base environment with the default wrappers.
        It's important that this method be deterministic when given the same
        configuration twice -- this allows to generate identical copies of the same task.

        ## Arguments

        * `config` (TaskConfig) - Configuration of the task.

        ## Returns

        * `env` (Env) - The gym environment defined according to config.

        """
        pass

    def sample(self) -> TaskConfig:
        """
        ## Description

        Defines the sampling procedure over the task parameters.

        Use this method to choose the axes of variation across tasks.
        For example, the above snippet ensures that the driving environment will
        always be in Town2 with a sunny weather; the car make is randomized.

        ## Returns

        * `config` (TaskConfig) - A randomly sampled task configuration for this environment.

        """
        pass


class WrapperFactory(object):

    """
    <a href="https://github.com/learnables/metaschool/blob/master/metaschool/base.py" class="source-link">[Source]</a>

    ## Description

    Base class to define how a task is created, and how its configuration is sampled.

    It is useful when some axes of variation across tasks are more easily implemented
    through wrappers around a `gym.Env`.
    It also allows for re-use of wrapper factories across projects.

    ## Example

    ~~~python
    class ChangingHorizonFactory(WrapperFactory):

        min_horizon = 20
        max_horizon = 200

        def wrap(self, env, config):
            return gym.wrappers.TimeLimit(env, max_episode_steps=config.max_steps)

        def sample(self, env=None):
            return TaskConfig(max_steps=random.randint(self.min_horizon, self.max_horizon))

    wrapper_factory = ChangingHorizonFactory()
    wrapper_config = wrapper_config.sample()
    task = wrapper_factory.wrap(task, wrapper_config)
    ~~~

    """

    def wrap(self, env: gym.Env, config: TaskConfig) -> gym.Env:
        """
        ## Description

        Wraps a gym environment according to the parameters defined in config.

        ## Arguments

        * `env` (Env) - The environment to wrap.
        * `config` (TaskConfig) - The configuration of the wrapper parameters.

        ## Returns

        * `env` (Env) - The wrapped environment.
        """
        pass

    def sample(self, env: gym.Env = None) -> TaskConfig:
        """
        ## Description

        Samples a parameter configuration for the current wrapper.

        ## Arguments

        * `env` (Env, *optional*, default=None) - The environment to be wrapped.

        ## Returns

        * `config` (TaskConfig) - The sampled parameter configuration.
        """
        pass


class WrapperFactoryList(list):

    """
    <a href="https://github.com/learnables/metaschool/blob/master/metaschool/base.py" class="source-link">[Source]</a>

    ## Description

    List-like class to turn a list of `WrapperFactory` into a single one.

    ## Example

    ~~~python
    wrapper_factories = WrapperFactoryList([
        MyWrapperFactory1(),
        MyWrapperFactory2(),
        MyWrapperFactory3(),
    ])
    task_config = env_factory.sample()
    task = env_factory.make(task_config)
    wrapper_configs = [wf.sample() for wf in wrapper_factories]
    task = wrapper_factories.wrap(task, wrapper_configs)
    ~~~

    """
    def wrap(self, env: gym.Env, configs: Sequence[TaskConfig]):
        for wrapper, config in zip(self, configs):
            env = wrapper.wrap(env, config)
        return env
