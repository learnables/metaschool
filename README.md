<!-- <p align="center"><img src="assets/images/metaschool-full.png" height="120px" /></p> -->
<p align="center"><img src="https://raw.githubusercontent.com/learnables/metaschool/master/docs/assets/images/metaschool-full.png" height="120px" /></p>

--------------------------------------------------------------------------------

![Test Status](https://github.com/learnables/metaschool/workflows/Testing/badge.svg?branch=master)

`metaschool` provides a simple multi-task interface on top of OpenAI Gym environments.
Our hope is that it will simplify research in mutli-task, lifelong, and meta-reinforcement learning.


### Resources

* Website: [learnables.net/metaschool](https://learnables.net/metaschool)
* Slack: [slack.learn2learn.net](http://slack.learn2learn.net/)

### Supported Environments

* Mujoco Locomotion: [https://arxiv.org/abs/1703.03400](https://arxiv.org/abs/1703.03400)
* Meta-World: [https://arxiv.org/abs/1910.10897](https://arxiv.org/abs/1910.10897)
* MSR Jumping: [https://arxiv.org/abs/1809.02591](https://arxiv.org/abs/1809.02591)
* ... and more since metaschool is easily extensible.

## Mini Tutorial

At its essence, `metaschool` builds on 3 basic classes:

- [`EnvFactory`](http://learnables.net/metaschool/api/#metaschool.EnvFactory): A class to generate base Gym environments, given a configuration.
- [`WrapperFactory`](http://learnables.net/metaschool/api/#metaschool.WrapperFactory): An (optional) class to generate wrappers, given a configuration.
- [`TaskConfig`](http://learnables.net/metaschool/api/#metaschool.TaskConfig): A simple dict-like object to configure tasks.

Now, say we have a base Gym environment `DrivingEnv-v0` for training self-driving system in different conditions (locations, weather, car maker).
We can turn this environment into a set of multiple tasks as follows.

Note: we also use a [`GymTaskset`](http://learnables.net/metaschool/api/#metaschool.GymTaskset), which lets us automatically sample and keep track of tasks.

~~~python
import metaschool as ms

class DrivingFactory(ms.EnvFactory):  # defines how to sample new base environments

    def make(self, config):
        env = gym.make('DrivingEnv-v0', **config)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    def sample(self):
        config = ms.TaskConfig(location='Town2', weather='Sunny')  # TaskConfig is a dict-like configuration
        config.color = random.choice(['Volvo', 'Mercedes', 'Audi', 'Hummer'])
        return config

class ChangingHorizonFactory(ms.WrapperFactory):  # let's us randomize base envs with wrappers

    def wrap(self, env, config):
        return gym.wrappers.TimeLimit(env, max_episode_steps=config.max_steps)

    def sample(self, env=None):
        return ms.TaskConfig(max_steps=random.randint(20, 200))

taskset = GymTaskset(  # helps us create, replicate, and track tasks
    env_factory=DrivingFactory(),
    wrapper_factories=[ChangingHorizonFactory(), ],
)

for iteration in range(num_iterations):  # learning over multiple tasks
    train_task = taskset.sample()  # train_task is a TimeLimit(RecordEpisodeStatistics(DrivingEnv)) with randomized configurations
    learner.learn(train_task)

    for seen_task in iter(taskset):  # loops over all previously seen tasks
        loss = learner.eval(test_task)
~~~

## Changelog

A human-readable changelog is available in the [CHANGELOG.md](CHANGELOG.md) file.

## Citation

To cite this code in your academic publications, please use the following reference.

> Arnold, Sebastien M. R. “metaschool: A Gym Interface for Multi-Task Reinforcement Learning”. 2022.

You can also use the following Bibtex entry.

~~~bib
@software{Arnold20222022,
  author = {Arnold, Sebastien M. R.},
  doi = {10.5281/zenodo.1234},
  month = {12},
  title = {{metaschool: A Gym Interface for Multi-Task Reinforcement Learning}},
  url = {https://github.com/learnables/metaschool},
  version = {0.0.1},
  year = {2022}
}
~~~
