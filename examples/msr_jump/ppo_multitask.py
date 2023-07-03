# -*- coding=utf-8 -*-

import torch
import cherry
import metaschool as ms
import simple_parsing as sp
import tqdm
import wandb

import msr_jump_factories as mjf
import utils
import models


def main(args=None):
    vision_observations = True
    num_iterations = 1000
    train_tasks = 75
    train_episodes = 1
    eval_frequency = 1

    # initial setup
    device = utils.initial_setup(
        run_name='ppo-jump-cnn-multi-75ep',
        use_wandb=False,
        cuda=True,
    )

    # instantiate taskset
    taskset = ms.GymTaskset(
        env_factory=mjf.MSRJumpFactory(
            #  possible_heights=[15, ],
            #  possible_positions=[25, ],
            vision_observations=vision_observations,
            screen_size=84,
            device=device,
        ),
        wrapper_factories=mjf.JumpWrapperFactory(),
    )

    task = taskset.sample()
    eval_tasks = [taskset.wrapper_factories.wrap(
        taskset.env_factory.make(config), [config, None]
    ) for config in taskset.env_factory.all_configs()]

    # policy
    if vision_observations:
        policy = models.CNNPolicy(
            input_size=task.state_size,
            output_size=task.action_size,
        )
    else:
        policy = models.MLPPolicy(
            input_size=task.state_size,
            output_size=task.action_size,
        )
    policy.to(device)

    algo = cherry.algorithms.PPO()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    schedule = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer=optimizer,
        lr_lambda=lambda e: 0.995,
        #  lr_lambda=lambda e: 1.0,
    )

    total_steps = 0
    for iteration in tqdm.trange(num_iterations):
        stats = {
            'iteration': iteration,
            'episodes': iteration * train_episodes * train_tasks,
            'total_steps': total_steps,
        }

        # collect data
        replay = cherry.ExperienceReplay()
        for i in range(train_tasks):
            task = taskset.sample()
            task.reset()
            replay += task.run(policy.act, episodes=train_episodes)
        sum_rewards = replay.reward().sum().item()
        stats['train/rewards'] = sum_rewards / (train_tasks * train_episodes)
        total_steps += len(replay)

        # update policy
        replay = replay.to(device, non_blocking=True)
        update_stats = algo.update(
            replay=replay,
            optimizer=optimizer,
            policy=policy,
            state_value=policy.value,
        )
        stats.update(update_stats)
        schedule.step()

        # evaluate the policy
        if iteration % eval_frequency == 0:
            eval_replay = cherry.ExperienceReplay()
            for task in eval_tasks:
                task.reset()
                eval_replay += task.run(
                    lambda s: policy.act(s, deterministic=True),
                    episodes=1,
                )
            sum_rewards = eval_replay.reward().sum().item()
            stats['test/rewards'] = float(sum_rewards) / len(eval_tasks)

        wandb.log(stats)


if __name__ == "__main__":
    main()
