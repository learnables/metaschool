# -*- coding=utf-8 -*-

# globals
import random
import numpy as np
import torch
import learn2learn as l2l
import wandb


def initial_setup(
    seed=1234,
    cuda=False,
    run_name=None,
    use_wandb=False,
    tags='',
    args=None,
):

    # seed
    device = torch.device('cpu')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed(seed)

    # init wandb
    if tags != '':
        tags = tags.split(',')
        tags = [t for t in tags if not t == '']

    if args is not None:
        config = l2l.utils.flatten_config(args)
    else:
        config = None

    wandb.init(
        project='dev',
        name=run_name,
        config=config,
        tags=tags,
        mode='online' if use_wandb else 'disabled',
    )
    return device
