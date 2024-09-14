
"""
Experiment configuration file

This file contains the configuration for the experiments to be run. Each experiment is a dictionary with the following keys:
"""

experiments = [
    {
        'name': 'experiment_1',
        'voxel_size': 0.03,
        'normalize': True,
        'batch_size': 16,
        'num_epochs': 20
    },
    {
        'name': 'experiment_2',
        'voxel_size': 0.03,
        'normalize': False,
        'batch_size': 4,
        'num_epochs': 40
    },
    {
        'name': 'experiment_3',
        'voxel_size': 0.04,
        'normalize': True,
        'batch_size': 16,
        'num_epochs': 500
    },
    {
        'name': 'experiment_4',
        'voxel_size': 0.04,
        'normalize': False,
        'batch_size': 16,
        'num_epochs': 40
    },
    {
        'name': 'experiment_5',
        'voxel_size': 0.05,
        'normalize': True,
        'batch_size': 16,
        'num_epochs':20
    },
    {
        'name': 'experiment_6',
        'voxel_size': 0.05,
        'normalize': False,
        'batch_size': 16,
        'num_epochs': 40
    },
]
