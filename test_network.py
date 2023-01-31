import os
import sys
import json
import copy
import numpy as np
import tensorflow as tf
import motornet as mn
from motornet.tasks import Task
from motornet.plants import RigidTendonArm26
from motornet.plants.muscles import RigidTendonHillMuscleThelen
from motornet.nets.layers import GRUNetwork
from motornet.nets.models import MotorNetModel
from motornet.nets.losses import L2xDxActivationLoss, L2xDxRegularizer, PositionLoss
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.losses import LossFunctionWrapper
import plg_tasks
import matplotlib.pyplot as plt

from plotting_functions import hand_to_joints 
from plotting_functions import plot_simulations
from plotting_functions import plot1trial

# TEST NULL-TRAINED NETWORK IN A NULL FIELD
def null_test(folder, nn):
    condition = "test"
    name = "null"

    n_mov_circle = 8 # number of movement directions around the unit circle
    n_t = 100
    nn.task.angular_step = 360 / n_mov_circle

    [inputs, targets, init_states] = nn.task.generate(n_timesteps=n_t, batch_size=n_mov_circle, condition=condition)
    results = nn([inputs, init_states], training=False)

    #visualize the results
    plot_simulations(folder, name, xy=results["cartesian position"], target_xy=targets)

    L1,L2 = nn.task.network.plant.skeleton.L1, nn.task.network.plant.skeleton.L2
    targets_j = hand_to_joints(targets, L1, L2)
    n_mov = np.shape(results["joint position"])[0]
    for i in range(n_mov):
        plot1trial(inputs, results, targets_j, nn, i)


# TEST NETWORK IN A CURL FIELD
def curl_test(folder, nn):
    condition = "test"
    name = "curl"

    n_mov_circle = 8
    n_t = 150
    nn.task.angular_step = 360 / n_mov_circle

    [inputs, targets, init_states] = nn.task.generate(n_timesteps=n_t, batch_size=n_mov_circle, condition=condition, ff_coefficient=8)
    results = nn([inputs, init_states], training=False)

    plot_simulations(folder, name, xy=results["cartesian position"], target_xy=targets)

    L1,L2 = nn.task.network.plant.skeleton.L1, nn.task.network.plant.skeleton.L2
    targets_j = hand_to_joints(targets, L1, L2)
    n_mov = np.shape(results["joint position"])[0]
    for i in range(n_mov):
        plot1trial(inputs, results, targets_j, nn, i)