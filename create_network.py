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

def create_network():
    # SET UP PLANT
    plant = mn.plants.RigidTendonArm26(muscle_type=RigidTendonHillMuscleThelen(), visual_delay=0.05, proprioceptive_delay=0.02)
    # SET UP NETWORK
    n_units = 250
    # original: kernal regularizer = 1e-7, recurrent regularizer = 1e-5
    network = mn.nets.layers.GRUNetwork(plant=plant, n_units=n_units, kernel_regularizer=1e-9, name='network', recurrent_regularizer=1e-9)
    # SET UP TASK
    start_position = [0.785, 1.570] # [45,90] deg
    go_cue_range = (.100, .300)
    task = plg_tasks.CentreOutFF(network=network, start_position=start_position, go_cue_range=go_cue_range)
    task.network.do_recompute_inputs = True
    task.network.recompute_inputs = task.recompute_inputs
    # SET UP RNN
    rnn = tf.keras.layers.RNN(cell=network, return_sequences=True, name='RNN')
    input_dict = task.get_input_dict_layers()
    state0 = task.get_initial_state_layers()
    states_out = rnn(input_dict, initial_state=state0)
    # SET UP MAIN NETWORK OBJECT
    nn = mn.nets.MotorNetModel(inputs=[input_dict, state0], outputs=states_out, name='model', task=task)
    nn.compile(optimizer=tf.optimizers.Adam(clipnorm=1.), loss=task.losses, loss_weights=task.loss_weights)
    return nn