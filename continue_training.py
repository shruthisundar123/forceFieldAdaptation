import os
import json
import motornet as mn

from create_network import create_network

from plotting_functions import hand_to_joints 
from plotting_functions import plot_simulations
from plotting_functions import plot1trial
from plotting_functions import print_training_log

from test_network import null_test
from test_network import curl_test

# Recreate the model
nn = create_network()

# Load the weights obtained from the training in null field 1
weight_file = "save_NF1" + os.path.sep + "weights.h5"
nn.load_weights(weight_file) #.expect_partial()

# retrieve and plot training history
log_file = "save_NF1" + os.path.sep + "log.json"
with open(log_file, 'r') as file:
    loaded_training_log = json.load(file)

print_training_log(log=loaded_training_log)

####################################
# TRAIN NETWORK IN A NULL FIELD

condition = "train"

n_t = 100
n_batches = 256
batch_size = 64
# this callback logs training information for each batch passed, rather than for each epoch.
callbacks = [mn.nets.callbacks.BatchLogger()]
for i in range(10):
    print(i)
    
    # generate inputs and initial states based on the task
    [inputs, targets, init_states] = nn.task.generate(n_timesteps=n_t, batch_size=batch_size*n_batches, condition=condition, ff_coefficient=0) #coefficient = 0 for null field
    
    # fit the model
    # verbose = 1 will print the training losses
    h = nn.fit(x=[inputs, init_states], y=targets, verbose=1, epochs=1, batch_size=batch_size, shuffle=False, callbacks=callbacks)

# view training log
training_log = callbacks[0].history
print_training_log(log=training_log)

null_test(nn) #test in a null field
curl_test(nn) #test in a curl field