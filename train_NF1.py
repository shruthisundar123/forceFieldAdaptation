import os
import json
import motornet as mn
import numpy as np

from create_network import create_network

from plotting_functions import hand_to_joints 
from plotting_functions import plot_simulations
from plotting_functions import plot1trial
from plotting_functions import print_training_log

from test_network import null_test
from test_network import curl_test

######################################
## create the network
nn = create_network()

####################################
# TRAIN NETWORK IN A NULL FIELD

n_t = 100
n_batches = 256
batch_size = 64

#to store results after each fit
myLargeArr = np.empty((0, 100, 4), float)
myTargetArr = np.empty((0, 100, 4), float)


# this callback logs training information for each batch passed, rather than for each epoch.
callbacks = [mn.nets.callbacks.BatchLogger()]
for i in range(30):
    print(i + 1)
    condition = "train"
    nn.task.angular_step = 15 # reset to original amount
    
    # generate inputs and initial states based on the task
    [inputs, targets, init_states] = nn.task.generate(n_timesteps=n_t, batch_size=batch_size*n_batches, condition=condition, ff_coefficient=0) #coefficient = 0 for null field   
    # fit the model
    # verbose = 1 will print the training losses
    h = nn.fit(x=[inputs, init_states], y=targets, verbose=1, epochs=1, batch_size=batch_size, shuffle=False, callbacks=callbacks)

    ## collect results to assess learning curve
    #get results
    condition = "test"
    n_mov_circle = 8 # number of movement directions around the unit circle
    n_t = 100
    nn.task.angular_step = 360 / n_mov_circle

    [inputs, targets, init_states] = nn.task.generate(n_timesteps=n_t, batch_size=n_mov_circle, condition=condition)
    results = nn([inputs, init_states], training=False)

    #store results 
    myLargeArr = np.append(myLargeArr, results['cartesian position'], axis=0)
    myTargetArr = np.append(myTargetArr, targets, axis=0)

######################################
# SAVE MODEL PARAMETERS
folderLocation = "save_NF1" + os.path.sep

# view training log
training_log = callbacks[0].history
print_training_log(folderLocation, log=training_log)

# save the trained model
weight_file = folderLocation + "weights.h5"
log_file = folderLocation + "log.json"
# data_file = "save_NF1" + os.path.sep + "data.pickle"
# cfg_file = "save_NF1" + os.path.sep + "cfg"

# save model weights
nn.save_weights(weight_file, save_format='h5')

cartesian_results_filename = folderLocation + "cartesian_position.npy"
targets_filename = folderLocation + "targets.npy"
np.save(cartesian_results_filename, myLargeArr)
np.save(targets_filename, myTargetArr)

# save model configuration ### NOT NEEDED
# if not os.path.isfile(cfg_file + ".json"):
#   nn.save_model(cfg_file)

# save training history (log)
with open(log_file, 'w') as file:
    training_log = callbacks[0].history
    json.dump(training_log, file)

print("Done saving null trained model.")

########################################
# TESTING THE NULL-TRAINED NETWORK

null_test(folderLocation, nn) #test in a null field
curl_test(folderLocation, nn) #test in a curl field

