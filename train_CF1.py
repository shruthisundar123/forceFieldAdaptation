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

# Recreate the model
nn = create_network()

# Load the weights obtained from the training in null field 1
weight_file = "save_NF1" + os.path.sep + "weights.h5"
nn.load_weights(weight_file) #.expect_partial()

########################################
# TRAIN NETWORK IN A CURL FIELD

condition = "adapt" # to re-learn centre-out reaches in a given FF/NF

n_t = 100
n_batches = 1 #256
batch_size = 1 #64 #change to 1 when actually adapting to represent human learning more

#to store results after each fit
myLargeArr = np.empty((0, 100, 4), float)

# this callback logs training information for each batch passed, rather than for each epoch.
callbacks = [mn.nets.callbacks.BatchLogger()]
for i in range(2000):
    #generate the inputs and initial states for the curl field
    # ff_coefficient=8 makes it the curl field
    print(i + 1)
    nn.task.angular_step = 15 # reset to original amount
    [inputs, targets, init_states] = nn.task.generate(n_timesteps=n_t, batch_size=batch_size*n_batches, condition=condition, ff_coefficient=8) #coefficient = 8 for curl field
    h = nn.fit(x=[inputs, init_states], y=targets, verbose=1, epochs=1, batch_size=batch_size, shuffle=False, callbacks=callbacks)

    ## collect results to assess learning curve
    #get results
    n_mov_circle = 8 # number of movement directions around the unit circle
    n_t = 100
    nn.task.angular_step = 360 / n_mov_circle

    [inputs, targets, init_states] = nn.task.generate(n_timesteps=n_t, batch_size=n_mov_circle, condition=condition)
    results = nn([inputs, init_states], training=False)

    #store results 
    tempArr = np.array(results['cartesian position'])
    myLargeArr = np.append(myLargeArr, tempArr, axis=0)

######################################
# SAVE MODEL PARAMETERS
folderLocation = "save_CF1" + os.path.sep

# view training log
training_log = callbacks[0].history
print_training_log(folderLocation, log=training_log)

# save the trained model
weight_file = folderLocation + "weights.h5"
log_file = folderLocation + "log.json"

# save model weights
nn.save_weights(weight_file, save_format='h5')

cartesian_results_filename = folderLocation + "cartesian_position.npy"
np.save(cartesian_results_filename, myLargeArr)

# save training history (log)
with open(log_file, 'w') as file:
    training_log = callbacks[0].history
    json.dump(training_log, file)

print("Done saving curl trained model.")

########################################
# TESTING THE CURL-TRAINED NETWORK

null_test(folderLocation, nn) #test in a null field
curl_test(folderLocation, nn) #test in a curl field