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

######################################
## create the network
nn = create_network()

####################################
# TRAIN NETWORK IN A NULL FIELD

condition = "train"

n_t = 100
n_batches = 256
batch_size = 64
# this callback logs training information for each batch passed, rather than for each epoch.
callbacks = [mn.nets.callbacks.BatchLogger()]
for i in range(30):
    print(i + 1)
    
    # generate inputs and initial states based on the task
    [inputs, targets, init_states] = nn.task.generate(n_timesteps=n_t, batch_size=batch_size*n_batches, condition=condition, ff_coefficient=0) #coefficient = 0 for null field
    
    # fit the model
    # verbose = 1 will print the training losses
    h = nn.fit(x=[inputs, init_states], y=targets, verbose=1, epochs=1, batch_size=batch_size, shuffle=False, callbacks=callbacks)

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

