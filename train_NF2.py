import os
import json
import motornet as mn
import numpy as np

from create_network import create_network
from plotting_functions import print_training_log

from test_network import null_test
from test_network import curl_test

def train_NF2(trialNumber):
    ##############################
    print()
    print("RUNNING TRAIN NF2")
    print()
    # SETUP

    # Recreate the model
    nn = create_network()

    # Load the weights obtained from the training in curl field 1
    weight_file = "save_CF1" + os.path.sep + "weights_" + trialNumber + ".h5"
    nn.load_weights(weight_file) #.expect_partial()

    ########################################
    # TRAIN NETWORK IN A NULL FIELD TO WASHOUT

    n_t = 100
    n_batches = 1 #256
    batch_size = 64 #change to 1 when actually adapting to represent human learning more

    #to store test targets and results after each fit
    myResultsArr = np.empty((0, 100, 4), float) 
    myTargetsArr = np.empty((0, 100, 4), float)

    # this callback logs training information for each batch passed, rather than for each epoch.
    callbacks = [mn.nets.callbacks.BatchLogger()]
    for i in range(30):
        print(i + 1)

        #TRAINING
        condition = "adapt" # to re-learn centre-out reaches in a given FF/NF
        nn.task.angular_step = 15 # reset to original amount

        #generate the inputs and initial states for the null field
        [inputs, targets, init_states] = nn.task.generate(n_timesteps=n_t, batch_size=batch_size*n_batches, condition=condition, ff_coefficient=0) #coefficient = 0 for null field
        h = nn.fit(x=[inputs, init_states], y=targets, verbose=0, epochs=1, batch_size=batch_size, shuffle=False, callbacks=callbacks)

        #TESTING
        ## collect results to assess learning curve
        condition = "test"
        n_mov_circle = 8 # number of movement directions around the unit circle
        n_t = 100
        nn.task.angular_step = 360 / n_mov_circle

        # generate inputs and get model's predictions
        [inputs, targets, init_states] = nn.task.generate(n_timesteps=n_t, batch_size=n_mov_circle, condition=condition, ff_coefficient=0)
        results = nn([inputs, init_states], training=False)

        #store targets and results for this fit
        myResultsArr = np.append(myResultsArr, results['cartesian position'], axis=0)
        myTargetsArr = np.append(myTargetsArr, targets, axis=0)


    ######################################
    # SAVE MODEL PARAMETERS
    folderLocation = "save_NF2" + os.path.sep

    # view training log
    training_log = callbacks[0].history
    print_training_log(folderLocation, log=training_log)

    # save the trained model
    weight_file = folderLocation + "weights_" + trialNumber + ".h5"
    log_file = folderLocation + "log_" + trialNumber + ".json"

    # save model weights
    nn.save_weights(weight_file, save_format='h5')

    # save training targets and results after each fit
    cartesian_results_filename = folderLocation + "cartesian_position_" + trialNumber + ".npy"
    targets_filename = folderLocation + "targets_" + trialNumber + ".npy"
    np.save(cartesian_results_filename, myResultsArr)
    np.save(targets_filename, myTargetsArr)

    # save training history (log)
    with open(log_file, 'w') as file:
        training_log = callbacks[0].history
        json.dump(training_log, file)

    print("Done saving washout (null trained again) model.")

    ########################################
    # TESTING THE WASHED OUT NETWORK AND VIEW AND SAVE RESULTS

    # view training log
    training_log = callbacks[0].history
    print_training_log(folderLocation, log=training_log)

    null_test(folderLocation, nn) #test in a null field
    curl_test(folderLocation, nn) #test in a curl field