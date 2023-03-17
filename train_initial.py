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

def train_initial(trialNumber):

    ######################################
    print()
    print("RUNNING INITIAL TRAINING")
    print()

    ## CREATE THE NETWORK
    nn = create_network()

    ####################################
    ## TRAIN NETWORK IN A NULL FIELD

    n_t = 100
    n_batches = 256
    batch_size = 64

    #to store test targets and results after each fit
    # 100 data points in a single reach; each data point has 4 parameters (x, y, shoulder velocity, elbow velocity)
    myResultsArr = np.empty((0, 100, 4), float) 
    myTargetsArr = np.empty((0, 100, 4), float)

    # this callback logs training information for each batch passed, rather than for each epoch.
    callbacks = [mn.nets.callbacks.BatchLogger()]
    for i in range(50):
        print(i + 1)

        ## TRAINING
        condition = "train"
        nn.task.angular_step = 15 # reset to original amount
        
        # generate inputs and initial states based on the task
        [inputs, targets, init_states] = nn.task.generate(n_timesteps=n_t, batch_size=batch_size*n_batches, condition=condition, ff_coefficient=0) #coefficient = 0 for null field   
        # fit the model
        # verbose = 1 will print the training losses
        h = nn.fit(x=[inputs, init_states], y=targets, verbose=0, epochs=1, batch_size=batch_size, shuffle=False, callbacks=callbacks)

        ## TESTING
        # collect results to assess learning curve
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
    ## SAVE MODEL PARAMETERS

    print("Completed training and collected results")

    folderLocation = "save_initial" + os.path.sep

    # save the trained model
    weight_file = folderLocation + "weights_" + trialNumber + ".h5"
    log_file = folderLocation + "log_" + trialNumber + ".json"
    # data_file = "save_NF1" + os.path.sep + "data.pickle"
    # cfg_file = "save_NF1" + os.path.sep + "cfg"

    # save model weights
    nn.save_weights(weight_file, save_format='h5')

    # save training targets and results after each fit
    cartesian_results_filename = folderLocation + "cartesian_position_" + trialNumber + ".npy"
    targets_filename = folderLocation + "targets_" + trialNumber + ".npy"
    np.save(cartesian_results_filename, myResultsArr)
    np.save(targets_filename, myTargetsArr)

    # save model configuration ### NOT NEEDED
    # if not os.path.isfile(cfg_file + ".json"):
    #   nn.save_model(cfg_file)

    # save training history (log)
    with open(log_file, 'w') as file:
        training_log = callbacks[0].history
        json.dump(training_log, file)

    print("Done saving null trained model.")

    ########################################
    ## TESTING THE NULL-TRAINED NETWORK AND VIEW AND SAVE RESULTS

    # view training log
    training_log = callbacks[0].history
    print_training_log(folderLocation, log=training_log)

    null_test(folderLocation, nn) #test in a null field
    curl_test(folderLocation, nn) #test in a curl field

