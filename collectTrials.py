from train_initial import train_initial
from train_NF1 import train_NF1
from train_CF1 import train_CF1
from train_NF2 import train_NF2
from train_CF2 import train_CF2
import os

import time
start_time = time.time()

# python3 collectTrials.py > training_progress.txt & disown -h
# ps aux #to check whether the process is running and for how long
# kill 123456 #to kill the process with process id 123456

print("opened collectTrials file")
print()

## try out 500 units
#train_initial("0", "500unitsTrial/save_initial/")

for i in range(20):
    print("------------------------")
    print("Participant number ", i)
    print("------------------------")

    participantNumber = str(i)
    train_initial(participantNumber, "new500UnitNetwork/save_initial/")
    train_NF1(participantNumber, "new500UnitNetwork/save_NF1/")
    train_CF1(participantNumber, "new500UnitNetwork/save_CF1/")
    train_NF2(participantNumber, "new500UnitNetwork/save_NF2/")
    train_CF2(participantNumber, "new500UnitNetwork/save_CF2/")

print()
print("Finished collectTrials file")
print("My program took", time.time() - start_time, "to run")
