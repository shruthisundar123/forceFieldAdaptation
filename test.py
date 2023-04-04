from train_conditions import train_model
from train_initial import train_initial
import os
import time
start_time = time.time()

# python3 test.py > training_progress.txt & disown -h
# ps aux #to check whether the process is running and for how long
# kill 123456 #to kill the process with process id 123456

folderOrigin = "2_500unitsTrial/"

for i in range(20):
    num = str(i)
    train_initial(num, folderOrigin + "save_initial/")
    train_model(num, "NF1", folderOrigin + "save_NF1/", folderOrigin + "save_initial/")
    train_model(num, "CF1", folderOrigin + "save_CF1/", folderOrigin + "save_NF1/")
    train_model(num, "NF2", folderOrigin + "save_NF2/", folderOrigin + "save_CF1/")
    train_model(num, "CF2", folderOrigin + "save_CF2/", folderOrigin + "save_NF2/")

print("My program took", time.time() - start_time, "to run")
