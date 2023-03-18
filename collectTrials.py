from train_initial import train_initial
from train_NF1 import train_NF1
from train_CF1 import train_CF1
from train_NF2 import train_NF2
from train_CF2 import train_CF2

# python3 collectTrials.py > training_progress.txt & disown -h

print("opened collectTrials file")
print()

for i in range(20):
    print("------------------------")
    print("Participant number ", i)
    print("------------------------")

    participantNumber = str(i)
    train_initial(participantNumber)
    train_NF1(participantNumber)
    train_CF1(participantNumber)
    train_NF2(participantNumber)
    train_CF2(participantNumber)

print()
print("Finished collectTrials file")
