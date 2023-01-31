import numpy as np
import matplotlib.pyplot as plt

def hand_to_joints(xy, L1, L2):
      j = np.zeros(np.shape(xy))
      j[:,:,1] = np.arccos((xy[:,:,0]**2 + xy[:,:,1]**2 - L1**2 - L2**2) / (2*L1*L2))
      j[:,:,0] = np.arctan2(xy[:,:,1],xy[:,:,0]) - np.arctan2(L2*np.sin(j[:,:,1]),L1+(L2*np.cos(j[:,:,1])))
      return j

def plot_simulations(folder, name, xy, target_xy):
  target_x = target_xy[:, -1, 0]
  target_y = target_xy[:, -1, 1]
  plt.figure()
  nmov,ntime,_ = np.shape(xy)
  for i in range(nmov):
        plt.plot(xy[i,:,0],xy[i,:,1],color="b",linewidth=1.0)
  plt.scatter(target_x, target_y)
  plt.axis("equal")
  plt.xlabel("X (m)")
  plt.ylabel("Y (m)")
  #plt.show()
  saveLocation = folder + "TrainingLog" + name + ".png"
  plt.savefig(saveLocation)

def plot1trial(inputs, results, targets_j, nn, trial=0):
    j_results = results['joint position']
    m_results = results['muscle state']

    plt.figure(figsize=(14, 2.5)).set_tight_layout(True)

    plt.subplot(141)
    plt.plot(j_results[trial, :, 0]*180/np.pi, label='sho')
    plt.plot(j_results[trial, :, 1]*180/np.pi, label='elb')
    plt.plot(targets_j[trial, :, 0]*180/np.pi, '--')
    plt.plot(targets_j[trial, :, 1]*180/np.pi, '--')
    plt.axvline(np.where(inputs["inputs"][trial, :, -1] != 1)[0][0] - nn.task.network.visual_delay, c='grey')
    plt.legend()
    plt.xlabel('time (ms)')
    plt.ylabel('angle (deg)')

    plt.subplot(142)
    plt.plot(j_results[trial, :, 2], label='sho')
    plt.plot(j_results[trial, :, 3], label='elb')
    plt.legend()
    plt.xlabel('time (ms)')
    plt.ylabel('angle velocity (rad/sec)')

    plt.subplot(143)
    plt.plot(m_results[trial, :, 0, :])
    plt.xlabel('time (ms)')
    plt.ylabel('activation (a.u.)')
    plt.legend(["PEC","DEL","BRA","TRI","BIC","TR2"], loc=1)

    plt.subplot(144)
    plt.plot(m_results[trial, :, -1, :])
    plt.xlabel('time (ms)')
    plt.ylabel('force (N)')
    plt.legend(["PEC","DEL","BRA","TRI","BIC","TR2"], loc=1)

    #plt.show()

def print_training_log(folder, log):
  plt.figure().set_tight_layout(True)
  for kk, vv in log.items():
    plt.plot(vv, label=kk)

  plt.ylabel("Loss")
  plt.xlabel("Batch #")
  plt.legend()
  #plt.show()
  saveLocation = folder + "TrainingLog.png"
  plt.savefig(saveLocation)

