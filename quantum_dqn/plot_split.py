import numpy as np
from matplotlib import pyplot as plt

def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens), len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)

results_list = []
for i in range(10):
    filename = "reward_history_mv_avg_" + str(i) + ".txt"
    new_file = np.loadtxt(filename)
    results_list.append(new_file)

max_length = max([len(run) for run in results_list])
x = np.arange(max_length)

# # Saving the resulting plots
plt.figure(figsize=(10,5))
for i in range(len(results_list)):
    plt.plot(results_list[i], label=i+1)

plt.grid(True)
plt.legend(title="Run id")
plt.xlabel('Episode')
plt.ylabel('Average of collected rewards')
plt.savefig("quantum_dqn_avg_10_all_lines.png")
