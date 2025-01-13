import torch
from training import *


# Choose GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Parameters
seed =  42  # 42 101 300 482 708 
task = 'acrobot' # 'acrobot' 'cartpole' 'reachingd1d'
algorithm = 'ours' # 'ours' 'RMHL' 'BP' 'noisy-ours'
hidden_units = 64 # 64 128
hidden_layers = 1
lr = 5e-2 # 1e-2 5e-3 5e-2
gamma_avg = 0.66
noise_std = 0.001 # 0.001 0.1
n_noisy_passes = 1 # 10 # only needed if using 'noisy-ours'
log_simulation = True


# Training loop
if task == 'acrobot':
    training_gym('Acrobot-v1', algorithm, noise_std, gamma_avg, hidden_units, hidden_layers, lr, device, log_simulation, seed, n_noisy_passes)
elif task == 'cartpole':
    training_gym('CartPole-v1', algorithm, noise_std, gamma_avg, hidden_units, hidden_layers, lr, device, log_simulation, seed, n_noisy_passes)
elif task == 'reaching1d':
    if algorithm == 'noisy-ours':
        print("Algorithm not implemented for this task")
        exit()
    training_neurogym(algorithm, noise_std, gamma_avg, hidden_units, hidden_layers, lr, device, log_simulation, seed)
else:
    print('Task not available')
