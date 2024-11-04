import torch
from training import *


# Choose GPU
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")


# Parameters
seed =  15  # 42 101 300 482 708 94 15
task = 'acrobot' # 'acrobot' 'cartpole' 'reachingd1d'
algorithm = 'noisyNP' # 'NP' 'RMHL' 'BP' 'noisyNP'
hidden_units = 64 # 64 128
lr = 5e-2 # 5e-3 5e-2
gamma_avg = 0.66
noise_std = 0.001 
n_noisy_passes = 10 # only needed if using noisyNP
log_simulation = True


# Training loop
if task == 'acrobot':
    training_gym('Acrobot-v1', algorithm, noise_std, gamma_avg, hidden_units, lr, device, log_simulation, seed, n_noisy_passes)
elif task == 'cartpole':
    training_gym('CartPole-v1', algorithm, noise_std, gamma_avg, hidden_units, lr, device, log_simulation, seed, n_noisy_passes)
elif task == 'reachingd1d':
    if algorithm == 'noisyNP':
        print("Algorithm not implemented for this task")
        exit()
    training_neurogym(algorithm, noise_std, gamma_avg, hidden_units, lr, device, log_simulation, seed)
else:
    print('Task not available')
