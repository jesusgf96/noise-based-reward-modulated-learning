import torch
import torch.nn as nn


class ANN(nn.Module):
    def __init__(self, net_structure, batch_size, act_func, out_act_function, gamma_avg, seed=42, device=torch.device("cpu")):
        super().__init__()
        self.net_structure = net_structure
        self.fwd_layers = []
        self.x = []
        self.x_hist = []
        self.a = []
        self.a_hist = []
        self.a_avg = []
        self.a_avg_hist = []
        self.gamma_avg = gamma_avg
        self.eyes = []
        self.batch_size = batch_size
        self.act_func = act_func
        self.out_act_function = out_act_function
        self.device = device
        torch.manual_seed(seed)

        # Create layers and states
        self.a.append(None)
        self.a_hist.append(None)
        self.a_avg.append(None)
        self.a_avg_hist.append(None)
        self.x.append(None)
        self.x_hist.append(None)
        self.eyes.append(torch.eye(self.net_structure[0]))
        for indx in range(len(self.net_structure) - 1):
            self.a.append(None)
            self.a_hist.append(None)
            self.a_avg.append(None)
            self.a_avg_hist.append(None)
            self.x.append(None)
            self.x_hist.append(None)
            self.eyes.append(torch.eye(self.net_structure[indx+1]))
            self.fwd_layers.append(nn.Linear(net_structure[indx], net_structure[indx + 1], bias=False))
            torch.nn.init.xavier_normal_(self.fwd_layers[-1].weight)

        # This allows pytorch to manage moving everything to device
        self.fwd_mod = nn.ModuleList(self.fwd_layers)

        # Reset states
        self.reset_states()


    # Reset states and adjust the batch size
    def reset_states(self):
        for indx in range(len(self.net_structure)):
            self.x[indx] = torch.zeros(self.batch_size, self.net_structure[indx]).to(self.device)
            self.x_hist[indx] = [torch.zeros(self.batch_size, self.net_structure[indx]).to(self.device)]
            self.a[indx] = torch.zeros(self.batch_size, self.net_structure[indx]).to(self.device)
            self.a_hist[indx] = [torch.zeros(self.batch_size, self.net_structure[indx]).to(self.device)]
            self.a_avg[indx] = torch.zeros(self.batch_size, self.net_structure[indx]).to(self.device)
            self.a_avg_hist[indx] = [torch.zeros(self.batch_size, self.net_structure[indx]).to(self.device)]


    # Foward pass
    def forward(self, input_data, save_states=False, noise=False, noise_std=10e-6):

        # Noise per layer
        noises = []

        # Linear input
        self.a[0] = self.x[0] = torch.tensor(input_data, device=self.device)
        self.a_avg[0] = self.a_avg[0] + self.gamma_avg * (self.a[0] - self.a_avg[0])
        if save_states:
            self.a_hist[0].append(self.a[0])
            self.a_avg_hist[0].append(self.a_avg[0])
            self.x_hist[0].append(self.x[0])

        # Iterating hidden layers
        for indx in range(len(self.net_structure) - 2):
            self.a[indx+1] = self.x[indx] @ self.fwd_layers[indx].weight.transpose(0, 1)
            #NAS: should the next line not be after noise addition?
            self.a_avg[indx+1] = self.a_avg[indx+1] + self.gamma_avg * (self.a[indx+1] - self.a_avg[indx+1])
            if noise:
                noises.append(torch.randn_like(self.a[indx+1], device=self.device) * noise_std)
                self.a[indx+1] += noises[-1]
            self.x[indx+1] = self.act_func(self.a[indx+1])
            if save_states:
                self.a_hist[indx+1].append(self.a[indx+1])
                self.a_avg_hist[indx+1].append(self.a_avg[indx+1])
                self.x_hist[indx+1].append(self.x[indx+1])

        # Output layer
        self.a[-1] = self.x[-2] @ self.fwd_layers[-1].weight.transpose(0, 1)
        self.a_avg[-1] = self.a_avg[-1] + self.gamma_avg * (self.a[-1] - self.a_avg[-1])
        if noise:
            noises.append(torch.randn_like(self.a[-1], device=self.device) * noise_std)
            self.a[-1] += noises[-1]
        self.x[-1] = self.out_act_function(self.a[-1])
        if save_states:
            self.a_hist[-1].append(self.a[-1])
            self.a_avg_hist[-1].append(self.a_avg[-1])
            self.x_hist[-1].append(self.x[-1])

        # Return network output and noise
        return self.x[-1], noises


