
import gym
import neurogym as ngym
from utils import * 
from models import *
import wandb
import copy
import numpy as np



def training_gym(env_name, algorithm, noise_std, gamma_avg, hidden_units, hidden_layers, lr, device, log_simulation, seed, n_noisy_passes):

    # Create environment
    env = gym.make(env_name)

    
    # Control randomness for reproducibility
    control_randomness(seed, env)


    # Network params
    n_inputs = env.observation_space.shape[0]
    n_actions = env.action_space.n
    net_structure = [n_inputs, n_actions]
    for _ in range(hidden_layers):
        net_structure.insert(1, hidden_units)
    print('Net structure:', net_structure)
    act_func = leaky_relu(0.1)
    out_act_function = softmax()


    # Instantiate agents (clean or approx + noisy pass) and move to GPU
    if algorithm == 'noisy-ours':
        agents = [ANN(net_structure=net_structure, batch_size=1, act_func=act_func,
                    out_act_function=out_act_function, gamma_avg=gamma_avg, seed=42, device=device) for _ in range(n_noisy_passes)]
        _ = [agent.to(device) for agent in agents]
    else:
        agent = ANN(net_structure=net_structure, batch_size=1, act_func=act_func,
                    out_act_function=out_act_function, gamma_avg=gamma_avg, seed=42, device=device)
        _ = agent.to(device)
    if algorithm == 'ours' or algorithm == 'noisy-ours':
        agent_noisy = ANN(net_structure=net_structure, batch_size=1, act_func=act_func,
                    out_act_function=out_act_function, gamma_avg=gamma_avg, seed=42, device=device)
        _ = agent_noisy.to(device)


    # Simulation parameters
    save_states = False
    if env_name == 'Acrobot-v1':
        num_episodes = 8000
    else:
        num_episodes = 20000
    discount = 1.0
    prob_actions = True
    explicit_noise = True
    if algorithm == 'BP':
        optimizer = torch.optim.SGD(agent.parameters(), lr=lr)
    if algorithm == 'RMHL':
        noise = True
    else:
        noise = False


    # Log experiment in wandb
    if log_simulation:
        if explicit_noise:
            name = 'e'
        else:
            name = ''
        name += str(algorithm)+'_hl'+str(hidden_layers)+'_units'+str(net_structure[1])+'_lr'+str(lr)
        if algorithm != 'BP':
            name += '_noise'+str(noise_std)
        name += '_gamma'+str(gamma_avg)

        if env_name == 'Acrobot-v1':
            wandb.init(
                project="reward_based_learning-Acrobot", name=name,
                config={}
        )
        else:
            wandb.init(
                project="reward_based_learning-Cartpole", name=name,
                config={}
            )


    # Initialize stuff
    reward_episodes=[]
    reward_avg = None


    # Iterate episodes
    for episode in range(num_episodes):

        # Initial stuff
        total_reward=0
        state = env.reset()
        if algorithm == 'noisy-ours':
            [agent.reset_states() for agent in agents]
        else:
            agent.reset_states()
        done = False
        log_probs = []
        rewards = []
        noises_hist = []
        e = [0 for _ in range(len(net_structure) - 1)]

        # If noisy+clean passes needed
        if algorithm == 'ours':
            agent_noisy.reset_states()

        # Run simulation
        while not done:

            # Agent computes policy
            if algorithm == 'noisy-ours': # approximate clean pass with multiple noisy passes
                policy_passes, noises_passes = [], []
                for agent in agents:
                    policy, noises = agent.forward(state, save_states=save_states, noise=noise, noise_std=noise_std)
                    policy_passes.append(policy)
                    noises_passes.append(noises)
            else:
                policy, noises = agent.forward(state, save_states=save_states, noise=noise, noise_std=noise_std)
            if algorithm == 'ours' or algorithm == 'noisy-ours':
                policy_noisy, noises = agent_noisy.forward(state, save_states=save_states, noise=True, noise_std=noise_std)
            noises_hist.append(noises)

            # Agent chooses action
            if prob_actions:
                action = int(torch.multinomial(policy, 1)) # Probabilistic choice of actions
            else:
                action = int(torch.argmax(policy)) # Always best action

            # Store log probs
            if algorithm == 'BP':
                log_probs.append(torch.log(policy[action]))

            # Execute action and collect reward
            state, reward, done, _ = env.step(int(action))
            total_reward += reward
            rewards.append(reward)

            ###---------- Elegibility trace ----------###

            # ours
            if algorithm == 'ours':
                for indx in range(len(agent.net_structure) - 1):
                    pre_out = agent_noisy.x[indx]
                    if explicit_noise:
                        norm_noise = noises[indx] / (torch.norm(torch.cat(noises), 'fro')**2)
                    else:
                        noise_layer = agent_noisy.a[indx+1] - agent.a[indx+1]
                        noise_network = torch.cat([agent_noisy.a[i+1] - agent.a[i+1] for i in range(len(agent.net_structure) - 1)])
                        norm_noise = noise_layer / (torch.norm(noise_network, 'fro')**2)
                    dp = torch.log(policy_noisy[action]) - torch.log(policy[action])
                    e[indx] = discount * e[indx] + dp * torch.outer(norm_noise, pre_out)

            # noisy ours
            if algorithm == 'noisy-ours':
                for indx in range(len(agent.net_structure) - 1):
                    pre_out = agent_noisy.x[indx]
                    if explicit_noise:
                        norm_noise = noises[indx] / (torch.norm(torch.cat(noises), 'fro')**2)
                    else:
                        noise_layer = agent_noisy.a[indx+1] - agent.a[indx+1]
                        noise_network = torch.cat([agent_noisy.a[i+1] - agent.a[i+1] for i in range(len(agent.net_structure) - 1)])
                        norm_noise = noise_layer / (torch.norm(noise_network, 'fro')**2)
                    clean_policy_approx = torch.mean(torch.stack([policy[action] for policy in policy_passes]))
                    dp = torch.log(policy_noisy[action]) - torch.log(clean_policy_approx)
                    e[indx] = discount * e[indx] + dp * torch.outer(norm_noise, pre_out)

            # RMHL
            if algorithm == 'RMHL':
                for indx in range(len(agent.net_structure) - 1):
                    pre_out = agent.x[indx]
                    if explicit_noise:
                        e[indx] = discount * e[indx] + torch.outer(noises[indx], pre_out)
                    else:
                        # Noise extracted via moving averages
                        post_act = agent.a[indx+1]
                        post_act_avg = agent.a_avg[indx+1]
                        e[indx] = discount * e[indx] + torch.outer(torch.squeeze(post_act-post_act_avg), pre_out)


        # Initialize moving average for the total reward
        if reward_avg is None:
            reward_avg = total_reward


        ###--------- RPE computation ---------###

        # Compute the RPE
        RPE = total_reward - reward_avg

        # Scale the RPE according to the absolute R
        RPE /= np.abs(total_reward)


        ###--------- Learning ---------###

        # Backpropagation
        if algorithm == 'BP':
            loss = []
            for log_prob in log_probs:
                loss.append(-log_prob * RPE)
                # loss.append(-log_prob * total_reward)
            loss = torch.sum(torch.stack(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # RMHL
        elif algorithm == 'RMHL':
            # Modulate eligibility trace with RPE
            for indx in range(len(agent.net_structure) - 1):
                dW = RPE * e[indx]
                agent.fwd_layers[indx].weight.data = agent.fwd_layers[indx].weight.data + lr * dW


        # ours
        elif algorithm == 'ours':
            # Modulate eligibility trace with RPE
            for indx in range(len(agent.net_structure) - 1):
                dW = RPE * e[indx]
                agent.fwd_layers[indx].weight.data = agent.fwd_layers[indx].weight.data + lr * dW
                # agent.fwd_layers[indx].weight.data = agent.fwd_layers[indx].weight.data + lr * agent.net_structure[indx+1] * dW
                agent_noisy.fwd_layers[indx].weight.data = copy.deepcopy(agent.fwd_layers[indx].weight.data)


        # noisyNP-like
        elif algorithm == 'noisy-ours':
            # Modulate eligibility trace with RPE
            for indx in range(len(agent.net_structure) - 1):
                dW = RPE * e[indx]
                agent_noisy.fwd_layers[indx].weight.data = agent_noisy.fwd_layers[indx].weight.data + lr * dW
                # agent_noisy.fwd_layers[indx].weight.data = agent.fwd_layers[indx].weight.data + lr * agent.net_structure[indx+1] * dW
                for i in range(n_noisy_passes):
                    agents[i].fwd_layers[indx].weight.data = copy.deepcopy(agent_noisy.fwd_layers[indx].weight.data)


        ###----------------------###

        # Compute reward average accross trials
        reward_avg += gamma_avg * (total_reward - reward_avg)

        # Print the total reward every 100 episodes
        if episode % 10 == 0:
            print('Episode', episode,', Total reward:', total_reward, ', Avg reward:', np.mean(reward_episodes[episode-100:episode]))

        # Keep track of the rewards
        reward_episodes.append(total_reward)

        # Save datapoint in weights and biases
        if log_simulation:
            wandb.log({"reward": total_reward})


    # Close environment
    env.close()




def training_neurogym(algorithm, noise_std, gamma_avg, hidden_units, hidden_layers, lr, device, log_simulation, seed):

    # Create environment
    duration_stim = 3000
    ts_size = 10
    timing = {'fixation': ('constant', 0),
            'reach': ('constant', duration_stim)}
    kwargs = {'dt': ts_size, 'timing': timing}
    env = gym.make('neurogym:Reaching1D-v0', **kwargs)

    
    # Control randomness for reproducibility
    control_randomness(seed, env)


    # Network params
    n_inputs = env.observation_space.shape[0]
    n_actions = env.action_space.n
    net_structure = [n_inputs, n_actions]
    for _ in range(hidden_layers):
        net_structure.insert(1, hidden_units)
    act_func = leaky_relu(0.1)
    out_act_function = softmax()


    # Instantiate agents (clean or approx + noisy pass) and move to GPU
    agent = ANN(net_structure=net_structure, batch_size=1, act_func=act_func,
                out_act_function=out_act_function, gamma_avg=gamma_avg, seed=42, device=device)
    _ = agent.to(device)
    if algorithm == 'ours':
        agent_noisy = ANN(net_structure=net_structure, batch_size=1, act_func=act_func,
                    out_act_function=out_act_function, gamma_avg=gamma_avg, seed=42, device=device)
        _ = agent_noisy.to(device)


    # Simulation parameters
    n_steps = 5000000 #500000    
    prob_actions = True
    explicit_noise = True
    if algorithm == 'BP':
        optimizer = torch.optim.SGD(agent.parameters(), lr=lr)
    if algorithm == 'RMHL':
        noise = True
    else:
        noise = False


    # Log experiment in wandb
    if log_simulation:
        name = str(algorithm)+'_stim'+str(duration_stim)
        if algorithm != 'random':
            name += '_gamma'+str(gamma_avg)+'_hl'+str(hidden_layers)+'_units'+str(net_structure[1])+'_lr'+str(lr)
        if algorithm != 'BP' and algorithm != 'random':
            name += '_noise'+str(noise_std)+'_gamma_avg'+str(gamma_avg)
        wandb.init(
            project="Reaching1D", name=name,
            config={}
        )
        if algorithm == 'RMHL' and explicit_noise:
            name = 'e'+name
        wandb.log({"reward per trial": 0})
        wandb.log({"instantaenous reward": 0})


    # Initialize stuff
    obs = env.reset()
    n_trials = int(n_steps/(duration_stim/ts_size))
    total_reward_per_trial = [0]
    rewards = []
    all_obs = [obs]
    agent.reset_states()
    if algorithm == 'ours':
        agent_noisy.reset_states()
    reward_avg = None


    # Iterate steps (and trials)
    trial = 0
    for _ in range(n_steps):

        # Actor chooses action based on its policy
        policy, noises = agent.forward(obs, save_states=False, noise=noise, noise_std=noise_std)
        if algorithm == 'ours':
            policy_noisy, noises = agent_noisy.forward(obs, save_states=False, noise=True, noise_std=noise_std)
        if prob_actions:
            action = int(torch.multinomial(policy, 1)) # Probabilistic choice of actions
        else:
            action = int(torch.argmax(policy)) # Always best action
        if algorithm == 'random':
            action = np.random.randint(n_actions) # Random choice of actions
        obs, reward, done, info = env.step(action)
        all_obs.append(obs)


        # Accumulate rewards
        rewards.append(reward)
        total_reward_per_trial[-1] += reward


        # Initialize moving average for the total reward
        if reward_avg is None:
            reward_avg = reward


        ###--------- RPE computation ---------###

        # Compute the RPE
        RPE = reward - reward_avg

        # # Scale the RPE according to the absolute R
        # RPE /= np.abs(reward)


        ###------ LEARNING ------###

        # BP learning
        if algorithm == 'BP':
            loss = -torch.log(policy[action]) * RPE
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # RMHL
        elif algorithm == 'RMHL':
            for indx in range(len(agent.net_structure) - 1):
                pre_out = agent.x[indx]
                if explicit_noise:
                    dW = (reward - reward_avg) * torch.outer(noises[indx], pre_out)
                else:
                    post_act = agent.a[indx+1]
                    post_act_avg = agent.a_avg[indx+1]
                    dW = (reward - reward_avg) * torch.outer(torch.squeeze(post_act-post_act_avg), pre_out)
                agent.fwd_layers[indx].weight.data = agent.fwd_layers[indx].weight.data + lr * dW

        # ours
        elif algorithm == 'ours':
            for indx in range(len(agent.net_structure) - 1):
                pre_out = agent.x[indx]
                norm_noise = noises[indx] / (torch.norm(noises[indx], 'fro')**2)
                dp = torch.log(policy_noisy[action]) - torch.log(policy[action])
                dW = RPE * dp * torch.outer(norm_noise, pre_out)
                agent.fwd_layers[indx].weight.data = agent.fwd_layers[indx].weight.data + lr * dW
                # agent.fwd_layers[indx].weight.data = agent.fwd_layers[indx].weight.data + lr * agent.net_structure[indx+1] * dW
                agent_noisy.fwd_layers[indx].weight.data = copy.deepcopy(agent.fwd_layers[indx].weight.data)


        ###----------------------###


        # Compute reward moving average
        reward_avg += gamma_avg * (reward - reward_avg)


        # Trial is finished
        if info['new_trial']:
            trial += 1
            reward_avg = None
            agent.reset_states()
            if algorithm == 'ours':
                agent_noisy.reset_states()
            print("[Trial "+str(trial)+"/"+str(n_trials)+"] Total reward:", total_reward_per_trial[-1])
            if log_simulation:
                wandb.log({"reward per trial": total_reward_per_trial[-1]})
            total_reward_per_trial.append(0)


        # Save datapoint in weights and biases
        if log_simulation:
            wandb.log({"instantaenous reward": reward})




    # Close the environment
    env.close()
