import networkx as nx, random, numpy as np, torch, os, math, datetime
import torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque
from scipy.signal import savgol_filter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_DELAY = 50
MAX_PACKET_LOSS = 0.005
LOSS_THRESHOLD = 0.0005
class PathGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PathGCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class NetworkEnv:
    def __init__(self, seed=978):
        self.weight_pairs = [(0.9,0.1),(0.7,0.3),(0.5,0.5),(0.3,0.7),(0.1,0.9)]
        self.selected_weights = random.choice(self.weight_pairs)
        self.time_steps=100
        self.max_switches=2
        self.min_switches=1
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            self.seed = seed
        else:
            self.seed = None
        self.network,self.switches,self.comm_nodes = self.generate_network_topology()
        self.node1,self.node2,self.valid_paths = self.select_communication_nodes()
        self.update_edge_metrics()
        self.path_metrics = self.generate_path_metrics(self.valid_paths)
        self.gcn = PathGCN(3,16,8).to(device)
        self.current_step=0
        self.episode_results=[]
    def generate_network_topology(self):
        G=nx.Graph()
        switches=[f'S{i}' for i in range(1,11)]
        comm_nodes=[f'C{i}' for i in range(1,11)]
        G.add_nodes_from(switches,type='switch')
        G.add_nodes_from(comm_nodes,type='communication')
        for i in range(len(switches)):
            for j in range(i+1,len(switches)):
                G.add_edge(switches[i],switches[j],bandwidth=random.uniform(100,10000),delay=random.uniform(1,50),packet_loss=random.uniform(0,0.005))
        for _ in range(random.randint(3,6)):
            s1,s2 = random.sample(switches,2)
            G.add_edge(s1,s2,bandwidth=random.uniform(100,10000),delay=random.uniform(1,50),packet_loss=random.uniform(0,0.005))
        for c in comm_nodes:
            for switch in random.sample(switches,random.randint(1,3)):
                G.add_edge(c,switch,bandwidth=random.uniform(100,10000),delay=random.uniform(1,50),packet_loss=random.uniform(0,0.005))
        return G,switches,comm_nodes
    def select_communication_nodes(self):
        node1,node2 = random.sample(self.comm_nodes,2)
        all_paths = list(nx.all_simple_paths(self.network,source=node1,target=node2))
        valid_paths = [path for path in all_paths if self.min_switches <= sum(1 for node in path if self.network.nodes[node]['type']=='switch') <= self.max_switches]
        if not valid_paths:
            raise ValueError("No valid paths found.")
        return node1,node2,valid_paths
    def generate_path_metrics(self, paths):
        path_metrics={}
        for i,path in enumerate(paths):
            total_delay=0
            total_packet_loss=0
            total_bandwidth_usage=0
            for j in range(len(path)-1):
                edge = self.network[path[j]][path[j+1]]
                total_delay += edge['delay']
                total_packet_loss = 1 - (1 - total_packet_loss)*(1 - edge['packet_loss'])
                total_bandwidth_usage += edge['bandwidth']
            path_metrics[i]={'delay':total_delay,'bandwidth_usage':total_bandwidth_usage,'packet_loss':total_packet_loss}
        return path_metrics
    def update_edge_metrics(self):
        for u,v,data in self.network.edges(data=True):
            data['bandwidth'] = random.uniform(100,10000)
            network_load = random.uniform(0,1)
            base_delay = random.uniform(1,50)
            data['delay'] = base_delay*(1+network_load)
            data['packet_loss'] = min((data['delay']/MAX_DELAY)*MAX_PACKET_LOSS*(1+network_load), MAX_PACKET_LOSS)
    def reset(self):
        self.current_step=0
        self.update_edge_metrics()
        self.path_metrics = self.generate_path_metrics(self.valid_paths)
        self.episode_results=[]
        return self.get_obs()
    def step(self,action):
        self.current_step +=1
        self.update_edge_metrics()
        self.path_metrics = self.generate_path_metrics(self.valid_paths)
        total_reward = self.compute_reward(action)
        done = self.current_step >= self.time_steps
        next_state = self.get_obs()
        delay = self.path_metrics[action]['delay']
        packet_loss = self.path_metrics[action]['packet_loss']
        self.episode_results.append({'step':self.current_step,'action':action,'delay':delay,'bandwidth_usage':self.path_metrics[action]['bandwidth_usage'],'packet_loss':packet_loss,'path':self.valid_paths[action]})
        return next_state,total_reward,done,delay,packet_loss
    def get_obs(self):
        path_features = torch.tensor([[v['delay'],v['bandwidth_usage'],v['packet_loss']] for v in self.path_metrics.values()], dtype=torch.float32).to(device)
        embeddings = self.gcn(path_features).detach().cpu().numpy()
        return embeddings.flatten()
    def compute_reward(self,action,weights=None):
        path_index=action
        delay = self.path_metrics[path_index]['delay']
        packet_loss = self.path_metrics[path_index]['packet_loss']
        alpha,beta = weights if weights else self.selected_weights
        reward = 1 / (np.exp(alpha*(delay/MAX_DELAY)) + np.exp(beta*(packet_loss/MAX_PACKET_LOSS)))
        return max(reward,0)
    def compute_reward_dqn(self,action):
        delay = self.path_metrics[action]['delay']
        packet_loss = self.path_metrics[action]['packet_loss']
        alpha=0.5
        reward = 1 / (np.exp(alpha*(delay/MAX_DELAY)) + np.exp((1 - alpha)*(packet_loss/MAX_PACKET_LOSS)))
        return max(reward,0)
    def compute_cost_aco(self,path_index):
        delay = self.path_metrics[path_index]['delay']
        packet_loss = self.path_metrics[path_index]['packet_loss']
        alpha,beta = self.selected_weights
        return alpha*delay + beta*packet_loss*MAX_DELAY
    def plot_network_topology(self,output_dir):
        plt.figure(figsize=(12,8))
        pos = nx.spring_layout(self.network)
        edge_labels = {(u,v):f"Delay:{d['delay']:.2f},Loss:{d['packet_loss']:.5f}" for u,v,d in self.network.edges(data=True)}
        nx.draw(self.network,pos,with_labels=True,node_color='lightblue',edge_color='gray',node_size=3000,font_size=10)
        nx.draw_networkx_edge_labels(self.network,pos,edge_labels=edge_labels,font_color='red')
        plt.title("Generated Network Topology")
        plt.savefig(os.path.join(output_dir,'001A001.png'))
        plt.close()
class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DuelingQNetwork, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
class D3QNAgent:
    def __init__(self,state_dim,action_dim,hidden_dim,learning_rate=0.0001,buffer_size=20000,batch_size=512):
        self.state_dim = state_dim
        self.action_dim = action_dim if action_dim >0 else 1
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)
        self.q_net = DuelingQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_q_net = DuelingQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.gamma = 0.99
        self.update_counter=0
        self.update_interval=10
        self.epsilon=1.0
        self.epsilon_decay=0.999
        self.epsilon_min=0.01
    def select_action(self,state,env):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_dim -1)
        else:
            with torch.no_grad():
                action = self.q_net(state_tensor)[0].argmax().item()
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return action
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.from_numpy(np.array([e[0] for e in minibatch])).float().to(device)
        actions = torch.from_numpy(np.array([e[1] for e in minibatch])).long().to(device)
        rewards = torch.from_numpy(np.array([e[2] for e in minibatch])).float().to(device)
        next_states = torch.from_numpy(np.array([e[3] for e in minibatch])).float().to(device)
        dones = torch.tensor([e[4] for e in minibatch], dtype=torch.float32).to(device)
        q_values = self.q_net(states)
        expected_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        next_actions = self.q_net(next_states).detach().argmax(1).unsqueeze(1)
        next_q_values_target = self.target_q_net(next_states).gather(1, next_actions).squeeze()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values_target
        loss = self.loss_fn(expected_q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_counter +=1
        if self.update_counter % self.update_interval ==0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, learning_rate=0.0001, buffer_size=20000, batch_size=512):
        self.state_dim = state_dim
        self.action_dim = action_dim if action_dim > 0 else 1
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)
        self.target_q_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.gamma = 0.99
        self.update_counter = 0
        self.update_interval = 10
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
    def select_action(self, state):
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
                action = self.q_net(state_tensor.unsqueeze(0))[0].argmax().item()
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return action
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.from_numpy(np.array([e[0] for e in minibatch])).float().to(device)
        actions = torch.from_numpy(np.array([e[1] for e in minibatch])).long().to(device)
        rewards = torch.from_numpy(np.array([e[2] for e in minibatch])).float().to(device)
        next_states = torch.from_numpy(np.array([e[3] for e in minibatch])).float().to(device)
        dones = torch.tensor([e[4] for e in minibatch], dtype=torch.float32).to(device)
        q_values = self.q_net(states)
        expected_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q_values_target = self.target_q_net(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values_target
        loss = self.loss_fn(expected_q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_counter += 1
        if self.update_counter % self.update_interval == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
class AntColonyOptimizer:
    def __init__(self,env,num_ants,num_iterations,alpha=1,beta=5,rho=0.1,Q=100):
        self.env=env
        self.num_ants=num_ants
        self.num_iterations=num_iterations
        self.alpha=alpha
        self.beta=beta
        self.rho=rho
        self.Q=Q
        self.pheromone = np.ones(len(env.valid_paths))
    def heuristic(self,path_index):
        delay = self.env.path_metrics[path_index]['delay']
        packet_loss = self.env.path_metrics[path_index]['packet_loss']
        alpha,beta = self.env.selected_weights
        return 1.0/(alpha*delay + beta*packet_loss*MAX_DELAY +1e-6)
    def update_pheromone(self,ants):
        self.pheromone *= (1 - self.rho)
        for ant in ants:
            for path in ant['paths']:
                self.pheromone[path] += self.Q/(ant['cost'] +1e-6)
    def run(self):
        best_path=None
        best_cost=float('inf')
        for _ in range(self.num_iterations):
            ants=[]
            for _ in range(self.num_ants):
                heuristics = np.maximum(np.array([self.heuristic(i) for i in range(len(self.pheromone))]),1e-6)
                probabilities = (self.pheromone**self.alpha)*(heuristics**self.beta)
                if probabilities.sum() ==0:
                    probabilities = np.ones_like(probabilities)/len(probabilities)
                else:
                    probabilities /= probabilities.sum()
                selected_path = np.random.choice(len(self.env.valid_paths), p=probabilities)
                path_cost = self.env.compute_cost_aco(selected_path)
                ants.append({'paths':[selected_path],'cost':path_cost})
                if path_cost < best_cost:
                    best_cost=path_cost
                    best_path=self.env.valid_paths[selected_path]
            self.update_pheromone(ants)
        return best_path, best_cost
class OSPFAlgorithm:
    def __init__(self,env):
        self.env=env
    def run(self):
        delays = []
        for path in self.env.valid_paths:
            delay = sum(self.env.network[path[i]][path[i+1]]['delay'] for i in range(len(path)-1))
            delays.append(delay)
        if delays:
            min_delay = min(delays)
            min_index = delays.index(min_delay)
            shortest_path = self.env.valid_paths[min_index]
            total_delay = delays[min_index]
            total_packet_loss = 1 - np.prod([1 - self.env.network[shortest_path[i]][shortest_path[i+1]]['packet_loss'] for i in range(len(shortest_path)-1)])
            total_bandwidth_usage = sum(self.env.network[shortest_path[i]][shortest_path[i+1]]['bandwidth'] for i in range(len(shortest_path)-1))
            return shortest_path, total_delay, total_packet_loss, total_bandwidth_usage
        else:
            return [], float('inf'), float('inf'), float('inf')
def plot_bar_chart(data,title,ylabel,num,output_dir):
    labels = [f'Test {i+1}' for i in range(len(data['D3QN']))]
    x = np.arange(len(labels))
    width = 0.2
    fig,ax = plt.subplots(figsize=(10,6))
    ax.bar(x -1.5*width, data['D3QN'], width, label='D3QN', color='#f4a582', edgecolor='black')
    ax.bar(x -0.5*width, data['DQN'], width, label='DQN', color='#92c5de', edgecolor='black')
    ax.bar(x +0.5*width, data['Ant Colony'], width, label='Ant Colony', color='#b2abd2', edgecolor='black')
    ax.bar(x +1.5*width, data['OSPF'], width, label='OSPF', color='#ca0020', edgecolor='black')
    ax.set_xlabel('Test')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir,f'A{num:03d}.png'))
    plt.close()
    with open(os.path.join(output_dir,f'data_{num:03d}.txt'),'w') as file:
        file.write("Test\tD3QN\tDQN\tAnt Colony\tOSPF\n")
        for i,label in enumerate(labels):
            file.write(f"{label}\t{data['D3QN'][i]}\t{data['DQN'][i]}\t{data['Ant Colony'][i]}\t{data['OSPF'][i]}\n")
if __name__ == "__main__":
    seeds = [9177]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for seed in seeds:
        output_dir = f"{timestamp}_seed{seed}"
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir,'seed.txt'),'w') as f:
            f.write(str(seed))
        env = NetworkEnv(seed=seed)
        env.plot_network_topology(output_dir)
        num_episodes = 6000
        num_tests = 6
        rewards_per_weight_pair = {}
        delays_per_weight_pair = {}
        fig_weight_rewards, ax_weight_rewards = plt.subplots(figsize=(10,6))
        for wp in env.weight_pairs:
            env.selected_weights = wp
            agent_d3qn = D3QNAgent(state_dim=env.get_obs().shape[0], action_dim=len(env.valid_paths), hidden_dim=64, batch_size=256)
            rewards, delays = [], []
            for ep in range(num_episodes):
                state = env.reset()
                tr, td = 0,0
                for t in range(env.time_steps):
                    action = agent_d3qn.select_action(state, env)
                    next_state, reward, done, delay, _ = env.step(action)
                    agent_d3qn.remember(state, action, reward, next_state, done)
                    agent_d3qn.train()
                    state, tr, td = next_state, tr + reward, td + delay
                    if done: break
                rewards.append(tr)
                delays.append(td/env.time_steps)
            smoothed = savgol_filter(rewards,11,3)
            rewards_per_weight_pair[wp]=smoothed
            delays_per_weight_pair[wp]=delays
            ax_weight_rewards.plot(range(1,num_episodes+1), smoothed, label=f"Weights {wp}")
            np.savetxt(os.path.join(output_dir,f"rewards_weights_{wp[0]}_{wp[1]}.txt"), smoothed)
        ax_weight_rewards.set_title('D3QN Total Reward under Different Weight Pairs')
        ax_weight_rewards.set_xlabel('Episode')
        ax_weight_rewards.set_ylabel('Total Reward')
        ax_weight_rewards.legend()
        fig_weight_rewards.tight_layout()
        plt.savefig(os.path.join(output_dir,'A005.png'))
        plt.close()
        avg_rewards = {wp:np.mean(r) for wp,r in rewards_per_weight_pair.items()}
        optimal_wp = max(avg_rewards.items(), key=lambda x:x[1])[0]
        env.selected_weights = optimal_wp
        learning_rates = [0.1,0.01,0.001,0.0001]
        d3qn_lr_rewards = {lr:[] for lr in learning_rates}
        d3qn_lr_delays = {lr:[] for lr in learning_rates}
        fig_rewards, ax_rewards = plt.subplots(figsize=(10,6))
        fig_delays, ax_delays = plt.subplots(figsize=(10,6))
        for lr in learning_rates:
            agent_d3qn = D3QNAgent(state_dim=env.get_obs().shape[0], action_dim=len(env.valid_paths), hidden_dim=64, learning_rate=lr, batch_size=256)
            rewards, delays = [], []
            for ep in range(num_episodes):
                state = env.reset()
                tr, td =0,0
                for t in range(env.time_steps):
                    action = agent_d3qn.select_action(state, env)
                    next_state, reward, done, delay, _ = env.step(action)
                    agent_d3qn.remember(state, action, reward, next_state, done)
                    agent_d3qn.train()
                    state, tr, td = next_state, tr + reward, td + delay
                    if done: break
                rewards.append(tr)
                delays.append(td/env.time_steps)
            smoothed_r = savgol_filter(rewards,11,3)
            smoothed_d = savgol_filter(delays,11,3)
            d3qn_lr_rewards[lr]=smoothed_r
            d3qn_lr_delays[lr]=smoothed_d
            ax_rewards.plot(range(1,num_episodes+1), smoothed_r, label=f'LR={lr}')
            ax_delays.plot(range(1,num_episodes+1), smoothed_d, label=f'LR={lr}')
            np.savetxt(os.path.join(output_dir,f"d3qn_rewards_lr_{lr}_1.txt"), smoothed_r)
            np.savetxt(os.path.join(output_dir,f"d3qn_delays_lr_{lr}_1.txt"), smoothed_d)
        ax_rewards.set_title('D3QN Total Reward under Different Learning Rates')
        ax_rewards.set_xlabel('Episode')
        ax_rewards.set_ylabel('Total Reward')
        ax_rewards.legend()
        fig_rewards.tight_layout()
        plt.savefig(os.path.join(output_dir,'A006.png'))
        plt.close()
        ax_delays.set_title('D3QN Average Delay under Different Learning Rates')
        ax_delays.set_xlabel('Episode')
        ax_delays.set_ylabel('Average Delay (ms)')
        ax_delays.legend()
        fig_delays.tight_layout()
        plt.savefig(os.path.join(output_dir,'A007.png'))
        plt.close()
        results = {'delay':{'D3QN':[],'DQN':[],'Ant Colony':[],'OSPF':[]}, 'packet_loss':{'D3QN':[],'DQN':[],'Ant Colony':[],'OSPF':[]}, 'u_cv':{'D3QN':[],'DQN':[],'Ant Colony':[],'OSPF':[]}}
        num_cols = 2
        num_rows = (num_tests + num_cols -1) // num_cols
        if num_tests == 1:
            fig, ax = plt.subplots(figsize=(10,6))
        else:
            fig, axs = plt.subplots(num_rows,num_cols, figsize=(15,6*num_rows))
            axs = np.array(axs).flatten()
        for test in range(num_tests):
            agent_d3qn = D3QNAgent(state_dim=env.get_obs().shape[0], action_dim=len(env.valid_paths), hidden_dim=64, batch_size=256, learning_rate=0.0001)
            d3qn_rewards, d3qn_delays, d3qn_packet_losses, d3qn_u_cvs = [], [], [], []
            for ep in range(num_episodes):
                state = env.reset()
                tr, td, tpl =0,0,0
                for t in range(env.time_steps):
                    action = agent_d3qn.select_action(state, env)
                    next_state, reward, done, delay, packet_loss = env.step(action)
                    agent_d3qn.remember(state, action, reward, next_state, done)
                    agent_d3qn.train()
                    state, tr, td, tpl = next_state, tr + reward, td + delay, tpl + packet_loss
                    if done: break
                U_used = [res['bandwidth_usage'] for res in env.episode_results]
                mu = np.mean(U_used)
                sigma = np.sqrt(np.mean((U_used - mu) ** 2))
                U_cv = sigma / mu if mu != 0 else 0
                d3qn_u_cvs.append(U_cv)
                d3qn_rewards.append(tr)
                d3qn_delays.append(td/env.time_steps)
                d3qn_packet_losses.append(tpl/env.time_steps)
            smoothed_d3qn = savgol_filter(d3qn_rewards,11,3)
            dqn_rewards, dqn_delays, dqn_packet_losses, dqn_u_cvs = [], [], [], []
            agent_dqn = DQNAgent(state_dim=3*len(env.valid_paths), action_dim=len(env.valid_paths), hidden_dim=64, batch_size=256, learning_rate=0.0001)
            for ep in range(num_episodes):
                state = env.reset()
                tr, td, tpl =0,0,0
                for t in range(env.time_steps):
                    state_obs = [env.path_metrics[i]['delay'] for i in range(len(env.valid_paths))] + [env.path_metrics[i]['bandwidth_usage'] for i in range(len(env.valid_paths))] + [env.path_metrics[i]['packet_loss'] for i in range(len(env.valid_paths))]
                    action = agent_dqn.select_action(state_obs)
                    next_state, reward, done, delay, packet_loss = env.step(action)
                    reward_total = env.compute_reward_dqn(action)
                    next_state_obs = [env.path_metrics[i]['delay'] for i in range(len(env.valid_paths))] + [env.path_metrics[i]['bandwidth_usage'] for i in range(len(env.valid_paths))] + [env.path_metrics[i]['packet_loss'] for i in range(len(env.valid_paths))]
                    agent_dqn.remember(state_obs, action, reward_total, next_state_obs, done)
                    agent_dqn.train()
                    state = next_state
                    tr += reward_total
                    td += delay
                    tpl += packet_loss
                    if done: break
                U_used = [res['bandwidth_usage'] for res in env.episode_results]
                mu = np.mean(U_used)
                sigma = np.sqrt(np.mean((U_used - mu) ** 2))
                U_cv = sigma / mu if mu != 0 else 0
                dqn_u_cvs.append(U_cv)
                dqn_rewards.append(tr)
                dqn_delays.append(td/env.time_steps)
                dqn_packet_losses.append(tpl/env.time_steps)
            smoothed_dqn = savgol_filter(dqn_rewards,11,3)
            ant_colony_rewards, ant_colony_delays, ant_colony_packet_losses, ant_colony_u_cvs = [], [], [], []
            for ep in range(num_episodes):
                state = env.reset()
                tr, td, tpl =0,0,0
                for t in range(env.time_steps):
                    ant_colony = AntColonyOptimizer(env,20,20,1,5,0.1,100)
                    best_path, _ = ant_colony.run()
                    best_path_index = env.valid_paths.index(best_path) if best_path in env.valid_paths else 0
                    next_state, reward, done, delay, packet_loss = env.step(best_path_index)
                    tr += reward
                    td += delay
                    tpl += packet_loss
                    if done: break
                U_used = [res['bandwidth_usage'] for res in env.episode_results]
                mu = np.mean(U_used)
                sigma = np.sqrt(np.mean((U_used - mu) ** 2))
                U_cv = sigma / mu if mu != 0 else 0
                ant_colony_u_cvs.append(U_cv)
                ant_colony_rewards.append(tr)
                ant_colony_delays.append(td/env.time_steps)
                ant_colony_packet_losses.append(tpl/env.time_steps)
            smoothed_ant_colony = savgol_filter(ant_colony_rewards,11,3)
            ospf_rewards, ospf_delays_list, ospf_packet_losses_list, ospf_u_cvs = [], [], [], []
            for ep in range(num_episodes):
                state = env.reset()
                ospf_agent = OSPFAlgorithm(env)
                ospf_path, ospf_delay, ospf_loss, _ = ospf_agent.run()
                if ospf_path:
                    action = env.valid_paths.index(ospf_path)
                    tr, td, tpl = 0, 0, 0
                    for t in range(env.time_steps):
                        next_state, reward, done, delay, packet_loss = env.step(action)
                        tr += reward
                        td += delay
                        tpl += packet_loss
                        if done:
                            break
                    U_used = [res['bandwidth_usage'] for res in env.episode_results]
                    mu = np.mean(U_used)
                    sigma = np.sqrt(np.mean((U_used - mu) ** 2))
                    U_cv = sigma / mu if mu != 0 else 0
                    ospf_u_cvs.append(U_cv)
                    ospf_rewards.append(tr)
                    ospf_delays_list.append(td / env.time_steps)
                    ospf_packet_losses_list.append(tpl / env.time_steps)
                else:
                    ospf_rewards.append(0)
                    ospf_delays_list.append(float('inf'))
                    ospf_packet_losses_list.append(float('inf'))
                    ospf_u_cvs.append(float('inf'))
            if num_tests == 1:
                ax.plot(range(1,num_episodes+1), smoothed_d3qn, label='D3QN Total Reward')
                ax.plot(range(1,num_episodes+1), smoothed_dqn, label='DQN Total Reward')
                #ax.plot(range(1,num_episodes+1), smoothed_ant_colony, label='Ant Colony Total Reward')
                ax.set_title(f'Test {test +1} - Reward Curve')
                ax.set_xlabel('Episode')
                ax.set_ylabel('Total Reward')
                ax.legend()
            else:
                axs[test].plot(range(1,num_episodes+1), smoothed_d3qn, label='D3QN Total Reward')
                axs[test].plot(range(1,num_episodes+1), smoothed_dqn, label='DQN Total Reward')
                #axs[test].plot(range(1,num_episodes+1), smoothed_ant_colony, label='Ant Colony Total Reward')
                axs[test].set_title(f'Test {test +1} - Reward Curve')
                axs[test].set_xlabel('Episode')
                axs[test].set_ylabel('Total Reward')
                axs[test].legend()
            np.savetxt(os.path.join(output_dir,f'test_{test+1}_d3qn_rewards.txt'), smoothed_d3qn)
            np.savetxt(os.path.join(output_dir,f'test_{test+1}_dqn_rewards.txt'), smoothed_dqn)
            #np.savetxt(os.path.join(output_dir,f'test_{test+1}_ant_colony_rewards.txt'), smoothed_ant_colony)
            results['delay']['D3QN'].append(np.mean(d3qn_delays))
            results['packet_loss']['D3QN'].append(np.mean(d3qn_packet_losses))
            results['u_cv']['D3QN'].append(np.mean(d3qn_u_cvs))
            results['delay']['DQN'].append(np.mean(dqn_delays))
            results['packet_loss']['DQN'].append(np.mean(dqn_packet_losses))
            results['u_cv']['DQN'].append(np.mean(dqn_u_cvs))
            results['delay']['Ant Colony'].append(np.mean(ant_colony_delays))
            results['packet_loss']['Ant Colony'].append(np.mean(ant_colony_packet_losses))
            results['u_cv']['Ant Colony'].append(np.mean(ant_colony_u_cvs))
            results['delay']['OSPF'].append(np.mean(ospf_delays_list))
            results['packet_loss']['OSPF'].append(np.mean(ospf_packet_losses_list))
            results['u_cv']['OSPF'].append(np.mean(ospf_u_cvs))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,'A008.png'))
        plt.close()
        plot_bar_chart(results['delay'], 'Average Delay per Test', 'Delay (ms)',2,output_dir)
        plot_bar_chart(results['packet_loss'], 'Average Packet Loss per Test', 'Packet Loss',3,output_dir)
        plot_bar_chart(results['u_cv'], 'Average Load Coefficient of Variation per Test', 'U_cv',4,output_dir)
