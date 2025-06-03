from optimus_api import TestingService
from munch import DefaultMunch
import yaml
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

available_passes = {
    'UnrollingOuter': ' -ftransforms=unrolling_outer ',
    'BlockSimplify': ' -ftransforms=block_simplify ',
    'Fusion': ' -ftransforms=fusion ',
    'CombineAssignments': ' -ftransforms=combine_assignments ',
    'UnrollingInner': ' -ftransforms=unrolling_inner ',
}

DANGEROUS_COMBINATIONS = [
    ['UnrollingOuter', 'Fusion'],
]

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.embedding = nn.Embedding(len(available_passes) + 1, 16)
        self.lstm = nn.LSTM(16, 64, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(64 + 1, 64)
        self.fc2 = nn.Linear(64, output_size)
    
    def forward(self, state, length):
        x = self.embedding(state)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = torch.cat([x, length], dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DQNAgent:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(input_size, output_size).to(self.device)
        self.target_model = DQN(input_size, output_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, valid_actions):
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        
        state_tensor = torch.tensor([state], dtype=torch.long).to(self.device)
        length_tensor = torch.tensor([[len(state) / 10.0]], dtype=torch.float).to(self.device)
        
        with torch.no_grad():
            q_values = self.model(state_tensor, length_tensor).cpu().numpy()[0]
        
        valid_q_values = [q_values[i] for i in valid_actions]
        return valid_actions[np.argmax(valid_q_values)]
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        state_tensors = [torch.tensor(s, dtype=torch.long) for s in states]
        state_lengths = torch.tensor([[len(s) / 10.0] for s in states], dtype=torch.float)
        state_tensors = nn.utils.rnn.pad_sequence(
            state_tensors, batch_first=True, padding_value=0
        ).to(self.device)
        state_lengths = state_lengths.to(self.device)
        
        action_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        reward_tensor = torch.tensor(rewards, dtype=torch.float).to(self.device)
        done_tensor = torch.tensor(dones, dtype=torch.float).to(self.device)
        
        next_state_tensors = [torch.tensor(s, dtype=torch.long) for s in next_states if s is not None]
        next_state_lengths = torch.tensor(
            [[len(s) / 10.0] for s in next_states if s is not None], 
            dtype=torch.float
        )
        
        current_q = self.model(state_tensors, state_lengths)
        current_q = current_q.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        
        next_q = torch.zeros(self.batch_size, device=self.device)
        if next_state_tensors:
            next_state_tensors = nn.utils.rnn.pad_sequence(
                next_state_tensors, batch_first=True, padding_value=0
            ).to(self.device)
            next_state_lengths = next_state_lengths.to(self.device)
            
            with torch.no_grad():
                next_q_values = self.target_model(next_state_tensors, next_state_lengths)
                next_q_valid = next_q_values.max(1)[0]
            
            mask = torch.tensor([s is not None for s in next_states], dtype=torch.bool)
            next_q[mask] = next_q_valid
        
        target_q = reward_tensor + self.gamma * next_q * (1 - done_tensor)
        
        loss = self.loss_fn(current_q, target_q.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class SearchingOptimizer:
    def __init__(self):
        self.loadConfig()
        self.service = TestingService(
            self.config.url,
            self.config.username,
            self.config.password,
            self.config.settings
        )
        self.agent = DQNAgent(
            input_size=len(available_passes) + 1,
            output_size=len(available_passes))
        self.text = None
        self.history = []
        self.best_runtime = float('inf')
        self.best_sequence = []

    def loadConfig(self):
        with open("config.default.yml", "r") as stream:
            try:
                obj = yaml.safe_load(stream)
                self.config = DefaultMunch.fromDict(obj)
            except yaml.YAMLError as exc:
                print(exc)

    def getExtraArgs(self, passes_list):
        args = [available_passes[p] for p in passes_list]
        return " ".join(args)

    def state_representation(self, sequence):
        """Convert sequence of passes to numerical state representation"""
        pass_to_idx = {pass_name: idx+1 for idx, pass_name in enumerate(available_passes.keys())}
        state = [0] * 10
        
        for i, p in enumerate(sequence):
            if i < 10:
                state[i] = pass_to_idx.get(p, 0)
        
        return state

    def get_valid_actions(self, sequence):
        """Get valid next passes (avoid consecutive duplicates)"""
        last_pass = sequence[-1] if sequence else None
        valid_indices = []
        
        for idx, pass_name in enumerate(available_passes.keys()):
            if pass_name != last_pass:
                valid_indices.append(idx)
                
        return valid_indices

    def contains_dangerous_combination(self, sequence):
        """Check if the sequence contains any dangerous combination"""
        seq_str = ",".join(sequence)
        for comb in DANGEROUS_COMBINATIONS:
            comb_str = ",".join(comb)
            if comb_str in seq_str:
                return True
        return False

    def optimize(self, text, episodes=100):
        self.text = text
        
        for episode in range(episodes):
            sequence = []
            state = self.state_representation(sequence)
            done = False
            total_reward = 0
            step = 0
            
            while not done and step < 7:
                valid_actions = self.get_valid_actions(sequence)
                
                if not valid_actions:
                    break
                
                action_idx = self.agent.act(state, valid_actions)
                action = list(available_passes.keys())[action_idx]
                new_sequence = sequence + [action]
                
                if self.contains_dangerous_combination(new_sequence):
                    reward = -5
                    done = True
                    next_state = None
                    
                    result = {
                        "sequence": new_sequence.copy(),
                        "runtime": -1,
                        "ret_code": -1,
                        "reason": "dangerous_combination"
                    }
                    self.history.append(result)
                    print(result)
                else:
                    extraCmdArg = self.getExtraArgs(new_sequence)
                    
                    task_hash = self.service.sendOPSTask(extraCmdArg, self.text)
                    self.service.waitTaskFinish(task_hash)
                    runtime, ret_code = self.service.getTaskRunTime(task_hash)
                    
                    result = {
                        "sequence": new_sequence.copy(),
                        "runtime": runtime,
                        "ret_code": ret_code
                    }
                    self.history.append(result)
                    
                    print(result)

                    if ret_code != 2:
                        reward = -10
                        done = True
                        next_state = None
                    else:
                        if self.best_runtime == float('inf'):
                            reward = 1.0 / runtime
                        else:
                            improvement = self.best_runtime - runtime
                            reward = max(improvement, 0) * 10 + 1.0 / runtime
                        
                        if runtime < self.best_runtime:
                            self.best_runtime = runtime
                            self.best_sequence = new_sequence.copy()
                        
                        next_state = self.state_representation(new_sequence)
                        done = False
                
                self.agent.remember(state, action_idx, reward, next_state, done)
                
                self.agent.replay()
                
                if not done:
                    state = next_state
                    sequence = new_sequence
                    step += 1
                    total_reward += reward
                else:
                    break
            
            if episode % 10 == 0:
                self.agent.update_target_model()
            
            print(f"Episode: {episode+1}/{episodes}, Reward: {total_reward:.4f}, Best Runtime: {self.best_runtime:.6f}")
        
        print("\nOptimization complete!")
        print(f"Best sequence: {self.best_sequence}")
        print(f"Best runtime: {self.best_runtime:.6f}")
        

if __name__ == "__main__":
    searcher = SearchingOptimizer()
    with open('simple_convolution.c') as input_program:
        text = input_program.read()
        searcher.optimize(text, episodes=22)