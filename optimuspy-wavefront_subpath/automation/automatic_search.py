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
import hashlib
import json
import math

available_passes = {
    'UnrollingOuter': ' -ftransforms=unrolling_outer ',
    'BlockSimplify': ' -ftransforms=block_simplify ',
    'Fusion': ' -ftransforms=fusion ',
    'CombineAssignments': ' -ftransforms=combine_assignments ',
    'UnrollingInner': ' -ftransforms=unrolling_inner ',
    'LoopInvariant': ' -ftransforms=loop_invariant ',
    'ConstantPropagation': ' -ftransforms=const_propagation ',
    'UnusedDeclarations': ' -ftransforms=unused_declarations ',
    'FullUnrolling': ' -ftransforms=full_unrolling ',
    'LoopUnrolling': ' -ftransforms=unrolling ',
    'ArithmeticExpansion': ' -ftransforms=arithmetic_exp ',
    'ExpressionSimplifier': ' -ftransforms=expr_simplifier ',
    'FullInlining': ' -ftransforms=full_inlining ',
    'LoopNesting': ' -ftransforms=nesting '

}

DANGEROUS_COMBINATIONS = []

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.embedding = nn.Embedding(len(available_passes) + 1, 16)
        self.lstm = nn.LSTM(16, 64, num_layers=2, batch_first=True)
        self.attention = nn.MultiheadAttention(64, 4, batch_first=True)
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
    def __init__(self, input_size, output_size, model_path = None):
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
        self.model_path = model_path or "dqn_model.pth"
        if os.path.exists(self.model_path):
            self.load_model()

    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, self.model_path)

    def load_model(self):
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            print(f"Loaded model from {self.model_path}")
    
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
            output_size=len(available_passes),
            model_path="dqn_model.pth"
        )
        self.text = None
        self.history = []
        self.best_runtime = float('inf')
        self.best_sequence = []
        self.history_csv = "optimization_history.csv"
        self.best_unit = ""
        self.base_runtime = float('inf')
        self.bad_sequences_path = "bad_sequences.json"
        self.bad_sequences = self.load_bad_sequences()
        self.failed_outputs_dir = "failed_outputs"
        os.makedirs(self.failed_outputs_dir, exist_ok=True)

    def load_bad_sequences(self):
        if os.path.exists(self.bad_sequences_path):
            with open(self.bad_sequences_path, 'r') as f:
                return json.load(f)
        return {}
    
    def save_bad_sequences(self):
        with open(self.bad_sequences_path, 'w') as f:
            json.dump(self.bad_sequences, f)

    def save_failed_output(self, sequence, output, reason):
        filename = f"{hashlib.md5(self.text.encode()).hexdigest()}_{'_'.join(sequence)}.c"
        path = os.path.join(self.failed_outputs_dir, filename)
        with open(path, 'w') as f:
            f.write(f"// Failed reason: {reason}\n")
            f.write(f"// Sequence: {sequence}\n\n")
            f.write(output)
        return path

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
    
    def get_base_runtime(self, text):
        extraCmdArg = ""
        task_hash = self.service.sendOPSTask(extraCmdArg, text)
        if not self.service.waitTaskFinish(task_hash):
            return -1, -2, "timeout"
        runtime, ret_code, unit = self.service.getTaskRunTime(task_hash)
        return runtime, ret_code, unit
    
    def save_history_to_csv(self):
        import csv
        with open(self.history_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'Episode', 'Step', 'Sequence', 
                'Runtime', 'Return Code', 'Reason'
            ])
            for record in self.history:
                writer.writerow([
                    record.get('episode', ''),
                    record.get('step', ''),
                    ','.join(record['sequence']),
                    record['runtime'],
                    record.get('ret_code', ''),
                    record.get('reason', '')
                ])

    def state_representation(self, sequence):
        pass_to_idx = {pass_name: idx+1 for idx, pass_name in enumerate(available_passes.keys())}
        state = [0] * 10
        
        for i, p in enumerate(sequence):
            if i < 10:
                state[i] = pass_to_idx.get(p, 0)
        
        return state

    def get_valid_actions(self, sequence):
        last_pass = sequence[-1] if sequence else None
        valid_indices = []
        text_hash = hashlib.md5(self.text.encode()).hexdigest()
        for idx, pass_name in enumerate(available_passes.keys()):
            if pass_name != last_pass:
                new_seq = sequence + [pass_name]
                if text_hash in self.bad_sequences:
                    if new_seq in self.bad_sequences[text_hash]:
                        continue
                valid_indices.append(idx)
                
        return valid_indices

    def contains_dangerous_combination(self, sequence):
        seq_str = ",".join(sequence)
        for comb in DANGEROUS_COMBINATIONS:
            comb_str = ",".join(comb)
            if comb_str in seq_str:
                return True
        return False

    def optimize(self, text, episodes=100):
        self.text = text

        self.base_runtime, base_ret_code, base_unit = self.get_base_runtime(text)
        if base_ret_code != 2:
            print(f"Base run failed! Return code: {base_ret_code}")
            return
        
        print(f"Base runtime: {self.base_runtime} {base_unit}")
        
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
                        "episode": episode,
                        "step": step,
                        "sequence": new_sequence.copy(),
                        "runtime": -1,
                        "unit": "None",
                        "ret_code": -1,
                        "reason": "dangerous_combination"
                    }
                    self.history.append(result)
                    print(result)
                else:
                    extraCmdArg = self.getExtraArgs(new_sequence)
                    
                    task_hash = self.service.sendOPSTask(extraCmdArg, self.text)
                    if not self.service.waitTaskFinish(task_hash):
                        result = {
                            "episode": episode,
                            "step": step,
                            "sequence": new_sequence.copy(),
                            "runtime": -1,
                            "unit": "None",
                            "ret_code": -2,
                            "reason": "timeout"
                        }
                        self.history.append(result)
                        print(result)
                        reward = -10
                        done = True
                        next_state = None
                    else:
                        runtime, ret_code, unit = self.service.getTaskRunTime(task_hash)
                    
                    result = {
                        "episode": episode,
                        "step": step,
                        "sequence": new_sequence.copy(),
                        "runtime": runtime,
                        "unit": unit,
                        "ret_code": ret_code
                    }
                    self.history.append(result)
                    
                    print(result)

                    if ret_code != 2 or runtime is None:
                        text_hash = hashlib.md5(text.encode()).hexdigest()
                        if text_hash not in self.bad_sequences:
                            self.bad_sequences[text_hash] = []
                        if new_sequence not in self.bad_sequences[text_hash]:
                            self.bad_sequences[text_hash].append(new_sequence)
                        self.save_bad_sequences()
                        reward = -10
                        done = True
                        next_state = None
                    else:
                        if self.best_runtime == float('inf'):
                            reward = 1.0 / runtime
                        else:
                            improvement = self.base_runtime - runtime
                            reward = max(improvement, 0) * 10
                            sequence_length_bonus = len(new_sequence) * 0.5
                            unique_passes = len(set(new_sequence))
                            diversity_bonus = unique_passes * 0.3
                            repeat_penalty = 0
                            if len(new_sequence) > 1 and new_sequence[-1] == new_sequence[-2]:
                                repeat_penalty = -1
                            total_reward = reward + sequence_length_bonus + diversity_bonus + repeat_penalty
                        
                        if runtime < self.best_runtime:
                            self.best_runtime = runtime
                            self.best_unit = unit
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
                self.agent.save_model()
            
            print(f"Episode: {episode+1}/{episodes}, Reward: {total_reward:.4f}, Best Runtime: {self.best_runtime:.6f} {self.best_unit}")
        
        print("\nOptimization complete!")
        print(f"Base runtime: {self.base_runtime:.6f} {base_unit}")
        print(f"Best sequence: {self.best_sequence}")
        print(f"Best runtime: {self.best_runtime:.6f} {self.best_unit}")
        print(f"Faster then original for {(1 - (self.best_runtime / self.base_runtime)) * 100} percents")
        self.save_history_to_csv()
        print(f"\nOptimization history saved to {self.history_csv}")
        print(f"\nModel saved successfully")
        self.agent.save_model()


if __name__ == "__main__":
    searcher = SearchingOptimizer()
    with open('HUGE_FILE.c') as input_program:
        text = input_program.read()
        searcher.optimize(text, episodes=22)
