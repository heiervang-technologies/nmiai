import json
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Ensure tasks/astar-island is in path for imports
sys.path.append(os.path.dirname(__file__))
import benchmark_experiments

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Mapping for 8 distinct internal classes
# 0=Empty, 1=Settle, 2=Port, 3=Ruin, 4=Forest, 5=Mount, 10=Ocean, 11=Plains
INTERNAL_MAP = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 10:6, 11:7}
INV_INTERNAL_MAP = {v: k for k, v in INTERNAL_MAP.items()}

def map_to_internal(grid):
    return np.vectorize(lambda x: INTERNAL_MAP.get(x, 0))(grid)

def map_to_output(grid_tensor):
    # Output classes: 0=Empty(0,6,7), 1=Settle, 2=Port, 3=Ruin, 4=Forest, 5=Mount
    out = torch.zeros_like(grid_tensor)
    out[grid_tensor == 0] = 0 # Empty
    out[grid_tensor == 1] = 1 # Settle
    out[grid_tensor == 2] = 2 # Port
    out[grid_tensor == 3] = 3 # Ruin
    out[grid_tensor == 4] = 4 # Forest
    out[grid_tensor == 5] = 5 # Mount
    out[grid_tensor == 6] = 0 # Ocean -> Empty
    out[grid_tensor == 7] = 0 # Plains -> Empty
    return out

class AutomatonDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 8 (one-hot state) + 1 (normalized step 0-1) + 4 (one-hot phase) = 13 channels
        self.stem = nn.Conv2d(13, 64, kernel_size=7, padding=3) # 7x7 captures radius 3 conflict!
        self.norm1 = nn.GroupNorm(8, 64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, 128)
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.norm3 = nn.GroupNorm(8, 128)
        
        self.head = nn.Conv2d(128, 8, kernel_size=1)
        
    def forward(self, state_one_hot, step_norm, phase_one_hot):
        # state_one_hot: [B, 8, H, W]
        B, _, H, W = state_one_hot.shape
        
        step_map = step_norm.view(B, 1, 1, 1).expand(B, 1, H, W)
        phase_map = phase_one_hot.view(B, 4, 1, 1).expand(B, 4, H, W)
        
        x = torch.cat([state_one_hot, step_map, phase_map], dim=1)
        
        x = F.gelu(self.norm1(self.stem(x)))
        x = F.gelu(self.norm2(self.conv2(x)))
        x = F.gelu(self.norm3(self.conv3(x)))
        logits = self.head(x)
        return logits

def get_training_data():
    files = glob.glob('tasks/astar-island/replays/*.json')
    files = [f for f in files if 'dense_training' not in f and 'simseed' not in f]
    
    # Take 20 files to keep memory usage reasonable for fast iteration
    files = files[:20]
    
    X_states = []
    X_steps = []
    X_phases = []
    Y_next = []
    
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        frames = data['frames']
        if len(frames) < 51: continue
        
        for i in range(len(frames) - 1):
            prev = map_to_internal(np.array(frames[i]['grid']))
            curr = map_to_internal(np.array(frames[i+1]['grid']))
            
            # We only want to train on cells that *could* change or *did* change to balance the dataset
            # But the CNN needs the full spatial grid. We will train on the full grid.
            X_states.append(prev)
            X_steps.append(i / 50.0)
            X_phases.append(i % 4)
            Y_next.append(curr)
            
    # Convert to tensors
    states = torch.tensor(np.array(X_states), dtype=torch.long)
    steps = torch.tensor(np.array(X_steps), dtype=torch.float32)
    phases = torch.tensor(np.array(X_phases), dtype=torch.long)
    targets = torch.tensor(np.array(Y_next), dtype=torch.long)
    
    return states, steps, phases, targets

def calc_wkl(pred, target, ig):
    dynamic_mask = (ig != 10) & (ig != 5)
    p = np.clip(target, 1e-8, 1.0)
    q = np.clip(pred, 1e-8, 1.0)
    kl = (p * (np.log(p) - np.log(q))).sum(axis=-1)
    entropy_bits = -(p * np.log2(p)).sum(axis=-1)
    weights = (0.15 + entropy_bits) * dynamic_mask
    return np.sum(weights * kl) / np.sum(weights)

def train_and_eval():
    print("Loading data...")
    states, steps, phases, targets = get_training_data()
    print(f"Loaded {len(states)} frame transitions for training.")
    
    dataset_size = len(states)
    batch_size = 32
    epochs = 40
    
    model = AutomatonDenoiser().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Class weights to handle the massive imbalance (most cells stay empty/plains)
    # 0=Empty, 1=Settle, 2=Port, 3=Ruin, 4=Forest, 5=Mount, 6=Ocean, 7=Plains
    class_weights = torch.tensor([0.1, 5.0, 10.0, 10.0, 1.0, 0.01, 0.01, 0.1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    print("Training GPU Simulator...")
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(dataset_size)
        total_loss = 0
        
        for i in range(0, dataset_size, batch_size):
            idx = perm[i:i+batch_size]
            
            b_states = states[idx].to(device)
            b_steps = steps[idx].to(device)
            b_phases = phases[idx].to(device)
            b_targets = targets[idx].to(device)
            
            state_one_hot = F.one_hot(b_states, num_classes=8).float().permute(0, 3, 1, 2)
            phase_one_hot = F.one_hot(b_phases, num_classes=4).float()
            
            logits = model(state_one_hot, b_steps, phase_one_hot)
            
            loss = criterion(logits, b_targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss / (dataset_size/batch_size):.4f}")
            
    print("\nEvaluating on R15-R17 Ground Truth...")
    model.eval()
    
    val_files = glob.glob('tasks/astar-island/ground_truth/*.json')
    total_wkl = 0
    count = 0
    
    for f in val_files:
        basename = os.path.basename(f)
        rn = int(basename.split('_')[0].replace('round', ''))
        sn = int(basename.split('_')[1].replace('seed', '').replace('.json', ''))
        
        if rn < 15 or rn > 17:
            continue
            
        with open(f) as fh:
            d = json.load(fh)
            
        ig = np.array(d['initial_grid'], dtype=np.int32)
        gt = np.array(d['ground_truth'], dtype=np.float32)
        
        # Initialize 200 parallel universes!
        NUM_SIMS = 200
        ig_mapped = map_to_internal(ig)
        current_state = torch.tensor(ig_mapped, dtype=torch.long, device=device).unsqueeze(0).repeat(NUM_SIMS, 1, 1)
        
        with torch.no_grad():
            for step in range(50):
                step_norm = torch.tensor([step / 50.0] * NUM_SIMS, device=device)
                phase_vals = torch.tensor([step % 4] * NUM_SIMS, device=device)
                
                state_one_hot = F.one_hot(current_state, num_classes=8).float().permute(0, 3, 1, 2)
                phase_one_hot = F.one_hot(phase_vals, num_classes=4).float()
                
                logits = model(state_one_hot, step_norm, phase_one_hot)
                probs = F.softmax(logits, dim=1) # [200, 8, 40, 40]
                
                # Reshape for multinomial sampling
                B, C, H, W = probs.shape
                flat_probs = probs.permute(0, 2, 3, 1).reshape(-1, C) # [200*40*40, 8]
                sampled = torch.multinomial(flat_probs, 1).view(B, H, W)
                
                current_state = sampled
                
        # Average the 200 universes
        final_states = current_state # [200, 40, 40]
        mapped_states = map_to_output(final_states) # Maps 8 internal back to 6 final
        
        pred = np.zeros((40, 40, 6), dtype=np.float32)
        final_np = mapped_states.cpu().numpy()
        
        for c in range(6):
            pred[..., c] = (final_np == c).mean(axis=0)
            
        # Hard Structural Zeros (Safe bets)
        pred = np.maximum(pred, 1e-6)
        pred[ig == 5] = [0, 0, 0, 0, 0, 1]
        pred[ig == 10] = [1, 0, 0, 0, 0, 0]
        
        # Beyond dist 12 is deterministic
        from scipy.ndimage import distance_transform_edt
        mask = (ig == 1) | (ig == 2)
        dist = distance_transform_edt(~mask) if mask.any() else np.full_like(ig, 100)
        pred[(dist > 12) & ((ig == 1) | (ig == 2) | (ig == 0) | (ig == 11))] = [1, 0, 0, 0, 0, 0]
        pred[(dist > 12) & (ig == 4)] = [0, 0, 0, 0, 1, 0]
        
        pred /= pred.sum(axis=-1, keepdims=True)
        
        wkl = calc_wkl(pred, gt, ig)
        print(f"R{rn} S{sn}: wKL = {wkl:.4f}")
        total_wkl += wkl
        count += 1
        
    print(f"\nFinal GPU MC Simulator wKL: {total_wkl/count:.4f}")

if __name__ == '__main__':
    train_and_eval()