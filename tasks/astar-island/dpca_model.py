import json
import glob
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RAW_TO_IDX = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 6, 11: 7}

class DPCAModel(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(21, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim, 8, kernel_size=1)

    def forward(self, x, initial_x, step, regime_scalar, temperature=1.0, gumbel=False):
        B, _, H, W = x.shape

        phase = (step % 4).long()
        phase_onehot = F.one_hot(phase, num_classes=4).float()
        phase_map = phase_onehot.view(B, 4, 1, 1).expand(B, 4, H, W)

        regime_map = regime_scalar.view(B, 1, 1, 1).expand(B, 1, H, W)

        inputs = torch.cat([x, initial_x, phase_map, regime_map], dim=1)        
        h = F.relu(self.conv1(inputs))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        logits = self.conv4(h)
        
        mountain_mask = initial_x[:, 5:6, :, :]
        ocean_mask = initial_x[:, 6:7, :, :]
        
        ocean_adjacent = F.max_pool2d(ocean_mask, kernel_size=3, padding=1, stride=1)
        inland_mask = 1.0 - ocean_adjacent
        
        # Apply strict negative logits for impossible states BEFORE softmax/gumbel
        
        # 1. Inland port penalty
        port_penalty = torch.zeros_like(logits)
        port_penalty[:, 2:3, :, :] = inland_mask * -100.0
        logits = logits + port_penalty
        
        # 2. Ruins (3) last exactly 1 step. If x is Ruin, next state cannot be Ruin.
        ruin_t = x[:, 3:4, :, :]
        ruin_penalty = torch.zeros_like(logits)
        ruin_penalty[:, 3:4, :, :] = ruin_t * -100.0
        logits = logits + ruin_penalty
        
        # 3. Empty (0) and Plains (7) can NEVER directly become Forest (4).
        empty_t = torch.clamp(x[:, 0:1, :, :] + x[:, 7:8, :, :], 0, 1)
        forest_penalty = torch.zeros_like(logits)
        forest_penalty[:, 4:5, :, :] = empty_t * -100.0
        logits = logits + forest_penalty
        
        # 4. Forest (4) can NEVER directly become Empty (0) or Plains (7).
        forest_t = x[:, 4:5, :, :]
        empty_penalty = torch.zeros_like(logits)
        empty_penalty[:, 0:1, :, :] = forest_t * -100.0
        empty_penalty[:, 7:8, :, :] = forest_t * -100.0
        logits = logits + empty_penalty
        
        if gumbel:
            probs = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=1)
        else:
            probs = F.softmax(logits / temperature, dim=1)
        
        probs_adjusted = probs.clone()
        probs_adjusted[:, 5:6, :, :] = probs[:, 5:6, :, :] * mountain_mask
        probs_adjusted[:, 6:7, :, :] = probs[:, 6:7, :, :] * ocean_mask
        
        static_mask = torch.clamp(mountain_mask + ocean_mask, 0, 1)
        probs_final = probs_adjusted * (1.0 - static_mask) + initial_x * static_mask
        
        probs_final = probs_final / (probs_final.sum(dim=1, keepdim=True) + 1e-8)
        return probs_final

def get_all_regimes():
    files = glob.glob('tasks/astar-island/replays/*.json') + glob.glob('tasks/astar-island/replays_expanded/*.json')
    round_regimes = {}
    for f in files:
        rnd = int(os.path.basename(f).split('_')[0].replace('round', ''))
        with open(f, 'r') as fh: data = json.load(fh)
        frames = data['frames']
        init_sett = np.sum((np.array(frames[0]['grid']) == 1) | (np.array(frames[0]['grid']) == 2))
        step10_sett = np.sum((np.array(frames[min(10, len(frames)-1)]['grid']) == 1) | (np.array(frames[min(10, len(frames)-1)]['grid']) == 2))
        growth = step10_sett / (init_sett + 1e-5)
        if rnd not in round_regimes: round_regimes[rnd] = []
        round_regimes[rnd].append(growth)
    
    for r in round_regimes:
        round_regimes[r] = np.mean(round_regimes[r])
    return round_regimes

def load_replays_for_round(exclude_rounds=[], round_regimes=None):
    files = glob.glob('tasks/astar-island/replays/*.json') + glob.glob('tasks/astar-island/replays_expanded/*.json')
    X, Y, INITIAL_X, STEPS, REGIMES = [], [], [], [], []
    
    for f in files:
        rnd = int(os.path.basename(f).split('_')[0].replace('round', ''))
        if rnd in exclude_rounds: continue
            
        with open(f, 'r') as fh: data = json.load(fh)
        frames = data['frames']
        regime = round_regimes[rnd]
        initial_grid = np.array(frames[0]['grid'])
        idx_initial = np.vectorize(RAW_TO_IDX.get)(initial_grid)
        
        for t in range(len(frames) - 1):
            grid_t = np.array(frames[t]['grid'])
            grid_t1 = np.array(frames[t+1]['grid'])
            idx_t = np.vectorize(RAW_TO_IDX.get)(grid_t)
            idx_t1 = np.vectorize(RAW_TO_IDX.get)(grid_t1)
            
            X.append(idx_t)
            INITIAL_X.append(idx_initial)
            Y.append(idx_t1)
            STEPS.append(frames[t]['step'])
            REGIMES.append(regime)
            
    return np.array(X), np.array(INITIAL_X), np.array(Y), np.array(STEPS), np.array(REGIMES)

def to_onehot(grid_idx, num_classes=8):
    if not isinstance(grid_idx, torch.Tensor):
        grid_idx = torch.tensor(grid_idx)
    return F.one_hot(grid_idx.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

def train_phase1(model, X, INITIAL_X, Y, STEPS, REGIMES, epochs=10, batch_size=256):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Move entire dataset to GPU for massive speedup
    X_t = torch.tensor(X, device=device)
    INIT_X_t = torch.tensor(INITIAL_X, device=device)
    Y_t = torch.tensor(Y, device=device).long()
    STEPS_t = torch.tensor(STEPS, device=device)
    REGIMES_t = torch.tensor(REGIMES, device=device).float()
    
    dataset = torch.utils.data.TensorDataset(X_t, INIT_X_t, Y_t, STEPS_t, REGIMES_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_init_x, batch_y, batch_step, batch_reg in loader:
            batch_x_oh = to_onehot(batch_x).to(device)
            batch_init_x_oh = to_onehot(batch_init_x).to(device)
            
            optimizer.zero_grad()
            probs = model(batch_x_oh, batch_init_x_oh, batch_step, batch_reg, gumbel=False)
            loss = F.nll_loss(torch.log(probs + 1e-8), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"Phase 1 Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")

def compute_kl_divergence(pred_probs, gt_probs, H_gt=None):
    # Apply competition floor
    pred_probs = torch.clamp(pred_probs, min=0.01)
    pred_probs = pred_probs / pred_probs.sum(dim=1, keepdim=True)
    
    gt_probs = torch.clamp(gt_probs, min=0.01)
    gt_probs = gt_probs / gt_probs.sum(dim=1, keepdim=True)
    
    kl = torch.sum(gt_probs * (torch.log(gt_probs) - torch.log(pred_probs)), dim=1)
    if H_gt is not None:
        wkl = kl * H_gt
        dynamic = H_gt > 0.01
        if dynamic.sum() > 0:
            return wkl[dynamic].mean()
        return torch.tensor(0.0, device=pred_probs.device)
    return kl.mean()

def train_phase2_bptt(model, exclude_round, round_regimes, epochs=30):
    files = glob.glob('tasks/astar-island/ground_truth/*.json')
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_gts = []
    for f in files:
        rnd = int(os.path.basename(f).split('_')[0].replace('round', ''))
        if rnd == exclude_round: continue
        with open(f, 'r') as fh: train_gts.append((rnd, json.load(fh)))
            
    model.train()
    for epoch in range(epochs):
        total_kl = 0
        for rnd, data in train_gts:
            regime = round_regimes.get(rnd, 1.0)
            regime_t = torch.tensor([regime], device=device).float()
            
            init_grid = np.array(data['initial_grid'])
            idx_init = np.vectorize(RAW_TO_IDX.get)(init_grid)
            init_x_oh = to_onehot(np.expand_dims(idx_init, 0)).to(device)
            
            # Use small parallel rollout (4) for smooth probability estimate
            B_rollout = 4
            x_t = init_x_oh.repeat(B_rollout, 1, 1, 1)
            init_x_oh_b = init_x_oh.repeat(B_rollout, 1, 1, 1)
            regime_t_b = regime_t.repeat(B_rollout)
            
            gt_probs = torch.tensor(data['ground_truth']).permute(2,0,1).unsqueeze(0).to(device)
            gt_probs = torch.clamp(gt_probs, min=0.01)
            gt_probs = gt_probs / gt_probs.sum(dim=1, keepdim=True)
            H_gt = -torch.sum(gt_probs * torch.log(gt_probs), dim=1)
            
            optimizer.zero_grad()
            for step in range(50):
                step_t = torch.tensor([step]*B_rollout, device=device)
                x_t = model(x_t, init_x_oh_b, step_t, regime_t_b, gumbel=False, temperature=1.0)
                
            pred_probs_8 = x_t.mean(dim=0, keepdim=True)
            
            out_probs = torch.zeros(1, 6, 40, 40, device=device)
            out_probs[:, 0] = pred_probs_8[:, 0] + pred_probs_8[:, 6] + pred_probs_8[:, 7]
            out_probs[:, 1] = pred_probs_8[:, 1]
            out_probs[:, 2] = pred_probs_8[:, 2]
            out_probs[:, 3] = pred_probs_8[:, 3]
            out_probs[:, 4] = pred_probs_8[:, 4]
            out_probs[:, 5] = pred_probs_8[:, 5]
            
            loss = compute_kl_divergence(out_probs, gt_probs, H_gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_kl += loss.item()
        scheduler.step()
        print(f"Phase 2 Epoch {epoch+1}/{epochs} | BPTT wKL: {total_kl/len(train_gts):.4f}")

def validate_round(model, exclude_round, round_regimes, n_sims=300):
    files = glob.glob(f'tasks/astar-island/ground_truth/round{exclude_round}_seed*.json')
    if not files: return
        
    model.eval()
    kls, wkls = [], []
    with torch.no_grad():
        for f in files:
            with open(f, 'r') as fh: data = json.load(fh)
            regime = round_regimes.get(exclude_round, 1.0)
            regime_t = torch.tensor([regime], device=device).float()
            
            init_grid = np.array(data['initial_grid'])
            idx_init = np.vectorize(RAW_TO_IDX.get)(init_grid)
            init_x_oh = to_onehot(np.expand_dims(idx_init, 0)).to(device)
            x_t = init_x_oh.clone()
            
            for step in range(50):
                step_t = torch.tensor([step], device=device)
                x_t = model(x_t, init_x_oh, step_t, regime_t, gumbel=False)
                
            pred_probs_8 = x_t
            pred_probs = torch.zeros(1, 6, 40, 40, device=device)
            pred_probs[:, 0] = pred_probs_8[:, 0] + pred_probs_8[:, 6] + pred_probs_8[:, 7]
            pred_probs[:, 1] = pred_probs_8[:, 1]
            pred_probs[:, 2] = pred_probs_8[:, 2]
            pred_probs[:, 3] = pred_probs_8[:, 3]
            pred_probs[:, 4] = pred_probs_8[:, 4]
            pred_probs[:, 5] = pred_probs_8[:, 5]
            
            pred_probs = torch.clamp(pred_probs, min=1e-6)
            pred_probs = pred_probs / pred_probs.sum(dim=1, keepdim=True)
            
            gt_probs = torch.tensor(data['ground_truth']).permute(2,0,1).unsqueeze(0).to(device)
            gt_probs = torch.clamp(gt_probs, min=0.01)
            gt_probs = gt_probs / gt_probs.sum(dim=1, keepdim=True)
            H_gt = -torch.sum(gt_probs * torch.log(gt_probs), dim=1)
            
            kl = compute_kl_divergence(pred_probs, gt_probs)
            wkl = compute_kl_divergence(pred_probs, gt_probs, H_gt)
            kls.append(kl.item())
            wkls.append(wkl.item())
            
    print(f"Validation R{exclude_round} (Soft Rollout) | wKL: {np.mean(wkls):.5f} | KL: {np.mean(kls):.5f}")

if __name__ == "__main__":
    exclude_round = -1
    print("Computing regimes...")
    round_regimes = get_all_regimes()

    print("Loading data...")
    X, INIT_X, Y, STEPS, REGIMES = load_replays_for_round(exclude_rounds=[exclude_round], round_regimes=round_regimes)
    print(f"Loaded {len(X)} 1-step transitions.")

    model = DPCAModel(hidden_dim=256).to(device)
    model = torch.compile(model)

    print("\n--- Phase 1: 1-Step Pre-training ---")
    train_phase1(model, X, INIT_X, Y, STEPS, REGIMES, epochs=20) 
    
    print("\n--- Phase 2: 50-Step BPTT (KL Div) ---")
    train_phase2_bptt(model, exclude_round, round_regimes, epochs=50)
    
    print("\n--- Final Validation ---")
    validate_round(model, exclude_round, round_regimes, n_sims=300)
    
    print("\nSaving model weights to tasks/astar-island/dpca_model.pth...")
    torch.save(model.state_dict(), "tasks/astar-island/dpca_model.pth")
    print("Done!")
