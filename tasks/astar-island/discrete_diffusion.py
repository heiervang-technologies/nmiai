import json
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_cond_tensor(initial_grid):
    # Map [0, 1, 2, 4, 5, 10, 11] to one-hot 0..6
    mapping = {0:0, 1:1, 2:2, 4:3, 5:4, 10:5, 11:6}
    ig_mapped = np.vectorize(lambda x: mapping.get(x, 0))(initial_grid)
    one_hot = np.eye(7, dtype=np.float32)[ig_mapped]
    return one_hot.transpose(2, 0, 1)

def train():
    files = glob.glob('tasks/astar-island/ground_truth/*.json')
    data = []
    for f in files:
        d = json.load(open(f))
        ig = np.array(d['initial_grid'])
        gt = np.array(d['ground_truth'])
        data.append((ig, gt))

    conds = []
    gts = []
    for ig, gt in data:
        conds.append(get_cond_tensor(ig))
        gts.append(gt)

    conds = torch.tensor(np.array(conds)).to(device)
    gts = torch.tensor(np.array(gts)).to(device)

    # Model
    class Denoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(7 + 7 + 1, 128, 3, padding=1),
                nn.GroupNorm(8, 128),
                nn.GELU(),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.GroupNorm(8, 128),
                nn.GELU(),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.GroupNorm(8, 128),
                nn.GELU(),
                nn.Conv2d(128, 6, 1)
            )
        def forward(self, cond, state, t):
            t_map = t.expand(-1, 1, 40, 40)
            x = torch.cat([cond, state, t_map], dim=1)
            return self.net(x)

    model = Denoiser().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    print("Training small discrete diffusion model (Absorbing State)...")
    for epoch in range(150):
        model.train()
        opt.zero_grad()
        
        flat_gts = gts.view(-1, 6)
        hard_samples = torch.multinomial(flat_gts, 1).view(-1, 40, 40)
        
        B = hard_samples.shape[0]
        t = torch.rand((B, 1, 1, 1), device=device)
        
        # Masking
        mask = torch.rand((B, 40, 40), device=device) < t.view(B, 1, 1)
        state = hard_samples.clone()
        state[mask] = 6
        
        state_one_hot = F.one_hot(state, num_classes=7).float().permute(0, 3, 1, 2)
        
        logits = model(conds, state_one_hot, t)
        loss = F.cross_entropy(logits, hard_samples)
        
        loss.backward()
        opt.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1} loss: {loss.item():.4f}")

    print("Done training. Generating visualization sequence...")

    model.eval()
    with torch.no_grad():
        ig = data[0][0]
        cond = conds[0:1]
        gt_prob = gts[0:1]
        
        hard_sample = torch.multinomial(gt_prob.view(-1, 6), 1).view(40, 40)
        
        steps = 10
        gt_rewind = []
        mask_order = torch.rand(40, 40, device=device)
        
        # We only mask dynamic cells for better visualization
        is_dynamic = (torch.tensor(ig, device=device) != 10) & (torch.tensor(ig, device=device) != 5)
        
        for i in range(steps + 1):
            t_val = i / steps
            current_mask = (mask_order < t_val) & is_dynamic
            step_state = hard_sample.clone()
            step_state[current_mask] = 6
            gt_rewind.append(step_state.cpu().numpy().tolist())
            
        model_steps = []
        current_state = hard_sample.clone()
        current_state[is_dynamic] = 6 # Start with all dynamic cells masked
        model_steps.append(current_state.cpu().numpy().tolist())
        
        for i in reversed(range(steps)):
            t_val = (i + 1) / steps
            t_tensor = torch.tensor([[[[t_val]]]], device=device, dtype=torch.float32)
            state_one_hot = F.one_hot(current_state.unsqueeze(0), num_classes=7).float().permute(0, 3, 1, 2)
            
            logits = model(cond, state_one_hot, t_tensor)
            probs = F.softmax(logits, dim=1)
            pred_x0 = torch.argmax(probs, dim=1).squeeze(0)
            
            is_masked = (current_state == 6)
            unmask_prob = 1.0 / (i + 1) if i > 0 else 1.0
            unmask_decision = torch.rand((40, 40), device=device) < unmask_prob
            
            do_unmask = is_masked & unmask_decision
            current_state[do_unmask] = pred_x0[do_unmask]
            model_steps.append(current_state.cpu().numpy().tolist())
            
    out_data = {
        'initial_grid': ig.tolist(),
        'gt_rewind': gt_rewind,
        'model_steps': model_steps
    }
    with open('tasks/astar-island/diffusion_vis_data.json', 'w') as f:
        json.dump(out_data, f)
        
    print("Saved diffusion_vis_data.json")

if __name__ == '__main__':
    train()
