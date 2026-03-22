import json
import requests
import time
import os
import torch
import torch.nn.functional as F
import numpy as np

# Same definitions as training
RAW_TO_IDX = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 6, 11: 7}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DPCAModel(torch.nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(21, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(hidden_dim, 8, kernel_size=1)
        
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
        
        port_penalty = torch.zeros_like(logits)
        port_penalty[:, 2:3, :, :] = inland_mask * -100.0
        logits = logits + port_penalty
        
        ruin_t = x[:, 3:4, :, :]
        ruin_penalty = torch.zeros_like(logits)
        ruin_penalty[:, 3:4, :, :] = ruin_t * -100.0
        logits = logits + ruin_penalty
        
        empty_t = torch.clamp(x[:, 0:1, :, :] + x[:, 7:8, :, :], 0, 1)
        forest_penalty = torch.zeros_like(logits)
        forest_penalty[:, 4:5, :, :] = empty_t * -100.0
        logits = logits + forest_penalty
        
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

def to_onehot(grid_idx, num_classes=8):
    if not isinstance(grid_idx, torch.Tensor):
        grid_idx = torch.tensor(grid_idx)
    return F.one_hot(grid_idx.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

def solve_astar():
    with open("tasks/astar-island/.token") as f:
        token = f.read().strip()

    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    s.cookies.set("access_token", token)

    print("Loading DPCA model...")
    model = DPCAModel(hidden_dim=128).to(device)
    model.load_state_dict(torch.load("tasks/astar-island/dpca_model.pth"))
    model.eval()

    rounds = s.get("https://api.ainm.no/astar-island/rounds").json()
    active_round = next((r for r in rounds if r["status"] == "active"), None)
    if not active_round:
        print("No active round found!")
        return

    round_id = active_round["id"]
    print(f"Active round: {active_round['round_number']}")

    detail = s.get(f"https://api.ainm.no/astar-island/rounds/{round_id}").json()
    
    # We will use regime=1.0 since hypothesis testing says it's not detectable early anyway
    regime_t = torch.tensor([1.0], device=device).float()

    for seed_idx, initial_state in enumerate(detail["initial_states"]):
        print(f"Solving Seed {seed_idx}...")
        
        grid = np.array(initial_state["grid"])
        idx_init = np.vectorize(RAW_TO_IDX.get)(grid)
        init_x_oh = to_onehot(np.expand_dims(idx_init, 0)).to(device)
        
        x_t = init_x_oh.clone()
        with torch.no_grad():
            for step in range(50):
                step_t = torch.tensor([step], device=device)
                # Soft continuous rollout works best for predicting the probability distribution
                x_t = model(x_t, init_x_oh, step_t, regime_t, gumbel=False)

        pred_probs_8 = x_t
        pred_probs = torch.zeros(1, 6, 40, 40, device=device)
        pred_probs[:, 0] = pred_probs_8[:, 0] + pred_probs_8[:, 6] + pred_probs_8[:, 7]
        pred_probs[:, 1] = pred_probs_8[:, 1]
        pred_probs[:, 2] = pred_probs_8[:, 2]
        pred_probs[:, 3] = pred_probs_8[:, 3]
        pred_probs[:, 4] = pred_probs_8[:, 4]
        pred_probs[:, 5] = pred_probs_8[:, 5]
        
        # Apply the exact competition floor and renormalize
        pred_probs = torch.clamp(pred_probs, min=0.01)
        pred_probs = pred_probs / pred_probs.sum(dim=1, keepdim=True)
        
        # Format for submission: H x W x 6
        submission_tensor = pred_probs[0].permute(1, 2, 0).cpu().numpy().tolist()

        resp = s.post("https://api.ainm.no/astar-island/submit", json={
            "round_id": round_id,
            "seed_index": seed_idx,
            "prediction": submission_tensor
        })
        
        if resp.status_code == 200:
            print(f"Seed {seed_idx} submitted successfully!")
        else:
            print(f"Seed {seed_idx} submission failed: {resp.status_code} {resp.text}")

if __name__ == "__main__":
    solve_astar()