import torch
import numpy as np
from diffusion_model import MinimalDiffusionModel, MinimalDiffusionScheduler, sample_from_minimal_diffusion

model = MinimalDiffusionModel(max_size=32)
model.load_state_dict(torch.load('continuous_diffusion_model.pth', map_location='cpu'))
model.eval()
scheduler = MinimalDiffusionScheduler(num_timesteps=50)
sizes = [(10,12),(12,10),(8,15),(15,8),(11,11),(9,13),(13,9),(14,10),(10,14),(12,12)]
results = []
for i, (rows, cols) in enumerate(sizes):
    size_info = torch.tensor([[rows, cols]], dtype=torch.float32)
    sample = sample_from_minimal_diffusion(model, scheduler, size_info, num_steps=25, device='cpu', threshold=0.5)
    mask = sample[0, 1, :rows, :cols].cpu().numpy()
    mask = (mask > 0.5).astype(int)
    results.append(mask)
    print(f'--- {i+1} ---')
    print(mask) 