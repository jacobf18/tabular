import torch
from diffusion_model import MinimalDiffusionModel, MinimalDiffusionScheduler, sample_from_minimal_diffusion

# Load model
model = MinimalDiffusionModel(max_size=32)
model.load_state_dict(torch.load('hetero_staggered_diffusion_model.pth', map_location='cpu'))
model.eval()
scheduler = MinimalDiffusionScheduler(num_timesteps=20)

# Test different thresholds
size_info = torch.tensor([[10, 12]], dtype=torch.float32)
print('=== THRESHOLD EXPERIMENT ===')

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
for t in thresholds:
    print(f'Threshold {t}:')
    sample = sample_from_minimal_diffusion(model, scheduler, size_info, num_steps=10, device='cpu', threshold=t)
    mask = (sample[0,1,:10,:12].cpu().numpy() > t).astype(int)
    print(mask)
    print(f'Density: {mask.mean():.3f}')
    print() 