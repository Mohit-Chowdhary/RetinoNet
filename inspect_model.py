import torch

# Load your model file
MODEL_PATH = 'best_model.pth'
checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

# Check the structure
if isinstance(checkpoint, dict):
    print("Checkpoint is a dictionary with keys:", checkpoint.keys())
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

# Print first 20 layer names
print("\n=== First 20 Layer Names ===")
for i, key in enumerate(list(state_dict.keys())[:20]):
    print(f"{i+1}. {key}")

print("\n=== Last 5 Layer Names ===")
for i, key in enumerate(list(state_dict.keys())[-5:]):
    print(f"{i+1}. {key}")

print(f"\n=== Total Parameters: {len(state_dict.keys())} ===")
