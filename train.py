"""
Thanks to Gemini 2.5 for pairing here and there.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from model import NanoDiT
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from contextlib import nullcontext

# --- Hyperparameters ---
NUM_CLASSES = 75  
IMG_SIZE = 64
IMG_CHANNELS = 3 
# DiT specific parameters
LATENT_DIM = 768
PATCH_SIZE = 2
MODEL_DEPTH = 12
MODEL_HEADS = 12
# Diffusion process parameters
NUM_ODE_STEPS = 150
# Training parameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
EPOCHS = 600
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPILE = True
AMP_DTYPE = torch.bfloat16 # automatic mixed-precision unless you wanna touch grass
# Sampling parameters
SAMPLE_INTERVAL = 10  # Sample every N epochs
NUM_SAMPLES_PER_CLASS = 4  # Number of images to sample per class during evaluation
CFG_SCALE = 5.0
# Others
CHECKPOINT_SAVE_INTERVAL = 25
DATA_DIR = "butterflies" # Directory where the dataset is stored.

# --- Helper functions for flow-matching math ---
def flow_lerp(x):
    bsz = x.shape[0]
    t = torch.rand((bsz,)).to(x.device)
    t_broadcast = t.view(bsz, 1, 1, 1)
    z1 = torch.randn_like(x)
    # the flow-matching lerp
    zt = (1 - t_broadcast) * x + t_broadcast * z1
    return zt, z1, t

# --- Sampling function (Euler) ---
@torch.no_grad()
def sample_conditional_cfg(
    model, target_classes_list, ode_steps, cfg_scale=5.0, num_samples_per_cls=1, null_token_id=NUM_CLASSES
):
    """Generate images for specified target classes using CFG."""
    model.eval()
    num_target_cls = len(target_classes_list)
    total_images_to_sample = num_samples_per_cls * num_target_cls

    # Initial state
    z = torch.randn((total_images_to_sample, IMG_CHANNELS, IMG_SIZE, IMG_SIZE), device=DEVICE)

    # Prepare conditional labels
    sample_cls_labels_list = []
    for c_idx in target_classes_list:
        sample_cls_labels_list.extend([c_idx] * num_samples_per_cls)
    conditional_labels = torch.tensor(sample_cls_labels_list, device=DEVICE).long()

    # CFG stuff
    using_cfg = cfg_scale >= 1.0
    y = conditional_labels
    if using_cfg:
        n = conditional_labels.shape[0]
        y_null = torch.tensor([null_token_id] * n, device=conditional_labels.device)
        y = torch.cat([y, y_null], 0)

    # ODE derivative term
    bsz = z.shape[0]
    dt = 1.0 / ode_steps
    dt = torch.tensor([dt] * bsz).to(z.device).view([bsz, *([1] * len(z.shape[1:]))])
    
    # Flow-sampling
    for i in range(ode_steps, 0, -1):
        z_final = torch.cat([z, z], 0) if using_cfg else z
        t = i / ode_steps
        t = torch.tensor([t] * z_final.shape[0]).to(z_final.device, dtype=z_final.dtype)
        predicted_velocity = model(z_final, t, y)
        if using_cfg:
            pred_cond, pred_uncond = predicted_velocity.chunk(2, dim=0)
            predicted_velocity = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
        # Euler step
        z = (z - dt * predicted_velocity).to(z.dtype)
    
    images = (z + 1) / 2.0  # De-normalize from [-1, 1] to [0, 1]
    images = torch.clamp(images, 0.0, 1.0)

    model.train() # Set model to train.
    return images, conditional_labels

def train():
    # --- Model Instantiation ---
    model = NanoDiT(
        input_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=IMG_CHANNELS,
        hidden_size=LATENT_DIM,
        depth=MODEL_DEPTH,
        num_heads=MODEL_HEADS,
        num_classes=NUM_CLASSES,
        timestep_freq_scale=1000,
    ).to(DEVICE)
    if COMPILE:
        print("`torch.compile()` enabled.")
        model.compile()

    # --- Optimizer and loss  ---
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.GradScaler() if AMP_DTYPE is not None else None
    criterion = nn.MSELoss()
    amp_context = (
        torch.autocast(device_type=torch.device(DEVICE).type, dtype=AMP_DTYPE) 
        if AMP_DTYPE is not None
        else nullcontext()
    )
    if AMP_DTYPE:
        print(f"Using automatic mixed-precision in {AMP_DTYPE} (change if needed).")

    # --- Dataset and DataLoader  ---
    ds_trfs = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
        ]
    )
    train_dataset = torchvision.datasets.ImageFolder(DATA_DIR, transform=ds_trfs)
    train_classes = list(set(train_dataset.class_to_idx.values()))
    assert NUM_CLASSES == len(train_classes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,  # Adjust based on your system
        pin_memory=True,  # Useful when training on GPU
        drop_last=True,
        prefetch_factor=2, # Adjust based on your system
    )

    # --- Training Loop ---
    print(f"Training on {DEVICE}")
    print(f"Using custom model: {type(model).__name__}")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    for epoch in range(EPOCHS):
        model.train()
        for step, (real_images, class_ids) in enumerate(train_dataloader):
            optimizer.zero_grad()
            real_images = real_images.to(DEVICE, non_blocking=True)  # Shape: (B, C, H, W)
            class_ids = class_ids.to(DEVICE, non_blocking=True)  # Shape: (B,)

            # 1. Interpolate between clean data and noise
            zt, z1, t = flow_lerp(real_images)
            # 2. Target velocity: u_t = x1 - x0
            target_velocity_field = z1 - real_images
            with amp_context:
                # 3. Predict velocity using the model
                predicted_velocity = model(zt, t, class_ids)
                # 4. Calculate loss
                loss = criterion(target_velocity_field, predicted_velocity)
            # 5. Backpropagate and update weights
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if (step + 1) % 50 == 0:  # Log less frequently
                print(f"Epoch [{epoch + 1}/{EPOCHS}], Step [{step + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

        # --- Perform Sampling and Save Images (Intermediate Evaluation) ---
        if (epoch + 1) % SAMPLE_INTERVAL == 0 or epoch == EPOCHS - 1:
            print(f"\nSampling images at epoch {epoch + 1}...")
            classes_to_sample_list = list(range(min(NUM_CLASSES, 5)))
            generated_sample_images, _ = sample_conditional_cfg(
                model, classes_to_sample_list, ode_steps=NUM_ODE_STEPS,
                cfg_scale=CFG_SCALE, num_samples_per_cls=NUM_SAMPLES_PER_CLASS
            )
            # Save as a grid
            if generated_sample_images.nelement() > 0:  # Check if any images were generated
                grid = torchvision.utils.make_grid(generated_sample_images, nrow=NUM_SAMPLES_PER_CLASS)
                torchvision.utils.save_image(grid, f"sample_epoch_{epoch + 1}.png")
                print(f"Saved sample images to sample_epoch_{epoch + 1}.png")
            print("-" * 30)

        # Optional: Save model checkpoint
        if (epoch + 1) % CHECKPOINT_SAVE_INTERVAL == 0:
            torch.save(model.state_dict(), f"dit_conditional_epoch_{epoch + 1}.pth")

    print("Training finished.")

if __name__ == "__main__":
    train()