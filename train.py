"""
Thanks to Gemini 2.5 for pairing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from model import NanoDiT
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision

# --- Hyperparameters ---
NUM_CLASSES = 10  # Example: CIFAR-10 or your number of classes
IMG_SIZE = 32  # Example: CIFAR-10 image size
IMG_CHANNELS = 3  # Example: CIFAR-10 image channels
# DiT specific parameters
LATENT_DIM = 384
PATCH_SIZE = 2
MODEL_DEPTH = 6
MODEL_HEADS = 8
# Diffusion process parameters
NUM_TIMESTEPS = 1000
# Training parameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPILE = False
# Sampling parameters
SAMPLE_INTERVAL = 10  # Sample every N epochs
NUM_SAMPLES_PER_CLASS = 4  # Number of images to sample per class during evaluation
CFG_SCALE = 5.0
# Others
CHECKPOINT_SAVE_INTERVAL = 25
DATA_DIR = "." # directory to store CIFAR10.

# --- Helper Functions for diffusion math ---
def extract(a, t, x_shape):
    """Extracts coefficients at specific timesteps t and reshapes to x_shape."""
    batch_size = t.shape[0]
    out = a.gather(-1, t.long())  # Ensure t is long
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def q_sample(x_start, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, t, noise=None):
    """Forward diffusion process: q(x_t | x_0)."""
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise

def train():
    # --- Diffusion Schedule (Linear) ---
    # Define all the co-efficients.
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(beta_start, beta_end, NUM_TIMESTEPS, device=DEVICE)

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    # --- Model Instantiation ---
    model = NanoDiT(  # Pass your DiT args here
        input_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=IMG_CHANNELS,
        hidden_size=LATENT_DIM,
        depth=MODEL_DEPTH,
        num_heads=MODEL_HEADS,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)
    if COMPILE:
        model.compile()

    # --- Optimizer and loss  ---
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()  # Loss is typically MSE between predicted noise and actual noise

    # --- Dataset and DataLoader  ---
    trfs_cifar = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
        ]
    )

    # Download and load the CIFAR-10 training dataset
    train_dataset = torchvision.datasets.CIFAR10(train=True, root=DATA_DIR, download=True, transform=trfs_cifar)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,  # Adjust based on your system
        pin_memory=True,  # Useful when training on GPU
        drop_last=True,
    )


    # --- Sampling Functions (DDPM Ancestral Sampler) ---
    @torch.no_grad()
    def p_sample_cfg(model, x, t, t_index_sampler, conditional_class_labels, cfg_scale, null_class_token_id):
        """Single step of the reverse diffusion process with CFG."""
        # Enable CFG in model
        n = conditional_class_labels.shape[0]
        y = conditional_class_labels
        y_null = torch.tensor([null_class_token_id] * n, device=conditional_class_labels.device)
        y = torch.cat([y, y_null], 0)
        noise_pred = model(x, t, y)

        # Perform CFG
        pred_noise_cond, pred_noise_uncond = noise_pred.chunk(2, dim=0)
        noise_pred = pred_noise_uncond + cfg_scale * (pred_noise_cond - pred_noise_uncond)

        # DDPM sampling step
        betas_t = extract(betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)

        if t_index_sampler == 0:
            return model_mean
        else:
            posterior_variance_t = extract(posterior_variance, t, x.shape)
            noise_z = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise_z


    @torch.no_grad()
    def sample_conditional_cfg(
        model_to_sample, target_classes_list, cfg_scale_val=5.0, num_samples_per_cls=1, null_token_id=NUM_CLASSES
    ):
        """Generate images for specified target classes using CFG."""
        model_to_sample.eval()
        num_target_cls = len(target_classes_list)
        total_images_to_sample = num_samples_per_cls * num_target_cls
        current_images = torch.randn((total_images_to_sample, IMG_CHANNELS, IMG_SIZE, IMG_SIZE), device=DEVICE)

        # Prepare conditional labels
        sample_cls_labels_list = []
        for c_idx in target_classes_list:
            sample_cls_labels_list.extend([c_idx] * num_samples_per_cls)
        conditional_labels = torch.tensor(sample_cls_labels_list, device=DEVICE).long()

        for i in reversed(range(0, NUM_TIMESTEPS)):
            t_for_sampling = torch.full((total_images_to_sample,), i, device=DEVICE, dtype=torch.long)
            current_images = p_sample_cfg(
                model_to_sample, current_images, t_for_sampling, i, conditional_labels, cfg_scale_val, null_token_id
            )

        current_images = (current_images + 1) / 2.0  # De-normalize from [-1, 1] to [0, 1]
        current_images = torch.clamp(current_images, 0.0, 1.0)

        model_to_sample.train()
        return current_images, conditional_labels


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

            # 1. Sample timesteps
            t_timesteps = torch.randint(0, NUM_TIMESTEPS, (real_images.shape[0],), device=DEVICE).long()
            # 2. Noise images according to t (forward process q(x_t | x_0))
            xt_noisy_images, added_noise_epsilon = q_sample(
                real_images, sqrt_alphas_cumprod,sqrt_one_minus_alphas_cumprod, t=t_timesteps
            )
            # 3. Predict noise using the model
            pred_noise = model(xt_noisy_images, t_timesteps, class_ids)

            # 4. Calculate loss
            loss = criterion(pred_noise, added_noise_epsilon)
            # 5. Backpropagate and update weights
            loss.backward()
            optimizer.step()

            if (step + 1) % 50 == 0:  # Log less frequently
                print(f"Epoch [{epoch + 1}/{EPOCHS}], Step [{step + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

        # --- Perform Sampling and Save Images (Intermediate Evaluation) ---
        if (epoch + 1) % SAMPLE_INTERVAL == 0 or epoch == EPOCHS - 1:
            print(f"\nSampling images at epoch {epoch + 1}...")
            classes_to_sample_list = list(range(min(NUM_CLASSES, 5)))
            generated_sample_images, _ = sample_conditional_cfg(
                model, classes_to_sample_list, cfg_scale_val=CFG_SCALE, num_samples_per_cls=NUM_SAMPLES_PER_CLASS
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