import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import torchvision.transforms as transforms
from diffusers import AutoencoderDC
from typing import Optional, Union
from torchsummary import summary
import numpy as np
torch.backends.cudnn.enabled = False

def modulate(x, shift, scale):
    # unsqueeze to help broadcast for the batch dimension
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

"""transforms= transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(0.5, 0.5)])
transforms need to come from the dataloader"""

class ImageEmbedder(nn.Module):
    def __init__(self,
                 model_id: str = "mit-han-lab/dc-ae-f64c128-in-1.0-diffusers",
                 image_size: int = 256,
                 device = "cuda" if torch.cuda.is_available() else "cpu"
                 ):
        super().__init__()
        
        self.ae = AutoencoderDC.from_pretrained(model_id, torch_dtype =torch.float32).to(device).eval()
        self.transforms = transforms
        self.num_patches = (image_size//64)**2 #32x compression factor, so 4X4 patches

    def forward(self, x):
        with torch.no_grad():
            latent = self.ae.encode(x).latent
        # return latent.flatten(2).transpose(1, 2) #confirm if it works in this way, latent is expected to be N, C, H, W
        assert type(latent) == torch.Tensor, "latent should be a torch.Tensor"
        return latent.flatten(2).transpose(1, 2)  # DEVICE: ensure latent is on the same device as the model

class LabelEmbedder(nn.Module):
    """In this case I wanna experiment by initialising the class embeddings
    by using the textual embeddings of the CLIP Text Encoder after that I'll
    initialise them by encoding the properties of the similar type pokemons together
    for e.g. fire and water can be thought of as vectors on opposite sides of the vector space

    In our case we have made CLIPtext Embeddings for the base 18 classes
    then combination of types is simulated by vector additions of the base class

    for e.g a fire and steel type's class token is made by vec_fire + vec_steel"""

    def __init__(self,
                 lookup_table: Union[list, torch.Tensor],
                 dropout_prob: float,
                 device = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        if isinstance(lookup_table, list):
            num_classes = len(lookup_table)
            initial_embeddings = [nn.Parameter(t, requires_grad=False) for t in lookup_table]
            self.embedding_table = nn.ParameterList(initial_embeddings).to(device)  # DEVICE: ensure all parameters are on the correct device

        elif isinstance(lookup_table, torch.Tensor):
            num_classes = lookup_table.shape[0]
            self.embedding_table = [nn.Parameter(lookup_table[i], requires_grad=False) for i in range(num_classes)]

        else:
            raise TypeError("lookup_table must be either a torch.Tensor or list of tensors")

        hidden_size = lookup_table[0].shape[0]

        dtype = self.embedding_table[0].dtype if len(self.embedding_table) > 0 else torch.float32
        device = self.embedding_table[0].device if len(self.embedding_table) > 0 else torch.device("cpu")  # DEVICE: overriding input `device`, may need to handle explicitly if inconsistency is observed

        cfg_embedding = nn.Parameter(torch.randn(hidden_size, dtype=dtype, device=device), requires_grad=True)  # DEVICE: correctly initialized on the same device

        if use_cfg_embedding:
            self.embedding_table = nn.ParameterList(list(self.embedding_table)+ [cfg_embedding])  # ✅ FIXED


        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob  # DEVICE: rand must be on same device as labels
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        # assuming labels to be of size (N,)
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)

        embeddings = F.embedding(labels, torch.stack(list(self.embedding_table)))  # DEVICE: stacked tensor must match label device
        return embeddings


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, use_bias=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=use_bias),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=use_bias),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000, freq_scale=1):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                  These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :freq_scale: controls the scaling factor of the frequencies
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = freq_scale * torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)  # DEVICE: ensure freqs are on same device as t

        args = t[:, None].float() * freqs[None]  # DEVICE: t and freqs must match
        args = args.unsqueeze(-1)
        # print(args.shape)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # DEVICE: cos/sin keep device consistency
        # print(embedding.shape)
        embedding = embedding.flatten(1)

        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)  # DEVICE: ensure zeros_like stays on correct device

        return embedding

    def forward(self, t, freq_scale=1):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size, freq_scale=freq_scale)
        t_emb = self.mlp(t_freq)  # DEVICE: ensure self.mlp is on the same device as t_freq
        return t_emb




class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#################################################################################
#                                   MLP                                         #
#################################################################################


class Mlp(nn.Module):
    # Adapted from `timm`.
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = bias if isinstance(bias, tuple) else (bias, bias)
        drop_probs = drop if isinstance(drop, tuple) else (drop, drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, drop=0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class nanoDit(nn.Module):
    def __init__(
            self, 
            hidden_size: int,
            num_heads: int,
            depth: int,
            lookup_table: Union[list, torch.tensor],
            device,
            mlp_ratio = 4.0,
            time_step_freq_scale = 1.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.timestep_freq_scale = time_step_freq_scale

        self.ImageEmbedder = ImageEmbedder(model_id="mit-han-lab/dc-ae-f64c128-in-1.0-diffusers")  # DEVICE: ensure this component handles device logic internally
        self.LabelEmbedder = LabelEmbedder(lookup_table=lookup_table, dropout_prob=0.1, device=device)  # DEVICE: pass actual device (change "cpu" if needed)
        self.TimeEmbedder = TimestepEmbedder(hidden_size=hidden_size)

        self.pose_embed = nn.Parameter(torch.zeros(1, self.ImageEmbedder.num_patches, hidden_size), requires_grad=False)  # DEVICE: move to correct device after init if needed

        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio) for i in range(depth)])  # DEVICE: blocks will need `.to(device)` if model is moved

    def initalize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pose_embed = get_2d_sincos_pos_embed(self.pose_embed.shape[-1], int(self.ImageEmbedder.num_patches**0.5))
        print("pose_embed shape:", pose_embed.shape, int(self.ImageEmbedder.num_patches**0.5))
        self.pose_embed.data.copy_(torch.from_numpy(pose_embed).float().unsqueeze(0))  # DEVICE: copy must match self.pose_embed device

        nn.init.normal_(self.TimeEmbedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.TimeEmbedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    # x should be a Image-> from transforms totensor/Normalize/Resize, t-> (N,) tensor of timesteps, y-> (N,) 
    def forward(self, x, t, y):
        # DEVICE: assume x, t, y are already on the same device as the model
        latent = self.ImageEmbedder(x)
        print(latent.device, self.pose_embed.device)  # DEVICE: check if all inputs are on the same device
        
        x =  latent + self.pose_embed  # DEVICE: ensure pose_embed is on the same device as x
        t = self.TimeEmbedder(t, freq_scale=self.timestep_freq_scale)  # DEVICE: TimeEmbedder must match device
        y = self.LabelEmbedder(y, self.training)  # DEVICE: LabelEmbedder must return output on same device

        c = t + y  # (N, D)

        for block in self.blocks:
            x = block(x, c=c)  # DEVICE: x and c must match device

        return x, latent

            

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

if __name__ =="__main__":
    lookup_table = torch.tensor(np.load('og_lookup.npy'))
    assert lookup_table.shape == (152, 128)
    nanoDit(hidden_size=128, num_heads=8, depth=24, lookup_table=lookup_table, device="cuda")  # DEVICE: ensure model is on the correct device
    dit = nanoDit(128, 8, 24, lookup_table=lookup_table, device="cuda").to("cuda")  # DEVICE: ensure model is on the correct device
    dit.initalize_weights()  
    print("model initialized")
    x = torch.randn((1, 3, 256, 256), device="cuda")  # DEVICE: ensure input tensor is on the same device as the model
    t = torch.randint(2, (1, ), device="cuda")  # DEVICE: ensure timestep tensor is on the same device as the model
    y = torch.randint(2, (1, ), device="cuda")  # DEVICE: ensure label tensor is on the same device as the model

    assert t.shape == (1,)
    assert y.shape == (1,)
    print("shapes  have been asserted")
    
    z1 = dit(x, t, y)
    print("forward pass done", z1.shape)