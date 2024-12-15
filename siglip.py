from typing import Optional, Tuple
import torch
import torch.nn as nn


class SiglipVisionConfig:
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid"  # no padding is added
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(
            self.num_positions,
            self.embed_dim
        )
        self.register_buffer(
            "position_ids",
            torch.arrange(self.num_positions).expand((-1, 1)),
            persistent=False
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # [Batch_size, Channels, Height, Width]
        _, _, height, width = pixel_values.shape
        # After convolution the output is of dimension [Batch_size, Embedding_Dimension, Num_Patches_H, Num_Patches_W]

        patch_embeds = self.patch_embedding(pixel_values)

        # [Batch_size, Embedding_Dimension, Num_Patches_H, Num_Patches_W] -> [Batch_Size, Embeddings_Dimension, Num_Patches]
        embeddings = patch_embeds.flatten(2)

        # [Batch_Size, Embedding_Dimension, Num_Patches] -> [Batch_Size, Num_Patches, Embedding_Dimension]
        embeddings = embeddings.transpose(1, 2)

        # Adding positional embeddings
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


class SigLipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layer1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.layer2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Patches, Embedding_Dimension] -> [Batch_Size, Num_Patches, Intermediate_Dimension]
        hidden_states = self.layer1(hidden_states)

        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")

        # [Batch_Size, Num_Patches, Intermediate_Dimension] -> [Batch_Size, Num_Patches, Embedding_Dimension]
        hidden_states = self.layer2(hidden_states)

        return hidden_states


class SigLipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SigLipAttetion(config)
        self.layer_norm = nn.LayerNorm(
            self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)
        self.layer_norm2 = nn.LayerNorm(
            self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.layer_norm(hidden_states)

        hidden_states, _ = self.self_attn(hidden_states=hidden_states)

        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.layer_norm2(hidden_states)

        hidden_states = self.mlp(hidden_states)

        output_state = residual + hidden_states

        return output_state


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SigLipEncoder(config)
        self.post_layernorm = nn.LayerNorm(
            embed_dim,
            eps=config.layer_norm_eps
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # [Pixel_Values: [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embedding_Dimension]]
        hidden_states = self.embeddings(pixel_values)

        last_hidden_state = self.encoder(input_embeds=hidden_states)

        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embedding_Dimension]
        return self.vision_model(pixel_values=pixel_values)
