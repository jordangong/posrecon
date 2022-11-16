import copy
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from pl_bolts.models.self_supervised.resnets import Bottleneck, ResNet
from timm.models.vision_transformer import PatchEmbed, Block, LayerScale, DropPath
from torch import nn

from utils.pos_embed import get_2d_sincos_pos_embed


class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor)
                           for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input,
                                     op=torch.distributed.ReduceOp.SUM,
                                     async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


class SimCLRResNet(nn.Module):

    def __init__(
            self,
            block: Callable = Bottleneck,
            layers: tuple = (3, 4, 6, 3),
            embed_dim: int = 2048,
            proj_dim: int = 128,
    ):
        super(SimCLRResNet, self).__init__()

        # Encoder
        self.encoder = ResNet(block, layers)

        # Projection head
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(
            self,
            img,
            position=True,
            shuffle=True,
            mask_ratio=0.75,
            attn_mask=False,
            attn_mask_mode='high',
            hint_ratio=0.,
            temp=0.01,
            pretrain=True,
            target_network=False,
    ):
        # img: [batch_size(*2), in_chans, height, weight]
        embed = self.encoder(img)[0]
        if not pretrain:
            return embed
        # embed: [batch_size*2, embed_dim]
        proj = self.proj_head(embed)
        # proj: [batch_size*2, proj_dim]
        if target_network:
            return proj
        else:
            return embed, torch.zeros_like(embed), None, proj


class MaskedPosReconCLRViT(nn.Module):
    """
    Masked contrastive learning Vision Transformer w/ positional reconstruction
    Default params are from ViT-Base.
    """

    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            encoder_depth: int = 12,
            encoder_num_heads: int = 12,
            decoder_depth: int = 3,
            decoder_num_heads: int = 12,
            mlp_ratio: int = 4,
            proj_dim: int = 128,
            drop_rate: float = 0.,
            attention_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            norm_layer: Callable = nn.LayerNorm,
    ):
        """
        Args:
            img_size: input image size
            patch_size: patch size
            in_chans: number of in channels
            embed_dim: embedding dimension
            encoder_depth: encoder number of Transformer blocks
            encoder_num_heads: encoder number of self-attention heads
            decoder_depth: decoder number of Transformer blocks
                           (set to 0 for linear layer)
            decoder_num_heads: decoder number of self-attention heads
            mlp_ratio: MLP dimension ratio (mlp_dim = embed_dim * mlp_ratio)
            proj_dim: projection head output dimension
            drop_rate: dropout rate
            attention_drop_rate: attention dropout rate
            drop_path_rate: stochastic depth rate
            norm_layer: normalization layer
        """
        super(MaskedPosReconCLRViT, self).__init__()

        # Encoder
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Following DeiT-3, exclude pos_embed from cls_token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                      requires_grad=False)

        self.blocks = nn.Sequential(*[Block(
            embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True,
            drop=drop_rate, attn_drop=attention_drop_rate, drop_path=dpr.item(),
            norm_layer=norm_layer
        ) for dpr in torch.linspace(0, drop_path_rate, encoder_depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Position predictor
        if decoder_depth == 0:
            self.pos_decoder = nn.Linear(embed_dim, embed_dim)
        else:
            self.pos_decoder = nn.Sequential(*[Block(
                embed_dim, decoder_num_heads, mlp_ratio,
                qkv_bias=True, norm_layer=norm_layer
            ) for _ in range(decoder_depth)
            ])

        # Projection head
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )

        self.init_weights()

    def init_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.size(-1), int(self.patch_embed.num_patches ** .5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Init weights in convolutional layers like in MLPs
        patch_conv_weight = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(patch_conv_weight.view(patch_conv_weight.size(0), -1))

        if isinstance(self.pos_decoder, nn.Conv1d):
            pos_conv_weight = self.pos_decoder.weight.data
            nn.init.xavier_uniform_(pos_conv_weight.view(pos_conv_weight.size(0), -1))

        nn.init.normal_(self.cls_token, std=.02)

        self.apply(self._init_other_weights)

    def _init_other_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def rand_shuffle(x, pos_embed):
        batch_size, seq_len, embed_dim = x.size()
        # pos_embed: [1, seq_len, embed_dim]
        batch_pos_embed = pos_embed.expand(batch_size, -1, -1)
        # batch_pos_embed: [batch_size, seq_len, embed_dim]
        noise = torch.rand(batch_size, seq_len, device=x.device)
        shuffled_indices = noise.argsort()
        # shuffled_indices: [batch_size, seq_len]
        expand_shuffled_indices = shuffled_indices.unsqueeze(-1).expand(-1, -1, embed_dim)
        # expand_shuffled_indices: [batch_size, seq_len, embed_dim]
        batch_shuffled_pos_embed = batch_pos_embed.gather(1, expand_shuffled_indices)
        # batch_shuffled_pos_embed: [batch_size, seq_len, embed_dim]
        return x + batch_shuffled_pos_embed

    @staticmethod
    def mask_interval(x, mask_ratio, left_ratio, indices):
        """
        Leave $seq_len * (1 - mask_ratio)$ elements after center masking,
        with $seq_len * left_ratio$ elements tailing in the end.
        indices: [batch_size, seq_len]
        """

        batch_size, seq_len, embed_dim = x.size()
        visible_len = int(seq_len * (1 - mask_ratio))
        invisible_len = seq_len - visible_len
        tail_len = int(seq_len * left_ratio)
        mask_start_index = visible_len - tail_len
        mask_end_index = mask_start_index + invisible_len

        visible_indices_mask = torch.ones(seq_len, dtype=torch.bool)
        visible_indices_mask[mask_start_index:mask_end_index] = False
        # visible_indices_mask: [seq_len]
        visible_indices = indices[:, visible_indices_mask]
        # visible_indices: [batch_size, seq_len * mask_ratio]
        expand_visible_indices = visible_indices.unsqueeze(-1).expand(-1, -1, embed_dim)
        # expand_visible_indices: [batch_size, seq_len * mask_ratio, embed_dim]
        x_masked = x.gather(1, expand_visible_indices)
        # x_masked: [batch_size, seq_len * mask_ratio, embed_dim]

        return x_masked, expand_visible_indices

    def first_k_mask(self, x, mask_ratio, indices):
        """
        Leave first $k = seq_len * (1 - mask_ratio)$ elements after masking
        indices: [batch_size, seq_len]
        """

        return self.mask_interval(x, mask_ratio, 0., indices)

    def rand_mask(self, x, mask_ratio):
        batch_size, seq_len, embed_dim = x.size()
        noise = torch.rand(batch_size, seq_len, device=x.device)
        shuffled_indices = noise.argsort()
        # shuffled_indices: [batch_size, seq_len]

        return self.first_k_mask(x, mask_ratio, shuffled_indices)

    @torch.no_grad()
    def pay_attention(self, x):
        batch_size, seq_len, embed_dim = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.blocks[:-1](x)
        x = self.blocks[-1].norm1(x)
        last_attention = self.blocks[-1].attn

        qkv = last_attention.qkv(x)
        qkv = qkv.reshape(batch_size, 1 + seq_len, 3, last_attention.num_heads,
                          embed_dim // last_attention.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # q, k, v: [batch_size, num_heads, 1+seq_len, embed_dim // num_heads]

        attention_weight = (q @ k.transpose(-2, -1)) * last_attention.scale
        attention_weight = attention_weight.softmax(dim=-1)
        # attention_weight: [batch_size, num_heads, 1+seq_len, 1+seq_len]

        return attention_weight

    def attn_mask(self, x, mask_ratio, hint_ratio, attn_mask_mode):
        attn_weight = self.pay_attention(x)
        # attn_weight: [batch_size, num_heads, 1+seq_len, 1+seq_len]
        cls_attn_weight = attn_weight[:, :, 0, 1:]
        # cls_attn_weight: [batch_size, num_heads, seq_len]
        cls_attn_head_avg_weight = cls_attn_weight.mean(1)
        # cls_attn_head_avg_weight: [batch_size, seq_len]
        attn_ranked_indices = cls_attn_head_avg_weight.argsort(
            descending=(attn_mask_mode == 'low')
        )

        return self.mask_interval(x, mask_ratio, hint_ratio, attn_ranked_indices)

    def forward_encoder(self, x, position, shuffle=False, mask_ratio=0.,
                        attn_mask=False, attn_mask_mode='high', hint_ratio=0.):
        x = self.patch_embed(x)

        if position:
            if shuffle:
                x = self.rand_shuffle(x, self.pos_embed)
            else:
                x += self.pos_embed
        # batch_size*2, seq_len, embed_dim

        if mask_ratio == 0:
            # make `visible_indices` empty when all patches are visible
            visible_indices = None
        else:
            if attn_mask:
                x, visible_indices = self.attn_mask(x, mask_ratio, hint_ratio,
                                                    attn_mask_mode)
            else:
                x, visible_indices = self.rand_mask(x, mask_ratio)
        # batch_size*2, seq_len * mask_ratio, embed_dim

        # Concatenate [CLS] tokens w/o pos_embed
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # batch_size*2, 1 + seq_len * mask_ratio, embed_dim

        x = self.blocks(x)
        x = self.norm(x)

        return x, visible_indices

    def pos_recon_loss(self, batch_pos_embed_pred, vis_ids):
        batch_size = batch_pos_embed_pred.size(0)
        # self.pos_embed: [1, seq_len, embed_dim]
        # batch_pos_embed_pred: [batch_size*2, seq_len, embed_dim]
        # vis_ids: [batch_size*2, seq_len * mask_ratio, embed_dim]

        batch_pos_embed_targ = self.pos_embed.expand(batch_size, -1, -1)
        # batch_pos_embed_targ: [batch_size*2, seq_len, embed_dim]

        # Only compute loss on visible patches
        if vis_ids is not None:
            batch_pos_embed_targ = batch_pos_embed_targ.gather(1, vis_ids)

        # visible_pos_embed_targ: [batch_size*2, seq_len * mask_ratio, embed_dim]
        loss = F.mse_loss(batch_pos_embed_pred, batch_pos_embed_targ)
        return loss

    def forward(
            self,
            img,
            position=True,
            shuffle=True,
            mask_ratio=0.75,
            attn_mask=False,
            attn_mask_mode='high',
            hint_ratio=0.,
            pretrain=True,
            target_network=False,
    ):
        # img: [batch_size*2, in_chans, height, weight]
        if pretrain:
            latent, vis_ids = self.forward_encoder(
                img, position, shuffle,
                mask_ratio, attn_mask, attn_mask_mode, hint_ratio,
            )
            # latent: [batch_size*2, 1 + seq_len * mask_ratio, embed_dim]

            proj = self.proj_head(latent[:, 0, :])
            # proj: [batch_size*2, proj_dim]

            if target_network:
                return proj

            pos_pred = self.pos_decoder(latent[:, 1:, :])
            # pos_pred: [batch_size*2, seq_len * mask_ratio, embed_dim]

            return latent, pos_pred, vis_ids, proj
        else:
            latent, _ = self.forward_encoder(img, position)
            # latent: [batch_size, 1 + seq_len * mask_ratio, embed_dim]

            return latent[:, 0, :]


class AttentionWithWeightSharing(nn.Module):
    def __init__(self, qkv, proj, num_heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        dim = qkv.weight.size(-1)
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = qkv
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = proj
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, weight_output=False):
        batch_size, seq_len, embed_dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads,
                          embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_weight = (q @ k.transpose(-2, -1)) * self.scale
        attn_weight = attn_weight.softmax(dim=-1)
        attn_weight = self.attn_drop(attn_weight)

        x = (attn_weight @ v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        return (x, attn_weight) if weight_output else (x, None)


class MlpWithWeightSharing(nn.Module):

    def __init__(
            self,
            fc1: nn.Linear,
            fc2: nn.Linear,
            act_layer: Callable = nn.GELU,
            drop: float = 0.,
    ):
        super().__init__()

        self.fc1 = fc1
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = fc2
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class BlockWithWeightSharing(nn.Module):

    def __init__(
            self,
            qkv: nn.Linear,
            proj: nn.Linear,
            mlp_fc1: nn.Linear,
            mlp_fc2: nn.Linear,
            num_heads: int,
            drop: float = 0.,
            attn_drop: float = 0.,
            attention_weight: bool = False,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
    ):
        super().__init__()
        self.attention_weight = attention_weight
        dim = qkv.weight.size(-1)
        self.norm1 = norm_layer(dim)
        self.attn = AttentionWithWeightSharing(qkv, proj, num_heads=num_heads,
                                               attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = MlpWithWeightSharing(mlp_fc1, mlp_fc2, act_layer, drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x, attn_weight = self.attn(self.norm1(x), self.attention_weight)
        x = x + self.drop_path1(self.ls1(x))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        # Use dynamic return type here to make sequential module easier
        return (x, attn_weight) if attn_weight is not None else x


class SimCLRViT(nn.Module):
    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: int = 4,
            proj_dim: int = 128,
            drop_rate: float = 0.,
            attention_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            weight_sharing: Optional[str] = None,
    ):
        """
        Args:
            img_size: input image size
            patch_size: patch size
            in_chans: number of in channels
            embed_dim: embedding dimension
            depth: encoder number of Transformer blocks
            num_heads: encoder number of self-attention heads
            mlp_ratio: MLP dimension ratio (mlp_dim = embed_dim * mlp_ratio)
            proj_dim: projection head output dimension
            drop_rate: dropout rate
            attention_drop_rate: attention dropout rate
            drop_path_rate: stochastic depth rate
            act_layer: activation layer
            norm_layer: normalization layer
            weight_sharing: ALBERT-like weight sharing,
                            choose from None, attn, ffn, or all
        """
        super().__init__()

        # Encoder
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Following DeiT-3, exclude pos_embed from cls_token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                      requires_grad=False)

        self.blocks = self.build_blocks(
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            drop_rate,
            attention_drop_rate,
            drop_path_rate,
            act_layer,
            norm_layer,
            weight_sharing,
        )
        self.norm = norm_layer(embed_dim)

        # Projection head
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )

        self.init_weights()

    def build_blocks(
            self,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: int = 4,
            drop_rate: float = 0.,
            attention_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            weight_sharing: Optional[str] = None,
    ) -> nn.Module:
        blocks = []
        qkv = nn.Linear(embed_dim, embed_dim * 3)
        proj = nn.Linear(embed_dim, embed_dim)
        mlp_fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        mlp_fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        for layer, dpr in enumerate(torch.linspace(0, drop_path_rate, depth)):
            block = BlockWithWeightSharing(
                qkv, proj, mlp_fc1, mlp_fc2, num_heads,
                drop=drop_rate, attn_drop=attention_drop_rate, drop_path=dpr.item(),
                act_layer=act_layer, norm_layer=norm_layer,
                attention_weight=(True if layer == depth - 1 else False),
            )
            if weight_sharing is None or weight_sharing == "attn":
                mlp_fc1 = copy.deepcopy(mlp_fc1)
                mlp_fc2 = copy.deepcopy(mlp_fc2)
            if weight_sharing is None or weight_sharing == "ffn":
                qkv = copy.deepcopy(qkv)
                proj = copy.deepcopy(proj)
            blocks.append(block)
        return nn.Sequential(*blocks)

    def init_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.size(-1), int(self.patch_embed.num_patches ** .5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Init weights in convolutional layers like in MLPs
        patch_conv_weight = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(patch_conv_weight.view(patch_conv_weight.size(0), -1))

        nn.init.normal_(self.cls_token, std=.02)

        self.apply(self._init_other_weights)

    def _init_other_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def pre_encode(self, img, position):
        x = self.patch_embed(img)
        if position:
            x += self.pos_embed
        # x: [batch_size, seq_len, embed_dim]

        return x

    def forward_encoder(self, x):
        # Concatenate [CLS] tokens w/o pos_embed
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # x: [batch_size, 1 + seq_len, embed_dim]

        x, attn_weight = self.blocks(x)
        x = self.norm(x)
        # x: [batch_size, 1 + seq_len, embed_dim]

        return x[:, 0, :], attn_weight

    def forward(self, img, position=True):
        # img: [batch_size, in_chans, height, weight]
        x = self.pre_encode(img, position)
        x, attn_weight = self.forward_encoder(x)
        proj = self.proj_head(x)
        # proj: [batch_size, proj_dim]

        return x, proj, attn_weight


class SimCLRMaskedViT(SimCLRViT):

    @staticmethod
    def mask_interval(x, mask_ratio, left_ratio, indices):
        """
        Leave $seq_len * (1 - mask_ratio)$ elements after center masking,
        with $seq_len * left_ratio$ elements tailing in the end.
        indices: [batch_size, seq_len]
        """

        batch_size, seq_len, embed_dim = x.size()
        visible_len = int(seq_len * (1 - mask_ratio))
        invisible_len = seq_len - visible_len
        tail_len = int(seq_len * left_ratio)
        mask_start_index = visible_len - tail_len
        mask_end_index = mask_start_index + invisible_len

        visible_indices_mask = torch.ones(seq_len, dtype=torch.bool)
        visible_indices_mask[mask_start_index:mask_end_index] = False
        # visible_indices_mask: [seq_len]
        visible_indices = indices[:, visible_indices_mask]
        # visible_indices: [batch_size, seq_len * mask_ratio]
        expand_visible_indices = visible_indices.unsqueeze(-1).expand(-1, -1, embed_dim)
        # expand_visible_indices: [batch_size, seq_len * mask_ratio, embed_dim]
        x_masked = x.gather(1, expand_visible_indices)
        # x_masked: [batch_size, seq_len * mask_ratio, embed_dim]

        return x_masked, expand_visible_indices

    def first_k_mask(self, x, mask_ratio, indices):
        """
        Leave first $k = seq_len * (1 - mask_ratio)$ elements after masking
        indices: [batch_size, seq_len]
        """

        return self.mask_interval(x, mask_ratio, 0., indices)

    def rand_mask(self, x, mask_ratio):
        batch_size, seq_len, embed_dim = x.size()
        noise = torch.rand(batch_size, seq_len, device=x.device)
        shuffled_indices = noise.argsort()
        # shuffled_indices: [batch_size, seq_len]

        return self.first_k_mask(x, mask_ratio, shuffled_indices)

    def pre_encode(self, img, position, mask_ratio=0.75):
        x = self.patch_embed(img)
        if position:
            x += self.pos_embed
        # x: [batch_size, seq_len, embed_dim]

        if mask_ratio > 0:
            x, _ = self.rand_mask(x, mask_ratio)
            # x: [batch_size, seq_len * (1 - mask_ratio), embed_dim]

        return x

    def forward(self, x, position=True, mask_ratio=0.):
        x = self.pre_encode(x, position, mask_ratio)
        x, attn_weight = self.forward_encoder(x)
        proj = self.proj_head(x)
        # proj: [batch_size, proj_dim]

        return x, proj, attn_weight


def info_nce_loss(feat1, feat2, temp, eps=1e-6):
    feat1 = F.normalize(feat1)
    feat2 = F.normalize(feat2)
    # feat{1,2}: [batch_size, proj_dim]
    feat = torch.stack((feat1, feat2), dim=1)
    # feat: [batch_size, 2, proj_dim]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        feat = SyncFunction.apply(feat)
    # feat: [batch_size (* world_size), 2, proj_dim]
    feat1, feat2 = feat.unbind(1)
    # feat{1,2}: [batch_size (* world_size), proj_dim]
    feat = torch.cat((feat1, feat2))
    # feat: [batch_size*2 (* world_size), proj_dim]

    # All samples, filling diagonal to remove identity similarity ((2N)^2 - 2N)
    all_sim = (feat @ feat.T).fill_diagonal_(0)
    # all_sim: [batch_size*2 (* world_size), batch_size*2 (* world_size)]
    all_ = torch.exp(all_sim / temp).sum(-1)
    # all_: [batch_size*2 (* world_size)]

    # Positive samples (2N)
    pos_sim = (feat1 * feat2).sum(-1)
    # pos_sim: [batch_size (* world_size)]
    pos = torch.exp(pos_sim / temp)
    # Following all samples, compute positive similarity twice
    pos = torch.cat((pos, pos))
    # pos: [batch_size*2 (* world_size)]

    loss = -torch.log(pos / (all_ + eps)).mean()

    return loss


def cov_reg_loss(proj, norm=False):
    _, proj_dim = proj.size()
    # proj: [batch_size, proj_dim]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        proj = SyncFunction.apply(proj)
    # proj: [batch_size (* world_size), proj_dim]
    proj_cov = proj.T.corrcoef() if norm else proj.T.cov()
    off_diag_mask = ~torch.eye(
        proj_dim, dtype=torch.bool, device=proj.device
    )
    proj_cov_off_diag = proj_cov.masked_select(off_diag_mask)

    return (proj_cov_off_diag ** 2).sum() / proj_dim
