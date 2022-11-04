import copy
from argparse import ArgumentParser, BooleanOptionalAction
from functools import partial

import torch
import torch.nn.functional as F
import torch_optimizer as optim
from pl_bolts.callbacks.knn_online import KNNOnlineEvaluator
from pl_bolts.datamodules import ImagenetDataModule
from pl_bolts.optimizers import linear_warmup_decay
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torchmetrics import Accuracy

from models import SimCLRViT, SyncFunction
from utils.datamodules import FewShotImagenetDataModule
from utils.lr_wt_decay import param_groups_lrd, exclude_from_wt_decay
from utils.transforms import SimCLRPretrainPostTransform, imagenet_normalization, SimCLRPretrainPreTransform


class MultiHeadAttnMaskCLR(LightningModule):
    def __init__(
            self,
            gpus: int,
            num_samples: int,
            batch_size: int,
            num_nodes: int = 1,
            dataset: str = "imagenet",
            num_classes: int = 1000,
            gaussian_blur: bool = True,
            jitter_strength: float = 1.,
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
            warmup_epochs: int = 10,
            max_epochs: int = 100,
            position: bool = True,
            attn_mask: bool = True,
            local_mask_ratio: float = 0.75,
            global_mask_ratio: float = 0.75,
            instance_temperature: float = 0.1,
            batch_temperature: float = 0.1,
            feat_align_temperature: float = 0.1,
            loss_ratio: float = 1.,
            optimizer: str = "adam",
            exclude_bn_bias: bool = False,
            learning_rate: float = 1e-3,
            ema_momentum: float = 0.99,
            weight_decay: float = 1e-6,
            layer_decay: float = 1.,
            **kwargs
    ):
        super(MultiHeadAttnMaskCLR, self).__init__()
        self.save_hyperparameters()

        self.gpus = gpus
        self.num_nodes = num_nodes
        self.dataset = dataset
        self.num_classes = num_classes
        self.gaussian_blur = gaussian_blur
        self.jitter_strength = jitter_strength
        self.num_samples = num_samples
        self.batch_size = batch_size

        # ViT params
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.proj_dim = proj_dim
        self.drop_rate = drop_rate
        self.attention_drop_rate = attention_drop_rate
        self.drop_path_rate = drop_path_rate

        self.optim = optimizer
        self.exclude_bn_bias = exclude_bn_bias
        self.weight_decay = weight_decay
        self.layer_decay = layer_decay
        self.position = position
        self.attn_mask = attn_mask
        self.local_mask_ratio = local_mask_ratio
        self.global_mask_ratio = global_mask_ratio
        self.instance_temperature = instance_temperature
        self.batch_temperature = batch_temperature
        self.feat_align_temperature = feat_align_temperature
        self.loss_ratio = loss_ratio

        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.ema_momentum = ema_momentum

        if dataset == "imagenet":
            normalization = imagenet_normalization()
        self.normalization = normalization
        self.transform = SimCLRPretrainPostTransform(
            img_size=img_size,
            gaussian_blur=gaussian_blur,
            jitter_strength=jitter_strength,
            normalize=normalization,
        )

        self.online_net = SimCLRViT(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            proj_dim,
            drop_rate,
            attention_drop_rate,
            drop_path_rate,
            partial(nn.LayerNorm, eps=1e-6),
        )
        self.target_net = copy.deepcopy(self.online_net)
        self.classifier = nn.Linear(embed_dim, num_classes)

        self.train_acc_top_1 = Accuracy(top_k=1)
        self.train_acc_top_5 = Accuracy(top_k=5)
        self.val_acc_top_1 = Accuracy(top_k=1, compute_on_step=False)
        self.val_acc_top_5 = Accuracy(top_k=5, compute_on_step=False)

        # compute iters per epoch
        global_batch_size = num_nodes * gpus * batch_size if gpus > 0 else batch_size
        self.train_iters_per_epoch = num_samples // global_batch_size

    def on_train_batch_end(self, outputs, batch, batch_idx):
        for online_p, target_p in zip(self.online_net.parameters(),
                                      self.target_net.parameters()):
            em = self.ema_momentum
            target_p.data = target_p.data * em + online_p.data * (1.0 - em)

    def forward(self, x, position=True):
        x = self.normalization(x)
        x, *_ = self.online_net(x, position)

        return x

    @staticmethod
    def mask_first_k(patch_embed, mask_ratio, ranked_indices):
        """
        Leave first $k$ elements after masking, where $k = seq_len * (1 - mask_ratio)$
        """
        _, seq_len, embed_dim = patch_embed.size()
        visible_len = int(seq_len * (1 - mask_ratio))
        visible_indices = ranked_indices[:, :visible_len]
        # visible_indices: [2 * num_masks * batch_size, seq_len * (1 - mask_ratio)]
        expand_visible_indices = visible_indices.unsqueeze(-1).expand(-1, -1, embed_dim)
        # expand_visible_indices: [2 * num_masks * batch_size, seq_len * (1 - mask_ratio), embed_dim]
        masked_patch_embed = patch_embed.gather(1, expand_visible_indices)
        # masked_patch_embed: [2 * num_masks * batch_size, seq_len * (1 - mask_ratio), embed_dim]
        return masked_patch_embed

    def multi_rand_mask(self, patch_embed, mask_ratio):
        effective_batch_size, seq_len, embed_dim = patch_embed.size()
        noise = torch.rand(effective_batch_size, seq_len, device=patch_embed.device)
        shuffled_indices = noise.argsort()

        return self.mask_first_k(patch_embed, mask_ratio, shuffled_indices)

    def multi_head_attn_mask(
            self,
            patch_embed,
            attn_weight,
            local_mask_ratio,
            global_mask_ratio,
    ):
        """
        Multi-head attention-guided masking
        Args:
            patch_embed: [2 * num_masks * batch_size, seq_len, embed_dim]
            attn_weight: [2 * batch_size, num_heads, 1 + seq_len, 1 + seq_len]
            local_mask_ratio: ratio of # of masked patches to # of patches
            global_mask_ratio: ratio of # of masked patches to # of patches
        return:
            masked_patch_embed: [2 * num_masks * batch_size,
                                 seq_len * (1 - mask_ratio), embed_dim]
        """
        _, seq_len, embed_dim = patch_embed.size()

        cls_attn_weight = attn_weight[:, :, 0, 1:]
        # cls_attn_weight: [2 * batch_size, num_heads, seq_len]
        cls_attn_head_avg_weight = cls_attn_weight.mean(1, keepdim=True)
        # cls_attn_head_avg_weight: [2 * batch_size, 1, seq_len]
        cls_attn_weight = torch.cat((
            cls_attn_weight, cls_attn_head_avg_weight
        ), dim=1)
        cls_attn_weight = cls_attn_weight.view(2, -1, *cls_attn_weight.shape[1:])
        cls_attn_weight = cls_attn_weight.transpose(1, 2)
        # cls_attn_weight: [2, num_masks, batch_size, seq_len]
        attn_ranked_indices = cls_attn_weight.argsort(descending=True)

        if local_mask_ratio == global_mask_ratio:
            return self.mask_first_k(
                patch_embed, local_mask_ratio, attn_ranked_indices.reshape(-1, seq_len)
            )
        else:
            patch_embed = patch_embed.view(*cls_attn_weight.shape[:-1], seq_len, embed_dim)
            local_masked_patch_embed = self.mask_first_k(
                patch_embed[:, :-1].reshape(-1, seq_len, embed_dim),
                local_mask_ratio, attn_ranked_indices[:, :-1].reshape(-1, seq_len)
            )
            global_masked_patch_embed = self.mask_first_k(
                patch_embed[:, -1].reshape(-1, seq_len, embed_dim),
                global_mask_ratio, attn_ranked_indices[:, -1].view(-1, seq_len)
            )
            return local_masked_patch_embed, global_masked_patch_embed

    @staticmethod
    def instance_contrast_loss(proj_global, proj_local, temp):
        # Instance-level contrast (cross views only)
        # positive pairs: mean attention masked (crop 1 & 2) and target (crop 1 & 2)
        proj_global = proj_global.transpose(0, 1).contiguous()
        proj_local = proj_local.transpose(0, 1).contiguous()
        batch_size, _, proj_dim = proj_global.size()
        # proj_local: [batch_size, 2 * num_heads, proj_dim]
        # proj_global: [batch_size, 4, proj_dim]

        pos_sim = proj_global @ proj_global.transpose(1, 2)
        pos_sim = pos_sim.masked_select(~torch.eye(
            4, dtype=torch.bool, device=proj_global.device
        )).view(batch_size, 4, -1)
        # pos_sim: [batch_size, 4, 3]
        pos = torch.exp(pos_sim / temp)

        neg_sim = proj_global @ proj_local.transpose(1, 2)
        # neg_sim: [batch_size, 4, 2 * num_heads]
        neg = torch.exp(neg_sim / temp).sum(-1)

        all_ = pos + neg.unsqueeze(-1)

        loss = -torch.log(pos / all_).mean()

        return loss

    @staticmethod
    def local_feature_alignment(proj1, proj2, temp):
        # Contrastive feature alignment between local features from 2 crops
        proj1 = proj1.transpose(0, 1).contiguous()
        proj2 = proj2.transpose(0, 1).contiguous()
        # proj{1,2}: [batch_size, num_heads, proj_dim]
        proj_local = torch.cat((proj1, proj2), dim=1)
        # proj_local: [batch_size, num_heads * 2, proj_dim]

        all_sim = proj_local @ proj_local.transpose(1, 2)
        all_sim.diagonal(dim1=-2, dim2=-1).fill_(0)
        all_ = torch.exp(all_sim / temp).sum(-1)

        pos_sim = (proj1 * proj2).sum(-1).repeat(1, 2)
        pos = torch.exp(pos_sim / temp)

        loss = -torch.log(pos / all_).mean()

        return loss

    @staticmethod
    def batch_contrast_loss(proj, temp):
        # Batch-level contrast (all views), similar to InfoNCE but with more positive pairs
        # positive pairs: all instance-level pairs = 2 * (num_heads+2)^2 * batch_size
        # swap batch size to the first, make it easier for all_gather
        proj = proj.transpose(0, 1).contiguous()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            proj = SyncFunction.apply(proj)
        # proj: [batch_size (* world_size), num_crops + (num_crops * num_masks), proj_dim]
        batch_size, num_pos_views, proj_dim = proj.size()

        pos_sim = proj @ proj.transpose(1, 2)
        pos_sim = pos_sim.masked_select(~torch.eye(
            num_pos_views, dtype=torch.bool, device=proj.device
        ))
        pos_sim = pos_sim.view(batch_size * num_pos_views, -1)
        pos = torch.exp(pos_sim / temp)

        neg_sim = []
        for pos_idx in torch.eye(batch_size, dtype=torch.bool):
            neg_sim.append(proj[pos_idx].squeeze(0)
                           @ proj[~pos_idx].view(-1, proj_dim).T)
        neg_sim = torch.stack(neg_sim).view(batch_size * num_pos_views, -1)
        neg = torch.exp(neg_sim / temp).sum(-1)

        all_ = pos + neg.unsqueeze(-1)

        loss = -torch.log(pos / all_).mean()

        return loss

    def shared_step(self, img):
        crop1, crop2 = img
        batch_size, *img_dim = crop1.size()
        crops = torch.stack((crop1, crop2))
        # crops: [2, batch_size, in_chans, height, width]
        num_masks = self.num_heads + 1  # Last one for attention mean
        #                                 another one for target
        crops = crops.unsqueeze(1).expand(-1, num_masks + 1, -1, *img_dim)
        # crops: [2, num_masks+1, batch_size, in_chans, height, width]
        img = self.transform(crops.reshape(-1, *img_dim)).view_as(crops)
        # img: [2, num_masks+1, batch_size, in_chans, height, width]

        # To fetch attention weight, forward target net first
        img_target = img[:, 0].reshape(-1, *img_dim)
        # img_online: [2 * batch_size, in_chans, height, width]
        with torch.no_grad():
            _, proj_target_, attn_weight = self.target_net(img_target, self.position)

        # To mask on embedding-level, manually forward online encoder
        img_online = img[:, 1:].reshape(-1, *img_dim)
        # img_online: [2 * num_masks * batch_size, in_chans, height, width]
        patch_embed = self.online_net.pre_encode(img_online, self.position)
        masked_patch_embed = self.multi_head_attn_mask(
            patch_embed, attn_weight, self.local_mask_ratio, self.global_mask_ratio,
        ) if self.attn_mask else self.multi_rand_mask(
            patch_embed, self.local_mask_ratio
        )
        if isinstance(masked_patch_embed, torch.Tensor):
            reps, _ = self.online_net.forward_encoder(masked_patch_embed)
        else:
            reps = torch.cat([
                self.online_net.forward_encoder(mpe)[0].view(2, -1, batch_size, self.embed_dim)
                for mpe in masked_patch_embed
            ], dim=1).view(-1, self.embed_dim)
        proj_online_ = self.online_net.proj_head(reps)

        # Normalize and reshape for loss calculation
        proj_target = F.normalize(proj_target_).view(-1, batch_size, self.proj_dim)
        proj_online = F.normalize(proj_online_).view(-1, batch_size, self.proj_dim)

        # Instance-level loss, cross-view contrast
        proj1_online, proj2_online = proj_online.chunk(2)
        proj1_target, proj2_target = proj_target.chunk(2)
        proj1_global, proj2_global = proj1_online[-1:], proj2_online[-1:]
        proj1_local, proj2_local = proj1_online[:-1], proj2_online[:-1]
        proj_global = torch.cat((proj1_global, proj1_target, proj2_global, proj2_target))
        proj_local = torch.cat((proj1_local, proj2_local))
        loss_instance_clr = self.instance_contrast_loss(
            proj_global, proj_local, self.instance_temperature
        )

        # Batch-level loss, all-view contrast
        proj = torch.cat((proj_online, proj_target))
        loss_batch_clr = self.batch_contrast_loss(proj, self.batch_temperature)

        # Contrastive local feature alignment, prevent local projection collapse
        loss_loc_feat_align = self.local_feature_alignment(
            proj1_local, proj2_local, self.feat_align_temperature
        )

        loss_total = (self.loss_ratio * loss_instance_clr
                      + (1 - self.loss_ratio) * loss_batch_clr
                      + loss_loc_feat_align)

        return (reps,
                (proj_online_, proj_target_),
                (loss_instance_clr, loss_batch_clr, loss_loc_feat_align, loss_total))

    def linear_probe(self, reps, labels, acc_top_1_fn, acc_top_5_fn):
        logits = self.classifier(reps)
        loss = F.cross_entropy(logits, labels)
        acc_top_1 = acc_top_1_fn(logits.softmax(-1), labels)
        acc_top_5 = acc_top_5_fn(logits.softmax(-1), labels)

        return loss, acc_top_1, acc_top_5

    def training_step(self, batch, batch_idx):
        img, labels = batch
        reps, proj_embed, losses = self.shared_step(img)

        num_crops = len(img)
        num_masks = self.num_heads + 1
        label_expand = labels.unsqueeze(0).expand(num_crops * num_masks, -1).reshape(-1)
        loss_xent, acc_top_1, acc_top_5 = self.linear_probe(
            reps.detach(), label_expand, self.train_acc_top_1, self.train_acc_top_5
        )

        self.log_dict({
            'loss/instance/train': losses[0],
            'loss/batch/train': losses[1],
            'loss/feat_align/train': losses[2],
            'loss/pretrain/train': losses[-1],
            'loss/linear_probe/train': loss_xent,
            'acc/linear_probe_top_1/train': acc_top_1,
            'acc/linear_probe_top_5/train': acc_top_5,
            'norm/reps': reps.norm(dim=-1).mean(),
            'norm/proj_online': proj_embed[0].norm(dim=-1).mean(),
            'norm/proj_target': proj_embed[1].norm(dim=-1).mean(),
        }, sync_dist=True)
        return losses[-1] + loss_xent

    def validation_step(self, batch, batch_idx):
        img, labels = batch
        reps, _, losses = self.shared_step(img)

        num_crops = len(img)
        num_masks = self.num_heads + 1
        label_expand = labels.unsqueeze(0).expand(num_crops * num_masks, -1).reshape(-1)
        loss_xent, acc_top_1, acc_top_5 = self.linear_probe(
            reps.detach(), label_expand, self.val_acc_top_1, self.val_acc_top_5
        )

        self.log_dict({
            'loss/instance/val': losses[0],
            'loss/batch/val': losses[1],
            'loss/feat_align/val': losses[2],
            'loss/pretrain/val': losses[-1],
            'loss/linear_probe/val': loss_xent,
            'acc/linear_probe_top_1/val': acc_top_1,
            'acc/linear_probe_top_5/val': acc_top_5,
        }, sync_dist=True)
        return losses[-1] + loss_xent

    def configure_optimizers(self):
        param_groups = param_groups_lrd(
            self.online_net,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            exclude_1d_params=self.exclude_bn_bias,
            no_weight_decay_list=("pos_embed", "cls_token"),
            layer_decay=self.layer_decay
        )
        if self.exclude_bn_bias:
            param_groups += exclude_from_wt_decay(self.classifier, self.weight_decay)
        else:
            param_groups += [
                {"params": self.classifier.parameters(), "weight_decay": self.weight_decay}
            ]

        if self.optim == "lars":
            optimizer = optim.LARS(param_groups, lr=self.learning_rate, momentum=0.9)
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(param_groups, lr=self.learning_rate)
        elif self.optim == "lamb":
            optimizer = optim.Lamb(param_groups, lr=self.learning_rate)
        elif self.optim == "adamw":
            optimizer = torch.optim.AdamW(param_groups, lr=self.learning_rate)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # model params
        parser.add_argument("--img_size", default=224, type=int,
                            help="input image size")
        parser.add_argument("--patch_size", default=16, type=int,
                            help="patch size")
        parser.add_argument("--in_chans", default=3, type=int,
                            help="number of in channels")
        parser.add_argument("--embed_dim", default=768, type=int,
                            help="embedding dimension")
        parser.add_argument("--depth", default=12, type=int,
                            help="number of Transformer blocks")
        parser.add_argument("--num_heads", default=12, type=int,
                            help="number of self-attention heads")
        parser.add_argument("--mlp_ratio", default=4, type=int,
                            help="Ratio of embedding dim to MLP dim")
        parser.add_argument("--proj_dim", default=128, type=int,
                            help="projection head output dimension")
        parser.add_argument("--mlp_dropout", default=0.0, type=float,
                            help="mlp dropout rate")
        parser.add_argument("--attention_dropout", default=0.0, type=float,
                            help="attention dropout rate")
        parser.add_argument("--path_dropout", default=0.0, type=float,
                            help="path dropout rate")
        parser.add_argument("--fp32", default=True, action=BooleanOptionalAction,
                            help="use fp32 or fp16")

        # transform params
        parser.add_argument("--dataset", type=str, default="imagenet",
                            help="dataset")
        parser.add_argument("--data_dir", type=str, default="dataset",
                            help="path to dataset")
        parser.add_argument("--sample_pct", type=int, default=100,
                            help="%% of samples for training (only for ablation)")
        parser.add_argument("--gaussian_blur", default=True,
                            action=BooleanOptionalAction, help="add gaussian blur")
        parser.add_argument("--jitter_strength", type=float, default=1.0,
                            help="jitter strength")

        # training params
        parser.add_argument("--fast_dev_run", default=False, type=int)
        parser.add_argument("--num_nodes", default=1, type=int,
                            help="number of nodes for training")
        parser.add_argument("--gpus", default=1, type=int,
                            help="number of gpus to train on")
        parser.add_argument("--num_workers", default=8, type=int,
                            help="num of workers per GPU")
        parser.add_argument("--optimizer", default="adam", type=str,
                            help="choose between adam/lars")
        parser.add_argument("--exclude_bn_bias", default=True,
                            action=BooleanOptionalAction,
                            help="exclude bn/ln/bias from weight decay")
        parser.add_argument("--max_epochs", default=100, type=int,
                            help="number of total epochs to run")
        parser.add_argument("--max_steps", default=-1, type=int,
                            help="max steps")
        parser.add_argument("--warmup_epochs", default=10, type=int,
                            help="number of warmup epochs")
        parser.add_argument("--ema_momentum", default=0.99, type=float,
                            help="ema momentum")
        parser.add_argument("--batch_size", default=128, type=int,
                            help="batch size per gpu")

        parser.add_argument("--position", default=True, action=BooleanOptionalAction,
                            help="add positional embedding or not")
        parser.add_argument("--attn_mask", default=True, action=BooleanOptionalAction,
                            help="attention-guide masking or random masking")
        parser.add_argument("--local_mask_ratio", default=0.75, type=float,
                            help="mask ratio of patches for local features")
        parser.add_argument("--global_mask_ratio", default=0.75, type=float,
                            help="mask ratio of patches for global features")
        parser.add_argument("--instance_temperature", default=0.1, type=float,
                            help="instance-level temperature in loss")
        parser.add_argument("--batch_temperature", default=0.1, type=float,
                            help="batch-level temperature in loss")
        parser.add_argument("--feat_align_temperature", default=0.1, type=float,
                            help="local feature alignment temperature in loss")
        parser.add_argument("--loss_ratio", default=.5, type=float,
                            help="weight of two losses")
        parser.add_argument("--weight_decay", default=1e-6, type=float,
                            help="weight decay")
        parser.add_argument("--layer_decay", default=1., type=float,
                            help="layer-wise decay")
        parser.add_argument("--learning_rate", default=1e-3, type=float,
                            help="base learning rate")

        return parser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--version", default=None, type=str)
    parser.add_argument("--log_path", default="lightning_logs", type=str)
    parser.add_argument("--resume_ckpt_path", default=None, type=str)
    parser.add_argument("--track_grad", default=True, action=BooleanOptionalAction)
    parser.add_argument("--knn_probe", default=False, action='store_true')
    parser = MultiHeadAttnMaskCLR.add_model_specific_args(parser)
    args = parser.parse_args()

    if args.dataset == "imagenet":
        if args.sample_pct < 100:
            dm = FewShotImagenetDataModule(args.data_dir,
                                           label_pct=args.sample_pct,
                                           batch_size=args.batch_size,
                                           num_workers=args.num_workers)
        else:
            dm = ImagenetDataModule(data_dir=args.data_dir,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers)
        args.num_samples = dm.num_samples
    else:
        raise NotImplementedError(f"Unimplemented dataset: {args.dataset}")

    dm.train_transforms = dm.val_transforms = SimCLRPretrainPreTransform(args.img_size)

    model = MultiHeadAttnMaskCLR(**args.__dict__)

    logger = TensorBoardLogger(args.log_path, name="pretrain", version=args.version)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(save_last=True, monitor="loss/pretrain/val")

    callbacks = [model_checkpoint]
    if args.knn_probe:
        callbacks.append(KNNOnlineEvaluator())
    callbacks.append(lr_monitor)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        devices=args.gpus if args.gpus > 0 else None,
        num_nodes=args.num_nodes,
        accelerator="gpu" if args.gpus > 0 else None,
        strategy="ddp_find_unused_parameters_false" if args.gpus > 1 else None,
        sync_batchnorm=True if args.gpus > 1 else False,
        track_grad_norm=2 if args.track_grad else -1,
        precision=32 if args.fp32 else 16,
        callbacks=callbacks,
        logger=logger,
        fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(model, datamodule=dm, ckpt_path=args.resume_ckpt_path)
