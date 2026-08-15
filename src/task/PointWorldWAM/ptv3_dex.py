from __future__ import annotations

import torch
import torch.nn as nn

from ptv3.module import PointModule
from ptv3.ptv3 import PointTransformerV3, RPE, SerializedAttention


class DexSerializedAttention(SerializedAttention):
    """PointWorld PTv3 的 task-local non-flash RPE attention。"""

    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
    ) -> None:
        # 上游 SerializedAttention 在 enable_rpe=True 时会 fail-fast；这里保留其
        # padding/relative-position helpers，只替换初始化和 attention forward。
        PointModule.__init__(self)
        if channels % num_heads:
            raise ValueError(f"channels={channels} 不能整除 num_heads={num_heads}")
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.enable_rpe = enable_rpe
        self.patch_size_max = patch_size
        self.patch_size = 0
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.rpe = RPE(patch_size, num_heads) if enable_rpe else None

    def forward(self, point):
        self.patch_size = min(
            int(torch.diff(nn.functional.pad(point.offset, (1, 0))).min()),
            self.patch_size_max,
        )
        heads = self.num_heads
        patch_size = self.patch_size
        channels = self.channels
        pad, unpad, _ = self.get_padding_and_inverse(point)
        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]
        qkv = self.qkv(point.feat)[order]
        q, k, v = (
            qkv.reshape(-1, patch_size, 3, heads, channels // heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(dim=0)
        )
        attention = (q * self.scale) @ k.transpose(-2, -1)
        if self.enable_rpe:
            attention = attention + self.rpe(self.get_rel_pos(point, order))
        attention = self.attn_drop(self.softmax(attention)).to(qkv.dtype)
        feat = (attention @ v).transpose(1, 2).reshape(-1, channels)
        point.feat = self.proj_drop(self.proj(feat[inverse]))
        return point


def build_dex_ptv3(channels, args):
    """保持 PointWorld architecture blueprint/key，只开放 dex geometry 参数。"""
    import pointworld.base as pw_base

    cfg = pw_base._resolve_arch_config(args.ptv3_size)
    if "channels_max" in cfg and channels > int(cfg["channels_max"]):
        raise ValueError(f"channels={channels} 超过 {args.ptv3_size} 上限 {cfg['channels_max']}")
    if "channels_eq" in cfg and channels != int(cfg["channels_eq"]):
        raise ValueError(f"channels={channels} 必须等于 {cfg['channels_eq']}")
    enc_depths = tuple(pw_base._require_list(cfg, "enc_depths"))
    dec_depths = tuple(pw_base._require_list(cfg, "dec_depths"))
    enc_channels = pw_base._resolve_channels(pw_base._require_list(cfg, "enc_channels"), channels)
    dec_channels = pw_base._resolve_channels(pw_base._require_list(cfg, "dec_channels"), channels)
    enc_num_head = tuple(int(value) for value in pw_base._require_list(cfg, "enc_num_head"))
    dec_num_head = tuple(int(value) for value in pw_base._require_list(cfg, "dec_num_head"))
    stride = pw_base._resolve_stride(cfg.get("stride", "auto"), enc_depths)
    enc_patch_size = pw_base._resolve_patch_size(
        cfg.get("enc_patch_size", "auto"), len(enc_depths), args.ptv3_patch_size
    )
    dec_patch_size = pw_base._resolve_patch_size(
        cfg.get("dec_patch_size", "auto"), len(dec_depths), args.ptv3_patch_size
    )
    return PointTransformerV3(
        in_channels=channels,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=stride,
        enc_depths=enc_depths,
        enc_channels=enc_channels,
        enc_num_head=enc_num_head,
        enc_patch_size=enc_patch_size,
        dec_depths=dec_depths,
        dec_channels=dec_channels,
        dec_num_head=dec_num_head,
        dec_patch_size=dec_patch_size,
        mlp_ratio=4,
        qkv_bias=True,
        attn_drop=0,
        proj_drop=0,
        drop_path=args.drop_path,
        pre_norm=True,
        shuffle_orders=args.shuffle_orders,
        enable_rpe=args.enable_rpe,
        traceable=True,
        mask_token=False,
        freeze_encoder=False,
        enc_mode=False,
        attention_cls=DexSerializedAttention,
        attention_kwargs={},
    )
