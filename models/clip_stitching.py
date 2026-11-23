"""OV-Stitcher's inference-time adapter for the public OpenCLIP ViT.

The CLIP checkpoint, tokenizer, text encoder, and model layers come from the
upstream ``open_clip_torch`` package. This module contains only the author's
high-resolution image path: crop stitching, region constrained attention, and
the spatial/semantic branches used by OV-Stitcher.
"""

import math

import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from torch import nn

from .features import make_sliding_windows, stitch_features


class StitchedCLIP(nn.Module):
    """Keep public OpenCLIP weights while adding OV-Stitcher's image path."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def encode_text(self, tokens):
        """Delegate text encoding unchanged to public OpenCLIP."""
        return self.model.encode_text(tokens)

    def encode_image(self, image_shape, crops, dino_features, dino_patch_size,
                     feature_shape, instance_masks, coords, paddings,
                     slide_stride, slide_crop):
        """Encode crops using OV-Stitcher's stitched CLIP visual path.

        Args:
            image_shape: Full preprocessed image shape, ``[1, 3, H, W]``.
            crops: CLIP normalized crops, ``[Ncrop, 3, Hcrop, Wcrop]``.
            dino_features: Stitched DINO tokens, ``[1, Hd*Wd, Cd]``.
            feature_shape: DINO grid ``(Hd, Wd)``.
            instance_masks: Resized region IDs, ``[1, 1, H, W]``.

        Returns:
            Unnormalized CLIP patch features, ``[1, Hd*Wd, Cclip]``.
        """
        return encode_stitched_visual(
            self.model.visual,
            image_shape=image_shape,
            crops=crops,
            dino_features=dino_features,
            dino_patch_size=dino_patch_size,
            feature_shape=feature_shape,
            instance_masks=instance_masks,
            coords=coords,
            paddings=paddings,
            slide_stride=slide_stride,
            slide_crop=slide_crop,
        )


def _position_embedding(visual, tokens, height, width):
    """Resize ViT's learned 2D position grid to the current crop grid."""
    patch_count = tokens.shape[1] - 1
    base_count = visual.positional_embedding.shape[0] - 1
    if patch_count == base_count and height == width:
        return visual.positional_embedding

    # The released ViT position embeddings are a square grid plus one CLS row.
    base_side = math.sqrt(base_count)
    if not base_side.is_integer():
        raise ValueError('Expected a square OpenCLIP position embedding grid')
    base_side = int(base_side)

    patch_height = height // visual.patch_size[0]
    patch_width = width // visual.patch_size[1]
    class_position = visual.positional_embedding[:1].unsqueeze(0)
    patch_position = visual.positional_embedding[1:]
    channels = patch_position.shape[-1]

    # Bicubic interpolation lets the pretrained 14x14 grid serve larger crops.
    patch_position = patch_position.reshape(1, base_side, base_side, channels)
    patch_position = patch_position.permute(0, 3, 1, 2)
    patch_position = F.interpolate(
        patch_position,
        scale_factor=((patch_height + 0.1) / base_side,
                      (patch_width + 0.1) / base_side),
        mode='bicubic',
    )
    if patch_position.shape[-2:] != (patch_height, patch_width):
        raise ValueError(
            'Interpolated position grid does not match crop patch grid: '
            f'{patch_position.shape[-2:]} != {(patch_height, patch_width)}')
    patch_position = patch_position.permute(0, 2, 3, 1).reshape(
        1, patch_height * patch_width, channels)
    return torch.cat((class_position, patch_position), dim=1)


def _embed_crops(visual, crops):
    """Convert pixel crops ``[Ncrop,3,H,W]`` to CLIP tokens ``[L,Ncrop,C]``."""
    num_crops, _, height, width = crops.shape

    # Conv patchification changes RGB pixels into one C-dimensional vector per
    # 16x16 patch: [Ncrop, 3, H, W] -> [Ncrop, C, H/16, W/16].
    patch_map = visual.conv1(crops)
    tokens = patch_map.flatten(2).transpose(1, 2)  # [Ncrop, Hc*Wc, C]
    class_token = visual.class_embedding.view(1, 1, -1).expand(
        num_crops, 1, -1).to(tokens.dtype)
    tokens = torch.cat((class_token, tokens), dim=1)  # add one CLS token

    # Position embeddings encode patch coordinates; resize them for 336px crops.
    tokens = tokens + _position_embedding(visual, tokens, height, width).to(
        tokens.dtype)
    tokens = visual.ln_pre(visual.patch_dropout(tokens))
    return tokens.transpose(0, 1), (height // visual.patch_size[0],
                                    width // visual.patch_size[1])


def _stitch_crop_tokens(tokens, image_shape, coords, paddings, patch_size):
    """Average crop-token overlaps onto one full-image patch grid.

    ``tokens`` is batch first, ``[Ncrop, Lcrop, C]``; the result is
    ``[1, Hgrid*Wgrid, C]`` plus ``(Hgrid, Wgrid)``.
    """
    return stitch_features(tokens, coords, paddings, patch_size, image_shape)


def _merge_similar_regions(dino_map, masks):
    """Merge highly similar foreground mask IDs before making attention scope.

    ``dino_map`` is ``[1, Hd, Wd, Cd]`` and ``masks`` is ``[1, Hd, Wd]``.
    Background ID 0 remains separate. DBSCAN labels are shifted to fresh IDs
    so existing mask IDs cannot collide with merged groups.
    """
    mask_ids = torch.unique(masks[0])
    if mask_ids.numel() <= 2:
        return masks

    regions = [masks[0] == mask_id for mask_id in mask_ids]
    region_features = torch.stack([
        dino_map[0][region].mean(dim=0) for region in regions
    ])
    region_features = F.normalize(region_features, dim=1)[1:]  # skip bg
    similarities = region_features @ region_features.T

    # DBSCAN expects distances, so convert cosine similarity to a distance
    # matrix. One sample per cluster allows pairs within EPS to merge.
    distances = (similarities.max() - similarities).cpu().numpy()
    cluster_ids = DBSCAN(metric='precomputed', eps=0.2,
                         min_samples=1).fit_predict(distances)
    for region, cluster_id in zip(regions[1:], cluster_ids):
        if cluster_id >= 0:
            masks[0][region] = int(cluster_id) + 10000
    return masks


def _dino_scope_attention(dino_features, feature_shape, instance_masks):
    """Build global DINO similarity attention restricted by mask regions.

    Returns a softmax matrix ``[1, Hd*Wd, Hd*Wd]`` and the mask grid
    ``[1, Hd, Wd]``. Similarity supplies affinity; mask IDs limit which pairs
    may contribute, preserving region boundaries while retaining global
    context for background patches.
    """
    height, width = feature_shape
    dino_map = dino_features.reshape(1, height, width, -1)
    similarity = torch.bmm(dino_features, dino_features.transpose(1, 2))
    masks = F.interpolate(instance_masks.float(), size=feature_shape,
                          mode='nearest').squeeze(1).long()
    masks = _merge_similar_regions(dino_map, masks)

    region_vectors = [
        (masks[0] == mask_id).flatten()
        for mask_id in torch.unique(masks[0])
    ]
    allowed = torch.zeros_like(similarity[0], dtype=torch.bool)
    for region in region_vectors[1:]:
        allowed |= torch.outer(region, region)

    # Let background connect to visually similar patches in other regions;
    # this avoids isolating background while limiting arbitrary cross-talk.
    background = region_vectors[0]
    background_pairs = background[:, None] | background[None, :]
    allowed |= background_pairs & (similarity[0] > similarity[0].mean())

    similarity = similarity.masked_fill(~allowed.unsqueeze(0), -torch.inf)
    return (similarity * 4.0).softmax(dim=-1), masks


def _crop_region_masks(dino_grid_masks, mask_ids, image_shape, stride, crop_size,
                       crop_token_shape):
    """Make one boolean region map per crop and mask ID.

    Output shape is ``[Ncrop, Nregion, Hclip, Wclip]``. Nearest-neighbor
    resizing keeps integer region IDs intact as masks move between pixel and
    patch grids.
    """
    height, width = image_shape[-2:]
    # Use merged DINO-grid IDs so DBSCAN merges also reach this branch.
    # Nearest-neighbor resizing preserves the discrete integer region IDs.
    full_size_masks = F.interpolate(dino_grid_masks.unsqueeze(1).float(),
                                    size=(height, width), mode='nearest')
    crop_masks, _, _ = make_sliding_windows(full_size_masks, stride, crop_size)
    crop_masks = F.interpolate(crop_masks.float(), size=crop_token_shape,
                               mode='nearest').squeeze(1).long()
    return torch.stack([crop_masks == mask_id for mask_id in mask_ids], dim=1)


def _make_crop_attention_mask(region_masks):
    """Create per-crop CLIP masks for region tokens plus image tokens.

    ``region_masks``: ``[Ncrop, Nregion, Hclip, Wclip]``. The returned
    legacy boolean mask is ``[Ncrop, S, S]``. Upstream v2.24 converts it to
    the feature dtype before MultiheadAttention, so True becomes an additive
    +1 logit bias rather than a hard block. Keep the author's original
    polarity here to preserve its behavior.
    """
    num_crops, num_regions, height, width = region_masks.shape
    patch_count = height * width
    class_index = num_regions
    sequence_length = num_regions + 1 + patch_count

    allowed = torch.zeros((num_crops, sequence_length, sequence_length),
                          dtype=torch.bool,
                          device=region_masks.device)
    allowed[:, class_index:, class_index:] = True
    allowed[:, :num_regions, class_index] = True
    allowed[:, :num_regions, class_index + 1:] = region_masks.flatten(2)
    return ~allowed


def _run_crop_block(block, tokens, attention_masks):
    """Run one stock OpenCLIP block independently for each crop.

    The region mask is passed through the upstream residual block; v2.24
    converts bool to the token dtype, so it acts as an additive 0/1 bias.
    """
    crop_first = tokens.transpose(0, 1)  # [Ncrop, S, C]
    outputs = [
        block(crop.unsqueeze(1), attn_mask=mask)
        for crop, mask in zip(crop_first, attention_masks)
    ]
    return torch.cat(outputs, dim=1)  # [S, Ncrop, C]


def _run_global_spatial_block(block, tokens, num_regions, image_shape,
                              coords, paddings, patch_size):
    """Recompute one selected CLIP block over the stitched global patch grid.

    The crop stream provides Q/K/V. Each is overlap-averaged onto the image
    grid before attention, so the block can exchange context across windows.
    ``tokens`` uses ``[S, Ncrop, C]`` layout.
    """
    patch_tokens = tokens[num_regions + 1:].transpose(0, 1)  # [Ncrop,L,C]
    global_input, _ = _stitch_crop_tokens(patch_tokens, image_shape, coords,
                                           paddings, patch_size)

    # Match OpenCLIP's pre-norm attention: normalize, project Q/K/V, then
    # stitch each projection with the same spatial crop mapping.
    qkv = F.linear(block.ln_1(tokens), block.attn.in_proj_weight,
                   block.attn.in_proj_bias)[num_regions + 1:]
    query, key, value = qkv.chunk(3, dim=-1)
    query, _ = _stitch_crop_tokens(query.transpose(0, 1), image_shape, coords,
                                   paddings, patch_size)
    key, _ = _stitch_crop_tokens(key.transpose(0, 1), image_shape, coords,
                                 paddings, patch_size)
    value, _ = _stitch_crop_tokens(value.transpose(0, 1), image_shape, coords,
                                   paddings, patch_size)

    head_dim = block.attn.embed_dim // block.attn.num_heads
    # Preserve the paper fork's full-width QK product and per-head scale;
    # splitting heads here would change the attention scores.
    attention = torch.bmm(query, key.transpose(1, 2)) * (head_dim**-0.5)
    # All global image patches are mutually visible in this branch. The
    # region-aware scope is applied by the DINO-guided final attention.
    attention = attention.softmax(dim=-1)
    output = torch.bmm(attention, value)  # [1, Hgrid*Wgrid, C]

    spatial = global_input + block.attn.out_proj(output)
    spatial = spatial + block.mlp(block.ln_2(spatial))
    return spatial


def _run_final_block(visual, tokens, spatial_features, num_regions,
                     region_masks, scope_attention, image_shape, coords,
                     paddings, clip_patch_size, dino_patch_size,
                     dino_grid_shape, slide_crop):
    """Fuse global CLIP, stitched spatial, and region semantic features."""
    block = visual.transformer.resblocks[-1]
    main_tokens = tokens[num_regions:]  # [CLS + Lclip, Ncrop, C]
    global_spatial = torch.stack(spatial_features).mean(dim=0)  # [1,Lclip,C]

    # Project final-block V for the crop stream and global spatial stream.
    width = block.attn.embed_dim
    value_weight = block.attn.in_proj_weight[2 * width:]
    value_bias = block.attn.in_proj_bias[2 * width:]
    crop_values = F.linear(block.ln_1(main_tokens), value_weight, value_bias)
    crop_values, clip_grid = _stitch_crop_tokens(
        crop_values[1:].transpose(0, 1), image_shape, coords, paddings,
        clip_patch_size)
    spatial_values = F.linear(block.ln_1(global_spatial.transpose(0, 1)),
                              value_weight, value_bias)
    spatial_values = spatial_values.transpose(0, 1)  # [1,Lclip,C]

    # Add the two CLIP value sources, then resize to DINO's grid so the
    # DINO-derived [Ndino,Ndino] attention can mix their information.
    crop_values = crop_values.reshape(1, *clip_grid, width).permute(0, 3, 1, 2)
    spatial_values = spatial_values.reshape(1, *clip_grid,
                                            width).permute(0, 3, 1, 2)
    values = torch.cat((crop_values, spatial_values), dim=0)
    values = F.interpolate(values, size=dino_grid_shape, mode='bilinear',
                           align_corners=False)
    values = values.flatten(2).transpose(1, 2)  # [2, Ndino, C]
    global_outputs = torch.bmm(scope_attention.expand(2, -1, -1), values)
    global_outputs = block.attn.out_proj(global_outputs.transpose(0, 1))
    # [Ndino, 2, C]: two global branches, one output vector per DINO patch.

    # Region tokens complete their normal final CLIP block under crop masks.
    crop_first = tokens.transpose(0, 1)  # [Ncrop, S, C]
    crop_region_outputs = [
        block(crop.unsqueeze(1), attn_mask=mask)
        for crop, mask in zip(crop_first, _make_crop_attention_mask(region_masks))
    ]
    crop_region_outputs = torch.cat(crop_region_outputs, dim=1)
    region_tokens = crop_region_outputs[:num_regions].transpose(0, 1)
    # [Ncrop, Nregion, C]

    # Convert crop-region membership into one semantic vector per DINO patch.
    dino_crop_side = slide_crop // dino_patch_size
    semantic_weights = F.interpolate(
        region_masks.to(region_tokens.dtype),
        size=(dino_crop_side, dino_crop_side),
        mode='nearest',
    ).flatten(2)  # [Ncrop, Nregion, Ndino_crop]
    semantic_crop_tokens = semantic_weights.transpose(1, 2) @ region_tokens
    semantic_tokens, _ = _stitch_crop_tokens(
        semantic_crop_tokens, image_shape, coords, paddings, dino_patch_size)
    semantic_tokens = semantic_tokens.transpose(0, 1)  # [Ndino, 1, C]

    # Public OpenCLIP's final LayerNorm and projection apply to each branch.
    branches = torch.cat((global_outputs, semantic_tokens), dim=1)
    branches = visual.ln_post(branches.transpose(0, 1))  # [3, Ndino, C]
    main, spatial, semantic = branches.chunk(3, dim=0)
    fused = main + spatial + 0.5 * semantic
    return fused @ visual.proj  # [1, Ndino, 512] for ViT-B/16


@torch.inference_mode()
def encode_stitched_visual(visual, image_shape, crops, dino_features,
                           dino_patch_size, feature_shape, instance_masks,
                           coords, paddings, slide_stride, slide_crop):
    """Run the author's three-branch high-resolution CLIP feature path.

    Public OpenCLIP owns every learned layer and checkpoint parameter. This
    function changes only how crop tokens flow through those existing layers.
    """
    if visual.attn_pool is not None or visual.pool_type != 'tok':
        raise ValueError('OV-Stitcher adapter expects a non-pooled ViT model')

    crop_tokens, crop_grid = _embed_crops(visual, crops)
    clip_patch_size = visual.patch_size[0]
    dino_grid_shape = tuple(feature_shape)
    num_layers = len(visual.transformer.resblocks)

    # Keep the original layer schedule used for CLIP ViT-B/L/H checkpoints.
    spatial_layer_count = {12: 5, 24: 12, 32: 28}.get(num_layers,
                                                       num_layers - 4)
    spatial_layer_start = num_layers - 2 - spatial_layer_count

    scope_attention, dino_grid_masks = _dino_scope_attention(
        dino_features, dino_grid_shape, instance_masks)
    mask_ids = torch.unique(dino_grid_masks[0])
    region_masks = _crop_region_masks(
        dino_grid_masks, mask_ids, image_shape, slide_stride, slide_crop,
        crop_grid)

    # One copy of the CLS token becomes a query for each full-image region.
    num_regions = region_masks.shape[1]
    region_tokens = crop_tokens[:1].expand(num_regions, -1, -1)
    tokens = torch.cat((region_tokens, crop_tokens), dim=0)
    crop_attention_masks = _make_crop_attention_mask(region_masks)

    spatial_features = []
    for layer_index, block in enumerate(visual.transformer.resblocks[:-1]):
        if spatial_layer_start <= layer_index <= num_layers - 3:
            # These selected layers see all stitched image patches at once;
            # their outputs are averaged to form the later spatial branch.
            spatial = _run_global_spatial_block(
                block, tokens, num_regions, image_shape, coords, paddings,
                clip_patch_size)
            spatial_features.append(spatial)

        # The ordinary crop stream runs every block, with region-aware masks.
        tokens = _run_crop_block(block, tokens, crop_attention_masks)

    if not spatial_features:
        raise ValueError('CLIP layer schedule produced no spatial features')
    return _run_final_block(
        visual,
        tokens,
        spatial_features,
        num_regions,
        region_masks,
        scope_attention,
        image_shape,
        coords,
        paddings,
        clip_patch_size,
        dino_patch_size,
        dino_grid_shape,
        slide_crop,
    )
