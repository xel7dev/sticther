"""Sliding-window extraction and global DINO attention reconstruction."""

import math

import torch
import torch.nn.functional as F


def compute_padsize(height, width, patch_size):
    left = right = top = bottom = 0
    if width % patch_size:
        horizontal = patch_size - width % patch_size
        left, right = horizontal // 2, horizontal - horizontal // 2
    if height % patch_size:
        vertical = patch_size - height % patch_size
        top, bottom = vertical // 2, vertical - vertical // 2
    return left, right, top, bottom


def make_sliding_windows(img, slide_stride, slide_crop):
    """Tile one image tensor into overlapping, patch-aligned crops.

    Input is ``[1, C, H, W]``. The output is ``[Ncrop, C, Hcrop, Wcrop]``;
    ``coordinates`` stores each crop's unpadded ``(y1,x1,y2,x2)`` in the full
    image, while ``paddings`` stores ``(left,right,top,bottom)``. Overlap keeps
    edge patches covered; coordinates later place those patches back globally.
    """
    stride = (slide_stride,
              slide_stride) if isinstance(slide_stride, int) else slide_stride
    crop_size = (slide_crop,
                 slide_crop) if isinstance(slide_crop, int) else slide_crop
    h_stride, w_stride = stride
    h_crop, w_crop = crop_size
    _, _, h_img, w_img = img.shape
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

    crops, paddings, coordinates = [], [], []
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop = img[:, :, y1:y2, x1:x2]
            coordinates.append(torch.tensor([y1, x1, y2, x2]))

            crop_height, crop_width = crop.shape[2:]
            # 56 keeps crop boundaries aligned to common 8/14-pixel ViT patch
            # grids used by DINO checkpoints; normal configured crops (224/336)
            # are already multiples of 56 and therefore remain unchanged.
            padding = compute_padsize(crop_height, crop_width, 56)
            if any(padding):
                crop = F.pad(crop, padding)
            crops.append(crop)
            paddings.append(padding)

    return torch.cat(crops, dim=0), coordinates, paddings


def stitch_features(feature, pixel_locs, paddings, patch_size, image_shape):
    """Average overlapping crop tokens on one full-image patch grid.

    Args:
        feature: Crop tokens ``[Ncrop, Lcrop, C]``.
        pixel_locs: Crop coordinates in full-image pixels.
        paddings: Pixel padding applied to each crop.
        patch_size: Encoder patch stride (DINO or CLIP).
        image_shape: Full image tensor shape ``[1, C, H, W]``.

    Returns:
        Tokens ``[1, Hgrid*Wgrid, C]`` and grid shape ``(Hgrid,Wgrid)``.
        Overlap counts divide summed features so overlapping windows contribute
        an average instead of changing feature scale.
    """
    channels = feature.shape[-1]
    feature = feature.permute(0, 2, 1)  # [Ncrop, C, Lcrop] for 2D placement
    height = image_shape[2] // patch_size
    width = image_shape[3] // patch_size
    stitched = torch.zeros((1, channels, height, width),
                           device=feature.device,
                           dtype=feature.dtype)
    counts = torch.zeros((1, 1, height, width),
                         device=feature.device,
                         dtype=feature.dtype)

    for index, (loc, padding) in enumerate(zip(pixel_locs, paddings)):
        y1, x1, y2, x2 = loc
        pad_l, pad_r, pad_t, pad_b = padding
        padded_h = (y2 - y1) + pad_t + pad_b
        padded_w = (x2 - x1) + pad_l + pad_r
        # Conv patchifiers use floor division when a padded edge is not an
        # exact patch multiple, so reconstruct the tensor's real output grid.
        crop_h = padded_h // patch_size
        crop_w = padded_w // patch_size
        y1_patch, x1_patch = y1 // patch_size, x1 // patch_size
        pad_t_patch = math.ceil(pad_t / patch_size)
        pad_b_patch = math.ceil(pad_b / patch_size)
        pad_l_patch = math.ceil(pad_l / patch_size)
        pad_r_patch = math.ceil(pad_r / patch_size)
        # Keep only tokens whose patch starts inside the unpadded crop. This
        # matches the encoder's floor-sized output for short edge windows.
        valid_h = (y2 - y1) // patch_size
        valid_w = (x2 - x1) // patch_size

        # Restore the crop's patch grid before removing padding-only tokens.
        crop_features = feature[index].reshape(channels, crop_h, crop_w)
        crop_features = crop_features[:, pad_t_patch:pad_t_patch + valid_h,
                                      pad_l_patch:pad_l_patch + valid_w]
        stitched[:, :, y1_patch:y1_patch + valid_h,
                 x1_patch:x1_patch + valid_w] += crop_features.unsqueeze(0)
        counts[:, :, y1_patch:y1_patch + valid_h,
               x1_patch:x1_patch + valid_w] += 1

    assert (counts == 0).sum() == 0, 'Some positions not filled.'
    stitched = stitched / counts  # average patches seen by multiple crops
    return stitched.reshape(1, channels, -1).permute(0, 2,
                                                     1), stitched.shape[-2:]


@torch.inference_mode()
def stitch_dino_features(dino, img, coords, paddings, image_shape,
                         normalize_image, unnormalize_image, patch_size):
    """Rebuild DINO's final attention block over the full image.

    ``img`` contains CLIP-normalized crops ``[Ncrop,3,Hcrop,Wcrop]``. The
    returned normalized tokens are ``[1,Hgrid*Wgrid,Cdino]``. All but DINO's
    final block still run independently per crop; this function replaces that
    last local attention step with one global attention pass over stitched Q/K/V.
    """
    # DINO expects ImageNet normalization. Undo the CLIP normalization applied
    # by MMSeg, then normalize each crop for DINO without changing its pixels.
    normalized = torch.stack([
        normalize_image(unnormalize_image(img[index]))
        for index in range(len(img))
    ]).half()

    captured = {}

    def capture_qkv(module, inputs, output):
        captured['qkv'] = output

    qkv_layer = dino.blocks[-1].attn.qkv
    # Capture Q/K/V from DINO's last block while requesting the two final
    # hidden states; the first returned state is the penultimate block output.
    hook = qkv_layer.register_forward_hook(capture_qkv)
    try:
        penultimate = dino.get_intermediate_layers(normalized, n=2)[0]
    finally:
        hook.remove()

    # Hook output is [Ncrop, Lcrop+1, 3*Cdino]. DINO tensors include CLS at
    # token 0; remove CLS from spatial Q/K/V because it has no pixel location.
    batch_size, token_count = penultimate.shape[:2]
    qkv = captured['qkv'][:, 1:].reshape(batch_size, token_count - 1, 3,
                                         -1).permute(2, 0, 1, 3)
    # Split Q/K/V: each is [Ncrop, Lcrop, Cdino].
    query, key, value = qkv.unbind(0)

    # Average overlapping crops into one global token grid: [1, Hd*Wd, Cdino].
    query, _ = stitch_features(query, coords, paddings, patch_size,
                               image_shape)
    key, _ = stitch_features(key, coords, paddings, patch_size, image_shape)
    value, _ = stitch_features(value, coords, paddings, patch_size,
                               image_shape)

    attention = dino.blocks[-1].attn
    head_dim = dino.embed_dim // attention.num_heads
    scale = head_dim**-0.5
    # Split channels into heads: [1,Nd,C] -> [heads,Nd,head_dim], so bmm
    # computes one global attention matrix per head.
    query = query.contiguous().view(-1, attention.num_heads, head_dim)
    query = query.transpose(0, 1).to(value.dtype)
    key = key.contiguous().view(-1, attention.num_heads, head_dim)
    key = key.transpose(0, 1).to(value.dtype)
    value = value.contiguous().view(-1, attention.num_heads, head_dim)
    value = value.transpose(0, 1)

    # Global self-attention costs O((H/patch * W/patch)^2), but permits tokens
    # from different crops to exchange context in the final DINO block.
    attention_output = torch.bmm(query, key.transpose(1, 2)) * scale
    attention_output = F.softmax(attention_output, dim=-1)
    attention_output = torch.bmm(attention_output, value)
    # Merge heads back to [Nd,1,Cdino], then restore batch-first [1,Nd,Cdino].
    attention_output = attention_output.transpose(0, 1).contiguous().view(
        -1, 1, dino.embed_dim)

    # Stitch the residual stream on the same grid as Q/K/V before applying the
    # final block's attention projection and MLP residual.
    penultimate, feature_shape = stitch_features(penultimate[:, 1:], coords,
                                                 paddings, patch_size,
                                                 image_shape)
    features = penultimate + F.linear(attention_output.permute(
        1, 0, 2), attention.proj.weight, attention.proj.bias)

    block = dino.blocks[-1]
    mlp_input = F.layer_norm(features,
                             normalized_shape=block.norm2.normalized_shape,
                             weight=block.norm2.weight,
                             bias=block.norm2.bias,
                             eps=block.norm2.eps)
    mlp_input = F.gelu(
        F.linear(mlp_input, block.mlp.fc1.weight, block.mlp.fc1.bias))
    features = features + F.linear(mlp_input, block.mlp.fc2.weight,
                                   block.mlp.fc2.bias)
    return F.normalize(features, dim=-1), feature_shape
