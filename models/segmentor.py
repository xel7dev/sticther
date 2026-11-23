"""MMSegmentation adapter for the OV-Stitcher inference pipeline."""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms

from mmengine.structures import PixelData
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.models.segmentors import BaseSegmentor
from mmseg.registry import MODELS
from open_clip import create_model, tokenize

# The canonical layout imports this as ``ov_stitcher.models``; a flattened
# checkout imports it as top-level ``models``. Resolve the sibling utility in
# either package layout without changing the inference path.
if __package__ == 'models':
    from ov_stitcher_utils import UnNormalize
else:
    from ..ov_stitcher_utils import UnNormalize
from .clip_stitching import StitchedCLIP
from .features import make_sliding_windows, stitch_dino_features
from .masks import correct_predictions_by_region, load_instance_mask
from .text import build_query_features


@MODELS.register_module()
class OVStitcherSegmentation(BaseSegmentor):
    """Training-free evaluation chain from image pixels to region labels.

    MMSeg supplies one normalized image ``[1,3,H,W]`` and image metadata.
    DINO and CLIP produce aligned patch features, text prompts produce class
    vectors, and mask-based voting converts patch scores into region-consistent
    predictions at the source image resolution.
    """

    def __init__(self,
                 clip_type,
                 model_type,
                 dino_type,
                 name_path,
                 instance_mask_path,
                 slide_stride,
                 slide_crop,
                 device=torch.device('cuda'),
                 prob_thd=0.0,
                 logit_scale=40,
                 template_path=None):
        preprocessor = SegDataPreProcessor(mean=[122.771, 116.746, 104.094],
                                           std=[68.501, 66.632, 70.323],
                                           bgr_to_rgb=True)
        super().__init__(data_preprocessor=preprocessor)

        # The stock package owns CLIP's weights, text encoder and blocks. The
        # adapter changes only image-token flow for OV-Stitcher's crop path.
        public_clip = create_model(model_type,
                                   pretrained=clip_type,
                                   precision='fp16')
        self.clip = StitchedCLIP(public_clip)
        self.clip.eval().to(device)
        for parameter in self.clip.parameters():
            parameter.requires_grad = False
        self.tokenize = tokenize

        self.dino = torch.hub.load('facebookresearch/dino:main', dino_type)
        self.dino.eval().to(device)
        for parameter in self.dino.parameters():
            parameter.requires_grad = False
        self.dino = self.dino.half()
        self.dino_patch_size = self._get_patch_size(
            self.dino.patch_embed.patch_size)

        # MMEngine's distributed test wrapper requires one trainable parameter.
        self._ddp_anchor = nn.Parameter(torch.zeros(1, device=device))
        self.unnormalize_image = UnNormalize(
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711])
        self.normalize_image = transforms.Normalize([0.485, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225])

        # Cache text prototypes once at model construction, not once per image.
        # query_features: [Nalias,512]; class_indices maps aliases to classes.
        self.query_features, class_indices = build_query_features(
            self.clip, self.tokenize, name_path, template_path, device)
        self.num_queries = len(class_indices)
        self.num_classes = max(class_indices) + 1

        self.instance_mask_path = Path(instance_mask_path)
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.device = device
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd

    @staticmethod
    def _get_patch_size(patch_size):
        if isinstance(patch_size, tuple):
            if patch_size[0] != patch_size[1]:
                raise ValueError('DINO patch size must be square')
            return patch_size[0]
        return patch_size

    @torch.inference_mode()
    def forward_feature(self, crops, coords, paddings, masks):
        """Return dense query logits on DINO's stitched patch grid.

        ``crops`` is ``[Ncrop,3,Hcrop,Wcrop]``; masks are ``[1,1,H,W]``.
        DINO and CLIP features align to ``[1,Hd*Wd,*]``. Their dot products
        become ``[1,Hd*Wd,Nalias]`` and then ``[1,Nalias,Hd,Wd]``.
        """
        # Reconstruct DINO's final attention block over all stitched patches,
        # while earlier DINO blocks remain batched over local crops.
        dino_features, feature_shape = stitch_dino_features(
            self.dino, crops, coords, paddings, self.img_shape,
            self.normalize_image, self.unnormalize_image, self.dino_patch_size)
        # Reuse public OpenCLIP weights through the OV-Stitcher image adapter;
        # both encoders now describe the same DINO patch grid.
        image_features = self.clip.encode_image(
            self.img_shape,
            crops.half(),
            dino_features=dino_features,
            dino_patch_size=self.dino_patch_size,
            feature_shape=feature_shape,
            instance_masks=masks,
            coords=coords,
            paddings=paddings,
            slide_stride=self.slide_stride,
            slide_crop=self.slide_crop)
        image_features = F.normalize(image_features, dim=-1)  # [1, Hd*Wd, 512]
        logits = image_features @ self.query_features.T  # [1, Hd*Wd, Nalias]
        logits = logits.permute(0, 2, 1).reshape(-1, logits.shape[-1],
                                                 *feature_shape)
        # Patch-grid scores -> preprocessed image pixels; source-size resize
        # happens in forward_slide after the original metadata is consulted.
        return F.interpolate(logits, size=self.img_shape[-2:], mode='bilinear')

    def forward_slide(self, image, instance_masks, image_metadata):
        """Tile one preprocessed image, infer global logits, restore source size."""
        self.img_shape = image.shape
        # [1,3,H,W] -> [Ncrop,3,Hcrop,Wcrop], plus crop placement metadata.
        crops, coords, paddings = make_sliding_windows(image,
                                                       self.slide_stride,
                                                       self.slide_crop)
        logits = self.forward_feature(crops, coords, paddings, instance_masks)
        original_size = image_metadata[0]['ori_shape'][:2]
        # Metrics use original-resolution labels, so restore logits to that size.
        return F.interpolate(logits, size=original_size, mode='bilinear')

    @torch.inference_mode()
    def predict(self, inputs, data_samples):
        if data_samples is None:
            raise ValueError(
                'Evaluation requires data_samples with image paths')
        if len(data_samples) != 1:
            raise ValueError('OV-Stitcher evaluation requires batch_size=1')

        image_metadata = [sample.metainfo for sample in data_samples]
        # .npz masks match the source image. Keep original_masks for the final
        # region vote; resize a copy to model input size for feature-grid masks.
        original_masks = torch.stack([
            load_instance_mask(self.instance_mask_path, sample.img_path,
                               self.device) for sample in data_samples
        ]).int()
        resized_masks = F.interpolate(original_masks.unsqueeze(1).float(),
                                      size=inputs.shape[2:],
                                      mode='nearest').int()
        logits = self.forward_slide(inputs, resized_masks, image_metadata)
        return self.postprocess_result(logits, data_samples, original_masks)

    def postprocess_result(self, logits, data_samples, instance_masks):
        for index, sample in enumerate(data_samples):
            # [Nalias,Horig,Worig] -> per-pixel class probabilities.
            class_scores = (logits[index] * self.logit_scale).softmax(0)
            if self.num_classes != self.num_queries:
                # Several aliases may represent the background class. Collapse
                # those alias channels by max, preserving one output per class.
                background_count = self.num_queries - self.num_classes + 1
                background_scores = class_scores[:background_count].max(
                    0, keepdim=True)[0]
                class_scores = torch.cat(
                    [background_scores, class_scores[background_count:]])

            prediction = class_scores.argmax(0, keepdim=True)  # [1,Horig,Worig]
            confidence = class_scores.max(0, keepdim=True)[0]
            # Low-confidence pixels become the reserved background ID 0.
            prediction[confidence < self.prob_thd] = 0
            # Force each SAM2 region to its majority predicted label.
            prediction = correct_predictions_by_region(prediction,
                                                       instance_masks[index])
            sample.set_data({
                'seg_logit': PixelData(data=class_scores),
                'pred_sem_seg': PixelData(data=prediction),
            })
        return data_samples

    def _forward(self, inputs, data_samples=None):
        raise NotImplementedError(
            'OV-Stitcher supports evaluation via predict only')

    def encode_decode(self, inputs, batch_img_metas):
        raise NotImplementedError(
            'OV-Stitcher uses its sliding-window predict path')

    def extract_feat(self, inputs):
        raise NotImplementedError(
            'OV-Stitcher extracts stitched features in predict')

    def loss(self, inputs, data_samples):
        raise NotImplementedError('OV-Stitcher is training-free')
