"""Text query preparation for open-vocabulary classification."""

import json

import torch

# Support both ``ov_stitcher.models`` in the canonical layout and top-level
# ``models`` imports in flattened checkouts.
if __package__ == 'models':
    from prompts.imagenet_template import openai_imagenet_template
else:
    from ..prompts.imagenet_template import openai_imagenet_template


def read_class_names(path):
    """Read class rows and expand semicolon-separated name aliases.

    Returns parallel lists: aliases are used to form prompts, and each alias
    retains the zero-based row index of its canonical segmentation class.
    """
    names, class_indices = [], []
    with open(path, encoding='utf-8') as class_file:
        for class_index, line in enumerate(class_file):
            aliases = [name.replace('\n', '') for name in line.split('; ')]
            names.extend(aliases)
            class_indices.extend([class_index] * len(aliases))
    return names, class_indices


def load_class_templates(path):
    if path is None:
        return {}
    with open(path, encoding='utf-8') as template_file:
        return json.load(template_file)


@torch.inference_mode()
def build_query_features(clip, tokenize, class_file, template_file, device):
    """Encode text prototypes consumed by the dense image-text dot product.

    For each alias, ``T`` prompt strings become token IDs ``[T,77]`` and then
    CLIP vectors ``[T,512]``. Normalizing each prompt before averaging reduces
    prompt-length/scale variation; the normalized mean gives one prototype per
    alias. The result is ``[Nalias,512]`` plus alias-to-class row indices.
    """
    class_names, class_indices = read_class_names(class_file)
    class_templates = load_class_templates(template_file)
    query_features = []
    for name in class_names:
        # Combine the standard ImageNet wording with optional class-specific
        # prompt variants; all variants share the same CLIP text encoder.
        prompts = [template(name) for template in openai_imagenet_template]
        prompts.extend(class_templates.get(name, []))
        tokens = tokenize(prompts).to(device)  # [T, 77] integer token IDs
        features = clip.encode_text(tokens)  # [T, 512] text embeddings
        features = features / features.norm(dim=-1, keepdim=True)  # unit rows
        feature = features.mean(dim=0)  # average prompt variants -> [512]
        feature = feature / feature.norm()  # unit class-alias prototype
        query_features.append(feature.unsqueeze(0))
    return torch.cat(query_features, dim=0).detach(), class_indices
