from functools import partial
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop
from tqdm import tqdm
import numpy as np

from visinv import VisualInversion, MultiHeadCrossAttention
from params import parse_args
from dataset import FashionIQ
from custom_clip import CLIPModel
from transformers.models.clip import CLIPTokenizer


def get_metrics_fashion(image_features, ref_features, target_names, answer_names):
    metrics = {}
    distances = 1 - ref_features @ image_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(target_names)[sorted_indices]
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(answer_names), len(target_names)).reshape(len(answer_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(answer_names)).int())
    # Compute the metrics
    for k in [1, 5, 10, 50, 100]:
        metrics[f"R@{k}"] = (torch.sum(labels[:, :k]) / len(labels)).item() * 100
    return metrics


def preprocess(n_px: int, is_train: bool):
    normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    if is_train:
        return Compose([
            RandomResizedCrop(n_px, scale=(0.9, 1.0), interpolation=Image.BICUBIC),
            lambda image: image.convert('RGB'),
            ToTensor(),
            normalize,
        ])
    else:
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert('RGB'),
            ToTensor(),
            normalize,
        ])


def load_model(args):
    clip_model = CLIPModel.from_pretrained('clip-vit-large-patch14')
    tokenizer = CLIPTokenizer.from_pretrained('clip-vit-large-patch14')
    checkpoint = torch.load('saved/20240316-18-20-05_dress/epoch00.pt')
    fm_model = VisualInversion(embed_dim=768, middle_dim=768, output_dim=768)
    visinv_attn = MultiHeadCrossAttention(src_dim=768, tgt_dim=1024, num_heads=8)
    # fm_model.load_state_dict(checkpoint['fm_state_dict'])
    # visinv_attn.load_state_dict(checkpoint['attn_state_dict'])
    # move models to GPU
    if args.gpu is not None:
        clip_model.cuda(args.gpu)
        fm_model.cuda(args.gpu)
        visinv_attn.cuda(args.gpu)
    return preprocess(224, False), clip_model, tokenizer, fm_model, visinv_attn


def fashion_eval(args, root_project):
    preprocess_val, clip_model, tokenizer, fm_model, visinv_attn = load_model(args)
    assert args.source_data in ['dress', 'shirt', 'toptee']
    source_dataset = FashionIQ(cloth=args.source_data,
                               split='val',
                               transforms=preprocess_val,
                               root=root_project,
                               is_return_target_path=True)
    target_dataset = FashionIQ(cloth=args.source_data,
                               split='val',
                               transforms=preprocess_val,
                               root=root_project,
                               mode='imgs')
    source_dataloader = DataLoader(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False)
    target_dataloader = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False)

    evaluate_fashion(
        clip_model=clip_model,
        tokenizer=tokenizer,
        fm_model=fm_model,
        visinv_attn=visinv_attn,
        source_loader=source_dataloader,
        target_loader=target_dataloader,
        args=args
    )


def tensor2img(tensor):
    # Check if input is a PyTorch tensor and has the correct shape
    if not isinstance(tensor, torch.Tensor) or tensor.dim() != 4 or tensor.shape[1] != 3:
        raise ValueError("Input must be a PyTorch tensor with shape [bs, 3, 224, 224]")

    # Normalize the tensor values to [0, 255] as PIL Images expect these ranges for RGB images
    # Assuming the tensor is in the range [0, 1], if it's not, you should adjust the normalization accordingly
    tensor = tensor.mul(255).byte()

    images = []
    for img_tensor in tensor:
        # Convert the tensor to PIL Image
        img = Image.fromarray(img_tensor.cpu().permute(1, 2, 0).numpy(), 'RGB')
        images.append(img)

    return images


def evaluate_fashion(
        clip_model,
        tokenizer,
        fm_model,
        visinv_attn,
        source_loader,
        target_loader,
        args
):
    all_composed_feat = []
    all_image_features = []
    all_target_paths = []
    all_answer_paths = []
    # get all image features
    with torch.no_grad():
        for batch in tqdm(target_loader, desc="Target Features:"):
            target_images, target_paths = batch
            target_images = target_images.cuda(0, non_blocking=True)
            image_features = clip_model.get_image_features(target_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features)
            for path in target_paths:
                all_target_paths.append(path)
    # get composed features
    with torch.no_grad():
        for batch in tqdm(source_loader):
            ref_images, target_images, target_caption, caption_only, answer_paths, ref_names, captions = batch
            for path in answer_paths:
                all_answer_paths.append(path)
            text_tokens = tokenizer(captions, padding=True, return_tensors="pt")
            if args.gpu is not None:
                text_tokens = text_tokens.to(f"cuda:{args.gpu}")
                ref_images = ref_images.to(f"cuda:{args.gpu}")
            text_feature_raw = clip_model.get_text_features(**text_tokens)
            text_feature_mapped = fm_model(text_feature_raw)
            composed_feature = text_feature_raw
            composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)
            all_composed_feat.append(composed_feature)
    metric_func = partial(get_metrics_fashion,
                          image_features=torch.cat(all_image_features),
                          target_names=all_target_paths, answer_names=all_answer_paths)
    feats = {
        'composed': torch.cat(all_composed_feat)
    }
    for key, value in feats.items():
        metrics = metric_func(ref_features=value)
        print(
            f"Eval {key} Feature"
            + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        )
    return metrics


if __name__ == '__main__':
    args = parse_args()
    fashion_eval(args, "data")
