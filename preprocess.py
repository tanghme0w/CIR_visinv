from torchvision.transforms import Normalize, Compose, RandomResizedCrop, ToTensor, Resize, CenterCrop
from PIL import Image


# copied from pic2word src/eval_minimal.preprocess
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