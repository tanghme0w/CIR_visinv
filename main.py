from transformers import CLIPModel, AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp.grad_scaler import GradScaler
from third_party.open_clip.scheduler import cosine_lr
from params import parse_args

from preprocess import preprocess
from visinv import VisualInversion
from dataset import FashionIQ
from trainer import train


def main():
    # parse args
    args = parse_args()
    # load CLIP model & CLIP tokenizer
    clip_model = CLIPModel.from_pretrained("../clip-vit-large-patch14")
    clip_tokenizer = AutoTokenizer.from_pretrained("../clip-vit-large-patch14")
    # create feature mapping model
    fm_model = VisualInversion(embed_dim=768, middle_dim=768, output_dim=768)
    if args.gpu is not None:
        clip_model.cuda(args.gpu)
        fm_model.cuda(args.gpu)
    # create dataset
    dataset = FashionIQ(
        cloth=args.source_data,
        transforms=preprocess(224, True),
        root="data",
        is_return_target_path=True
    )
    # create dataloaders
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,    # todo what does this mean?
        drop_last=False     # todo what does this mean?
    )
    # get optimizer, scaler, and scheduler
    exclude = lambda n: "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n: not exclude(n)
    named_parameters = list(fm_model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]
    # do not apply weight decay for norms, biases, and logit scales
    optimizer = AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": args.wd},
        ],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )
    scaler = GradScaler() if args.precision == 'amp' else None
    total_steps = len(dataloader) * args.epochs
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)     # todo explore the effect of warmup steps
    # run training script
    train(clip_model, clip_tokenizer, fm_model, dataloader, args.epochs, optimizer, scaler, scheduler, args)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
