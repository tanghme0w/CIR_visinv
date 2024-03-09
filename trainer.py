import time

import torch
from torch.cuda.amp import autocast
from torch.nn import CrossEntropyLoss


def calculate_loss(clip_model, clip_tokenizer, fm_model, ref_img, text, tgt_img, args):
    # get source embedding
    with torch.no_grad():
        tokenized_text = clip_tokenizer(text, padding=True, return_tensors="pt")
        if args.gpu is not None:
            tokenized_text.to(f"cuda:{args.gpu}")
        text_features = clip_model.get_text_features(**tokenized_text)
        ref_img_features = clip_model.get_image_features(ref_img)
    # get mapped text feature
    text_features = fm_model(text_features)
    # todo fusion by cross attention
    # todo get target embedding
    # todo calculate loss and return


def get_composed_feature(model, p_token, args):
    pass


def train(clip_model, clip_tokenizer, fm_model, dataloader, epoch, optimizer, scaler, scheduler, args):
    clip_model.eval()
    # get batches per epoch
    num_batches_per_epoch = len(dataloader)
    # track time
    time_track = time.time()
    # training pipeline
    for i, batch in enumerate(dataloader):
        global_step_num = num_batches_per_epoch * epoch + i
        scheduler(global_step_num)
        optimizer.zero_grad()

        ref_images, target_images, _, _, answer_paths, _, captions = batch

        data_identifier = -1
        if args.gpu is not None:
            ref_images = ref_images.cuda(args.gpu, non_blocking=True)   # todo what does non_blocking mean
            target_images = target_images.cuda(args.gpu, non_blocking=True)
        data_time = time.time() - time_track

        # with automatic mixed precision.
        if args.precision == "amp":
            with autocast():
                total_loss = calculate_loss(clip_model, clip_tokenizer, fm_model, ref_images, captions, target_images, args)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
            scaler.update()

        else:
            total_loss = calculate_loss(clip_model, clip_tokenizer, fm_model, ref_images, captions, target_images, args)
            total_loss.backward()
            optimizer.step()

        batch_time = time.time() - time_track
        time_track = time.time()
        # todo output pipeline

