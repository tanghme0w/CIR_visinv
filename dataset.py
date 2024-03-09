import logging
from torch.utils.data import Dataset
import os
import json
from PIL import Image
from third_party.open_clip.clip import tokenize


# Haomiao Tang: Copied from pic2word/src/data.FashionIQ
# Fashion-IQ: under ./data/fashion-iq
# validation images ./images
# caption split ./json/cap.{cloth_type}.val.json, cloth_type in [toptee, shirt, dress]
# image split ./image_splits/split.{cloth_type}.val.json, cloth_type in [toptee, shirt, dress]
class FashionIQ(Dataset):
    def __init__(self, cloth, transforms, vis_mode=False, mode='caps', is_return_target_path=False, root='./data'):
        root_iq = os.path.join(root, 'fashion-iq')
        self.root_img = os.path.join(root_iq, 'images')
        self.vis_mode = vis_mode
        self.mode = mode
        self.is_return_target_path = is_return_target_path
        self.transforms = transforms
        if mode == 'imgs':
            self.json_file = os.path.join(root_iq, 'image_splits',
                                          'split.{}.val.json'.format(cloth))
        else:
            self.json_file = os.path.join(root_iq, 'json',
                                          'cap.{}.val.json'.format(cloth))
        logging.debug(f'Loading json data from {self.json_file}.')

        self.ref_imgs = []
        self.target_imgs = []
        self.ref_caps = []
        self.target_caps = []
        if mode == 'imgs':
            self.init_imgs()
            logging.info("Use {} imgs".format(len(self.target_imgs)))
        else:
            self.init_data()
            logging.info("Use {} imgs".format(len(self.target_imgs)))

    def init_imgs(self):
        data = json.load(open(self.json_file, "r"))
        self.target_imgs = [key + ".png" for key in data]

    def init_data(self):
        def load_data(data):
            for d in data:
                ref_path = os.path.join(self.root_img, d['candidate'] + ".png")
                tar_path = os.path.join(self.root_img, d['target'] + ".png")
                try:
                    Image.open(ref_path)
                    Image.open(tar_path)
                    self.ref_imgs.append(ref_path)
                    self.target_imgs.append(tar_path)
                    self.ref_caps.append((d['captions'][0], d['captions'][1]))
                    # self.target_caps.append(d['captions'][1])
                except:
                    print('cannot load {}'.format(d['candidate']))

        if isinstance(self.json_file, str):
            data = json.load(open(self.json_file, "r"))
            load_data(data)
        elif isinstance(self.json_file, list):
            for filename in self.json_file:
                data = json.load(open(filename, "r"))
                load_data(data)

    def __len__(self):
        if self.mode == 'caps':
            return len(self.ref_imgs)
        else:
            return len(self.target_imgs)

    def return_imgs(self, idx):
        tar_path = str(self.target_imgs[idx])
        img_path = os.path.join(self.root_img, tar_path)
        target_images = self.transforms(Image.open(img_path))
        return target_images, os.path.join(self.root_img, tar_path)

    def return_all(self, idx):
        if self.vis_mode:
            tar_path = str(self.target_imgs[idx])
            target_images = self.transforms(Image.open(tar_path))
            return target_images, tar_path
        ref_images = self.transforms(Image.open(str(self.ref_imgs[idx])))
        target_images = self.transforms(Image.open(str(self.target_imgs[idx])))
        cap1, cap2 = self.ref_caps[idx]
        text_with_blank = 'a photo of * , {} and {}'.format(cap2, cap1)
        token_texts = tokenize(text_with_blank)[0]
        if self.is_return_target_path:
            return ref_images, target_images, token_texts, token_texts, \
                str(self.target_imgs[idx]), str(self.ref_imgs[idx]), \
                cap1
        else:
            return ref_images, target_images, text_with_blank

    def __getitem__(self, idx):
        if self.mode == 'imgs':
            return self.return_imgs(idx)
        else:
            return self.return_all(idx)
