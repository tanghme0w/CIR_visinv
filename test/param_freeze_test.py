from custom_clip import CLIPModel

model = CLIPModel.from_pretrained("../../clip-vit-large-patch14")

vision_encoder = model.vision_model.encoder

visinv_module = model.vision_model.encoder.visinv_attn

for name, param in model.named_parameters():
    if name.__contains__('visinv_attn'):
        print(name)
