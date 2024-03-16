from custom_clip import CLIPModel
from transformers import CLIPProcessor, AutoTokenizer
import torch


model = CLIPModel.from_pretrained("../../clip-vit-large-patch14")
preprocessor = CLIPProcessor.from_pretrained("../../clip-vit-large-patch14")
tokenizer = AutoTokenizer.from_pretrained("../../clip-vit-large-patch14")

# Generate a batch of images (random noise)
batch_size = 1  # Example batch size
images = torch.rand(batch_size, 3, 224, 224)  # Example image data with the shape (batch_size, channels, height, width)
texts = ["abaababa bababababa"]

# Preprocess images
image_inputs = preprocessor(images=images, return_tensors="pt")
text_inputs = tokenizer(text=texts, return_tensors="pt")

# Pass images to clip model
outputs = model.get_composed_features(text_inputs['input_ids'], **image_inputs)
print(outputs.shape)
