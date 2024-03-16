from custom_clip import CLIPModel
from transformers import CLIPFeatureExtractor
import torch


# Initialize the feature extractor and model
feature_extractor = CLIPFeatureExtractor.from_pretrained("../../clip-vit-large-patch14")
model = CLIPModel.from_pretrained("../../clip-vit-large-patch14")

test_image_encoder = True
if test_image_encoder:
    # Generate a batch of images (random noise)
    batch_size = 1  # Example batch size
    images = torch.rand(batch_size, 3, 224, 224)  # Example image data with the shape (batch_size, channels, height, width)

    # Preprocess images
    inputs = feature_extractor(images=images, return_tensors="pt")

    # Pass images to clip model
    outputs = model(pixel_values=inputs["pixel_values"])

    # Get the image features (embeddings)
    image_features = outputs.image_embeds