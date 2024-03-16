
# test the dimensions of text encoder's last hidden layer

from transformers import CLIPProcessor, CLIPModel

# Load the model and processor
model = CLIPModel.from_pretrained("../../clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("../../clip-vit-large-patch14")

# Example text inputs
texts = ["Hello, world!", "The quick brown fox jumps over the lazy dog."]

# Process the texts and generate outputs
inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
outputs = model.get_text_features(**inputs)

# outputs now contains the last hidden states of the text encoder
# Let's print the shape of the last hidden layer
print("Shape of the last hidden layer:", outputs.shape)
