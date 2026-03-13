import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np


class CLIPEmbedder:

    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(self.device)

        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    def embed_text(self,text: str) -> np.ndarray :
        """Embed text using CLIP"""
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77  #clip's max token length
        ).to(self.device)

        with torch.no_grad():
            features = self.model.get_text_features(**inputs)

            # In case it returns BaseModelOutputWithPooling
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output

            #normalize embeddings
            features = features / features.norm(p=2,dim=-1, keepdim=True)
            return features.squeeze(0).cpu().detach().numpy()
        
    def embed_image(self, image_data) -> np.ndarray :
        """Embed image using CLIP"""
        if isinstance(image_data, str): #if image path is passed then open image an convert to rgb and store in image
            image = Image.open(image_data).convert("RGB")
        else: # If PIL Image, directly image is passed then store in image
            image = image_data

        #to convert image into image vector we use clip processor. and return tensors in pytorch tensors.
        #it converts entire format into tensors
        inputs = self.processor(images=image, return_tensors='pt').to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)

            # In case it returns BaseModelOutputWithPooling (object) instead of tensor. we should extract .pooler_output
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output

            # normalize embeddings to unit vector
            features = features / features.norm(p=2,dim=-1, keepdim=True)
            return features.squeeze(0).cpu().detach().numpy()