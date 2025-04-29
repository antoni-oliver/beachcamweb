import io
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from django.utils import timezone
from torchvision import transforms

from predictions.DTO.PredictionDTO import PredictionDTO
from predictions.classes.bayesian_stuff.vgg import vgg19
from predictions.interfaces.PredictorInterface import PredictorInterface


class BayesianPredictor(PredictorInterface):
    transformer = None
    model = None
    device = "cpu"
    weigth_path = "./predictions/classes/bayesian_stuff/best_model.pth"
    alpha_channel = 75
    density_map_intensity = 250
    
    def __init__(self):
        self.transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])    
    
    def predict(self, image_path: str, mask_paths = []) -> PredictionDTO:
        self.prepareModel()
        inputs = self.processImage(image_path)

        with torch.set_grad_enabled(False):
            outputs = self.model(inputs)
            density_map = outputs.squeeze().cpu().numpy()
        
        if(mask_paths):
            density_map = self.applyMasks(mask_paths, density_map)
            
        merged_image = self.mergeDensityMapWithImage(image_path, density_map)
        
        return PredictionDTO(
            crowd_count= round(np.sum(density_map)),
            time_stamp= timezone.now(),
            img_predict_content=merged_image
        )
        
    def prepareModel(self):
        self.model = vgg19()
        device = torch.device(self.device)
        self.model.to(device)
        self.model.load_state_dict(torch.load(os.path.abspath(self.weigth_path), device))

    def processImage(self, image_path: str):
        img = Image.open(image_path).convert('RGB')
        img = self.transformer(img)
        img = img.unsqueeze(0)
        return img.to(self.device)
    
    def applyMasks(self, mask_paths: list, density_map: np):
        combined_mask = np.zeros_like(density_map)
        
        for mask_path in mask_paths:
            if not os.path.exists(mask_path):
                continue
            mask_image = Image.open(mask_path).convert('L')
            mask_image = mask_image.resize((density_map.shape[1], density_map.shape[0]), Image.BILINEAR)
            np_mask = np.array(mask_image) < 128 #dark areas -> 1
            combined_mask = combined_mask + np_mask
        
        combined_mask[combined_mask > 0] = 1
        return np.multiply(density_map, combined_mask)

    def mergeDensityMapWithImage(self, image_path: str, density_map: np):
        background_image = Image.open(image_path).convert('RGB')

        density_map_normalized = density_map / (np.max(density_map) + 1e-8)
        
        density_map_colored = plt.cm.jet(density_map_normalized)[:, :, :3]
        density_map_rgba = np.zeros((density_map.shape[0], density_map.shape[1], 4), dtype=np.uint8)
        density_map_rgba[..., :3] = density_map_colored * 255
        
        alpha_channel = density_map_normalized * self.density_map_intensity
        alpha_channel[alpha_channel > self.alpha_channel] = self.alpha_channel
        density_map_rgba[..., 3] = alpha_channel

        density_map_image = Image.fromarray(density_map_rgba)
        width, height = background_image.size
        density_map_image = density_map_image.resize((width, height), Image.BILINEAR)

        combined_image = Image.alpha_composite(background_image.convert('RGBA'), density_map_image)
        buffer = io.BytesIO()
        combined_image.save(buffer, format='PNG')
        return buffer.getvalue()
