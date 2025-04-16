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
    
    def __init__(self):
        self.transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])    
    
    def predict(self, image_path) -> PredictionDTO:
        self.prepareModel()
        inputs = self.processImage(image_path)

        with torch.set_grad_enabled(False):
            outputs = self.model(inputs)
            density_map = outputs.squeeze().cpu().numpy()
            
        merged_image = self.mergeDensityMapWithImage(image_path, density_map)
        
        return PredictionDTO(
            crowd_count= torch.sum(outputs).item(),
            time_stamp= timezone.now(),
            img_predict_content=merged_image
        )
    
    def mergeDensityMapWithImage(self, image_path, density_map):
        background_image = Image.open(image_path).convert('RGB')

        density_map_normalized = density_map / np.max(density_map)
        
        density_map_colored = plt.cm.jet(density_map_normalized)[:, :, :3]
        density_map_rgba = np.zeros((density_map.shape[0], density_map.shape[1], 4), dtype=np.uint8)
        density_map_rgba[..., :3] = density_map_colored * 255
        
        alpha_channel = density_map_normalized * 250
        alpha_channel[alpha_channel>75] = 75
        density_map_rgba[..., 3] = alpha_channel

        density_map_image = Image.fromarray(density_map_rgba)
        width, height = background_image.size
        density_map_image = density_map_image.resize((width, height), Image.BILINEAR)

        combined_image = Image.alpha_composite(background_image.convert('RGBA'), density_map_image)
        buffer = io.BytesIO()
        combined_image.save(buffer, format='PNG')
        return buffer.getvalue()
        
    def prepareModel(self):
        self.model = vgg19()
        device = torch.device(self.device)
        self.model.to(device)
        self.model.load_state_dict(torch.load(os.path.abspath(self.weigth_path), device))

    def processImage(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img = self.transformer(img)
        img = img.unsqueeze(0)
        return img.to(self.device)
