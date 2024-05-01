from predictions.DTO.PredictionDTO import PredictionDTO
from predictions.interfaces.PredictorInterface import PredictorInterface
import torch
from bayesian_stuff.vgg import vgg19
from torchvision import transforms
from PIL import Image
from django.utils import timezone

class BayesianPredictor(PredictorInterface):
    
    transformer = None
    model = None
    device = "cpu"
    weigth_path = ""
    
    def __init__(self):
        self.transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])    
    
    def prepareModel(self):
        self.model = vgg19()
        device = torch.device(self.device)
        self.model.to(device)
        self.model.load_state_dict(torch.load("./bayesian_stuff/best.pth", device))

    def processImage(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img = self.transformer(img)
        return img
    
    def predict(self, image_path) -> PredictionDTO:
        self.prepareModel()
        inputs = self.processImage(image_path)

        with torch.set_grad_enabled(False):
            outputs = self.model(inputs)
            density_map = outputs.squeeze().cpu().numpy()
        
        return PredictionDTO(
            crowd_count= 12,
            time_stamp= timezone.now(),
            #TODO devolver el contenido de la imagen en algo que se pueda manejar
            img_predict_content="contenido de prueba de la imagen"
        )
        