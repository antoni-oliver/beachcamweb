from predictions.DTO.PredictionDTO import PredictionDTO

class PredictorInterface:

    def predict(self, image_path, mask_paths = []) -> PredictionDTO:
        pass