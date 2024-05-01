from predictions.DTO.PredictionDTO import PredictionDTO

class PredictorInterface:

    def predict(self, img) -> PredictionDTO:
        pass