from predictions.interfaces.PredictorStrategy import PredictorStrategy
from predictions.DTO.PredictionDTO import PredictionDTO

class P2PPredictor(PredictorStrategy):
    
    #TODO implementar la evaluacion del modelo y el mergeo de las imagenes 
    # (p2p se implementará si da tiempo)
    def predict(self, img) -> PredictionDTO:
        return PredictionDTO(
            crowd_count= 12,
            img_predict_url="https://upload.wikimedia.org/wikipedia/commons/9/90/Escut_UIB.svg",
            img_predict_content="contenido de prueba de la imagen"
        )
