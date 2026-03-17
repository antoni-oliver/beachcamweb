from predictions.interfaces.PredictorInterface import PredictorInterface
from predictions.DTO.PredictionDTO import PredictionDTO

class P2PPredictor(PredictorInterface):
    
    #TODO implementar la evaluacion del modelo y el mergeo de las imagenes 
    # (p2p se implementará si da tiempo)
    def predict(self, img_path) -> PredictionDTO:
        return PredictionDTO(
            crowd_count= 12,
            img_predict_content="contenido de prueba de la imagen"
        )
