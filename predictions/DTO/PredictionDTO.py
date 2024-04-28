from dataclasses import dataclass

@dataclass
class PredictionDTO:
    crowd_count: int
    img_predict_url: str
    img_predict_content: bytes