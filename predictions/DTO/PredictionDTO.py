from dataclasses import dataclass

from datetime import datetime

@dataclass
class PredictionDTO:
    crowd_count: int
    img_predict_content: bytes
    time_stamp: datetime