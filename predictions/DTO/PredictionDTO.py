from dataclasses import dataclass
from datetime import datetime
import base64

@dataclass
class PredictionDTO:
    crowd_count: int
    img_predict_content: bytes
    time_stamp: datetime

    def to_dict(self):
        return {
            'crowd_count': self.crowd_count,
            'img_predict_content': base64.b64encode(self.img_predict_content.read()).decode('utf-8'),
            'time_stamp': self.time_stamp.isoformat() 
        }