from beachcams.classes.EntryPoint import EntryPoint
from beachcams.classes.ImageProvider import ImageProvider
from PIL import Image
from io import BytesIO
import requests

class StaticImageProvider(ImageProvider):
    
    def run(self, entryPoint: EntryPoint) -> str:
        filename = f'{entryPoint.generate_filename()}.jpg'
        response = requests.get(entryPoint.url)
        img = Image.open(BytesIO(response.content))
        img.save(filename)
        return filename