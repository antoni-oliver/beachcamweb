import logging
import os
from django.utils import timezone
from beachcams.utils import data_dir

log = logging.getLogger(__name__)

class EntryPoint:
    
    def __init__(self, provider: str, name: str,  url: str,  handler: object):
        self.provider = provider
        self.name = name
        self.url = url
        self.handler = handler
        
    def run(self):
        return self.handler.run(self)
    
    def generate_filename(self):
        folder = f'{data_dir()}/{self.provider}/{self.name}'
        os.makedirs(folder, exist_ok=True)
        filename = f'{folder}/{timezone.now().strftime("%Y%m%d%H%M%S")}'
        return filename
    