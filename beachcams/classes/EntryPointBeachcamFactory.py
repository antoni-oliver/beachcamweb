from data.models import BeachCam
from beachcams.classes.EntryPoint import EntryPoint
from beachcams.classes.YoutubeImageProvider import YoutubeImageProvider
from beachcams.classes.StaticImageProvider import StaticImageProvider
from beachcams.classes.ImageProvider import ImageProvider

class EntryPointBeachcamFactory:
    
    @staticmethod
    def createEntryPoint(beachcam: BeachCam) -> EntryPoint:
        return EntryPoint(
            provider= beachcam.prefered_image_provider,
            name= beachcam.beach_name,
            url=beachcam.getPreferedUrl(),
            handler=EntryPointBeachcamFactory.getHandler(beachcam.prefered_image_provider)
        )
        
    @staticmethod
    def getHandler(provider: str) -> ImageProvider:
        match provider:
            case "seemallorca":
                return StaticImageProvider()
            case "youtube":
                return YoutubeImageProvider()