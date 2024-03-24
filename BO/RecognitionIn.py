from pydantic import BaseModel






class RecognitionIn(BaseModel):
    """ Paramètre du controller recognize_face() """
    model: str
    image_location: str

