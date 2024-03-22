from pydantic import BaseModel





""" ********************************** Objets en paramètres des controllers ********************************** """


class EncodeKnownFaceIn(BaseModel):
    """ Paramètre du controller encode_known_faces() """
    model: str




class RecognitionIn(BaseModel):
    """ Paramètre du controller recognize_face() """
    model: str
    image_location: str




class ValidateIn(BaseModel):
    """ Paramètre du controller validate() """
    model: str

