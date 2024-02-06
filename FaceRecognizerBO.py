from pydantic import BaseModel






""" ********************************** Objets en paramètres des controllers ********************************** """


""" Paramètre du controller encode_known_faces()  """
class EncodeKnownFaceIn(BaseModel):
    model: str




""" Paramètre du controller recognize_face() """
class RecognitionIn(BaseModel):
    model: str
    image_location: str




""" Paramètre du controller validate() """
class ValidateIn(BaseModel):
    model: str



