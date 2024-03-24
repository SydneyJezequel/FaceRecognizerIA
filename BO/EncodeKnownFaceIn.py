from pydantic import BaseModel






class EncodeKnownFaceIn(BaseModel):
    """ Paramètre du controller encode_known_faces() """
    model: str

