from pydantic import BaseModel






class ValidateIn(BaseModel):
    """ Paramètre du controller validate() """
    model: str

