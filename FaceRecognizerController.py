from fastapi import FastAPI
from pydantic import BaseModel
from FaceRecognizerService import validate as modelValidation, recognize_faces as face_recognition, encode_known_faces as known_faces_encoding









# ****************************************************************************************************************************** #
# ******************************************** Commande pour démarrer l'application ******************************************** #
# ****************************************************************************************************************************** #

# uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 8008
# uvicorn FaceRecognizerController:app --reload --workers 1 --host 0.0.0.0 --port 8008








# ****************************************************************************************************************************** #
# **************************************************** Chargement de l'Api ***************************************************** #
# ****************************************************************************************************************************** #

app = FastAPI()








# ****************************************************************************************************************************** #
# ******************************************************** Api de test ********************************************************* #
# ****************************************************************************************************************************** #

@app.get("/ping")
async def pong():
    return {"ping": "pong!"}













# ****************************************************************************************************************************** #
# ********************************************************* Méthode d'encodage des images ******************************************************** #
# ****************************************************************************************************************************** #

# Objet en entrée :
class EncodeKnownFaceIn(BaseModel):
    model: str

# Api de reconnaissance faciale :
@app.post("/encode-and-train-dataset", status_code=200)
def encode_known_faces(payload: EncodeKnownFaceIn):
    print(" ****** TEST ENCODAGE / ENTRAINEMENT ****** ")
    print(payload.model)
    print(" ****** TEST ENCODAGE / ENTRAINEMENT ****** ")
    known_faces_encoding(payload.model)











# ****************************************************************************************************************************** #
# ********************************************************* Méthode de Reconnaissance Faciale ******************************************************** #
# ****************************************************************************************************************************** #

# Objet en entrée :
class RecognitionIn(BaseModel):
    model: str
    image_location: str

# Api de reconnaissance faciale :
@app.post("/recognize-face", status_code=200)
def recognize_face(payload: RecognitionIn):
    print(" ****** TEST RECONNAISSANCE FACIALE ****** ")
    print(payload.model)
    print(payload.image_location)
    print(" ****** TEST RECONNAISSANCE FACIALE ****** ")
    known_faces_encoding(payload.model)
    face_recognition(payload.image_location, payload.model)











# ****************************************************************************************************************************** #
# ********************************************************* Méthode validation Modèle ******************************************************** #
# ****************************************************************************************************************************** #

# Objet en entrée :
class ValidateIn(BaseModel):
    model: str

@app.post("/recognize-face-test", status_code=200)
def validate(payload: ValidateIn):
    print(" ****** TEST VALIDATION ****** ")
    print(payload.model)
    print(" ****** TEST VALIDATION ****** ")
    modelValidation(payload.model)






