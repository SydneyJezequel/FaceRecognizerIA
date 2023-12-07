from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from FaceRecognizerService import validate as modelValidation, recognize_faces as face_recognition, encode_known_faces as known_faces_encoding
from pathlib import Path








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
# ********************************************************* Test  ******************************************************** #
# ****************************************************************************************************************************** #



# Test 1 :
@app.get("/test-1")
async def pong():
    return {"test-1": "OK"}



# Test 2 :
@app.get("/test-2")
async def pong():
    return {"test-2": "OK"}



# Test 3 :
# Objet en entrée :
class StockIn(BaseModel):
    message_entree: str

# Objet en sortie :
class StockOut(StockIn):
    message_recu: str
    message_renvoye: str

@app.post("/test-3", response_model=StockOut, status_code=200)
def get_prediction(payload: StockIn):

    # Affichage du message reçu :
    print("Message reçu :", payload.message_entree)

    # Renvoi du résultat :
    response_object = StockOut(
        message_entree=payload.message_entree,
        message_recu=payload.message_entree,
        message_renvoye="Coucou, cette fonction s'est bien exécutée :) "
    )
    return response_object





# ****************************************************************************************************************************** #
# ********************************************************* Test  ******************************************************** #
# ****************************************************************************************************************************** #









































"""

# ****************************************************************************************************************************** #
# ******************************************* Appel des méthodes ****************************************** #
# ****************************************************************************************************************************** #




# Méthode d'encodage (L'entrainement est fait dans cette méthode) :
X encode_known_faces(model: str, encodings_location: Path = DEFAULT_ENCODINGS_PATH)

# Méthode de reconnaissance d'un visage (Appelle les méthodes _recognize_face() et _display_face()) :
X recognize_faces(image_location: str, model, encodings_location: Path = DEFAULT_ENCODINGS_PATH)

# Méthode de validation du modèle :
X validate(model: str = "hog")

"""




















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
# ********************************************************* Méthode d'encodage des images ******************************************************** #
# ****************************************************************************************************************************** #











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
# ********************************************************* Méthode de Reconnaissance Faciale ******************************************************** #
# ****************************************************************************************************************************** #











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

# ****************************************************************************************************************************** #
# ********************************************************* Méthode validation Modèle ******************************************************** #
# ****************************************************************************************************************************** #







































"""
# ******************************************************************************************************** #
# ******************************************* Types d'attributs ****************************************** #
# ******************************************************************************************************** #




=================================================================================
LISTE DES METHODES A EXPOSER VIA UNE API :



—————————————————————————————————————————————
==> ENCODAGE DES PHOTOS DE VISAGE :
encode_known_faces(model: str, encodings_location: Path = DEFAULT_ENCODINGS_PATH)

ATTRIBUTS :
-String model
-Path (équivalent dans Python ==> classe « Path » dans java.nio.file)



—————————————————————————————————————————————
==> RECONNAISSANCE FACIALE :
recognize_faces(image_location: str, model, encodings_location: Path = DEFAULT_ENCODINGS_PATH)

ATTRIBUTS :
-image_location: str
-String model
-Path —> Equivalent dans Java de la classe « Path » Python : « java.nio.file ».



—————————————————————————————————————————————
==> RECONNAISSANCE FACIALE (2) :
_recognize_face(unknown_encoding, loaded_encodings)

ATTRIBUTS :
unknown_encoding —> Pas besoin de définir ces attributs en Java. Cet attribut est directement généré dans la méthode recognize_faces(). Elle n’est pas définit dans le Backend SpringBoot.
loaded_encodings (dictionnaire)  —> Equivalent dans Java de la classe « Path » Python : « Map<String, Object>».



—————————————————————————————————————————————
==> AFFICHAGE DES VISAGES :
_display_face(draw, bounding_box, name)

ATTRIBUTS :
Pas besoin de définir ces attributs en Java (draw, bounding_box, name) car cette méthode sera appelée
à partir de la méthode _recognize_face. Elle n’a donc pas de paramètres générées depuis le Backend SpringBoot.



—————————————————————————————————————————————
==> VALIDATION DU MODELE :
validate(model: str = "hog")

ATTRIBUTS :
String model


=================================================================================
"""






