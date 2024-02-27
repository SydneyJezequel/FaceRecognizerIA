from fastapi import FastAPI
from FaceRecognizerBO import EncodeKnownFaceIn, RecognitionIn, ValidateIn
from FaceRecognizerService import FaceRecognizerService






""" ********************************** Commande pour démarrer l'application ********************************** """

# uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 8009
# uvicorn FaceRecognizerController:app --reload --workers 1 --host 0.0.0.0 --port 8009





""" ********************************** Méthodes ********************************** """


""" Chargement de l'Api """
app = FastAPI()




""" Api de test """
@app.get("/ping")
async def pong():
    return {"ping": "pong!"}




""" Api d'encodage des images et d'entrainement """
@app.post("/encode-and-train-dataset", status_code=200)
def encode_known_faces(payload: EncodeKnownFaceIn):
    print(" ****** TEST ENCODAGE / ENTRAINEMENT ****** ")
    print(payload.model)
    print(" ****** TEST ENCODAGE / ENTRAINEMENT ****** ")
    # Instanciation du service :
    face_recognizer_service_instance = FaceRecognizerService()
    # Encodage des photos et entrainement du modèle :
    face_recognizer_service_instance.encode_known_faces(payload.model)




""" Api de reconnaissance faciale """
@app.post("/recognize-face", status_code=200)
def recognize_face(payload: RecognitionIn):
    print(" ****** TEST RECONNAISSANCE FACIALE ****** ")
    print(payload.model)
    print(payload.image_location)
    print(" ****** TEST RECONNAISSANCE FACIALE ****** ")
    # Instanciation du service :
    face_recognizer_service_instance = FaceRecognizerService()
    # Encodage de la photo à identifier :
    face_recognizer_service_instance.encode_known_faces(payload.model)
    # Reconnaissance :
    face_recognizer_service_instance.recognize_faces(payload.image_location, payload.model)




""" Api de validation Modèle """
@app.post("/recognize-face-test", status_code=200)
def validate(payload: ValidateIn):
    print(" ****** TEST VALIDATION ****** ")
    print(payload.model)
    print(" ****** TEST VALIDATION ****** ")
    # Instanciation du service :
    face_recognizer_service_instance = FaceRecognizerService()
    # Validation de l'entrainement :
    face_recognizer_service_instance.validate(payload.model)



