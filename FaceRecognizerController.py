from fastapi import FastAPI
from BO.EncodeKnownFaceIn import EncodeKnownFaceIn
from BO.RecognitionIn import RecognitionIn
from BO.ValidateIn import ValidateIn
from service.FaceRecognizerService import FaceRecognizerService






""" ********************************** Commande pour démarrer l'application ********************************** """

# uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 8009
# uvicorn FaceRecognizerController:app --reload --workers 1 --host 0.0.0.0 --port 8009






""" ********************************** Méthodes ********************************** """


""" Chargement de l'Api """
app = FastAPI()




@app.get("/ping")
async def pong():
    """ Api de test """
    return {"ping": "pong!"}




@app.post("/encode-and-train-dataset", status_code=200)
def encode_known_faces(payload: EncodeKnownFaceIn):
    """ Api d'encodage des images et d'entrainement """
    print(" Logs d'encodage : ")
    print(payload.model)
    # Instanciation du service :
    face_recognizer_service_instance = FaceRecognizerService()
    # Encodage des photos et entrainement du modèle :
    face_recognizer_service_instance.encode_known_faces(payload.model)




@app.post("/recognize-face", status_code=200)
def recognize_face(payload: RecognitionIn):
    """ Api de reconnaissance faciale """
    print(" Logs de Reconnaissance Faciale : ")
    print(payload.model)
    print(payload.image_location)
    # Instanciation du service :
    face_recognizer_service_instance = FaceRecognizerService()
    # Encodage de la photo à identifier :
    face_recognizer_service_instance.encode_known_faces(payload.model)
    # Reconnaissance :
    face_recognizer_service_instance.recognize_faces(payload.image_location, payload.model)




""" Api de validation Modèle """
@app.post("/recognize-face-test", status_code=200)
def validate(payload: ValidateIn):
    """ Api de validation Modèle """
    print(" Logs de validation : ")
    print(payload.model)
    # Instanciation du service :
    face_recognizer_service_instance = FaceRecognizerService()
    # Validation de l'entrainement :
    face_recognizer_service_instance.validate(payload.model)

