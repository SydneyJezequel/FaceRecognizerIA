"""
from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw
import face_recognition
import pickle
import argparse








# ************************************************************************************************************
# ************************************************* Attributs ************************************************
# ************************************************************************************************************

# Fichier contenant les images encodées :
DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
# Couleur de la Bounding Box (rectangle qui contient les visages) :
BOUNDING_BOX_COLOR = "blue"
# Couleur du texte :
TEXT_COLOR = "white"








# ************************************************************************************************************
# ************************ Création des répertoires "training", "output" et "validation" ************************
# ************************************************************************************************************
Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)








# *************************************************************************************************************
# ***************************************** Définition des Arguments  *****************************************
# *************************************************************************************************************

# Définition des arguments :
parser = argparse.ArgumentParser(description="Recognize faces in an image")
# Validation :
parser.add_argument("--train", action="store_true", help="Train on input data")
# Entrainement :
parser.add_argument(
    "--validate", action="store_true", help="Validate trained model"
)
# Test :
parser.add_argument(
    "--test", action="store_true", help="Test the model with an unknown image"
)
# Choix du modèle :
parser.add_argument(
    "-m",
    action="store",
    default="hog",
    choices=["hog", "cnn"],
    help="Which model to use for training: hog (CPU), cnn (GPU)",
)

parser.add_argument(
    "-f", action="store", help="Path to an image with an unknown face"
)

# Intégration des arguments dans l'attribut "args" :
args = parser.parse_args()








# *********************************************************************************************************************************************
# ************************ Méthode pour charger les photos, identifier et enregistrer les visages dans un dictionnaire ************************
# *********************************************************************************************************************************************

def encode_known_faces(model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    names = []
    encodings = []

    # Boucle qui parcourt chaque fichier d'entrainement :
    for filepath in Path("training").glob("*/*"):

        # Extraction du nom et du contenu de l'image :
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        # Détection de l'emplacement des visages sur chaque image. Renvoie les 4 coordonnées d'une boite qui recouvre le visage :
        face_locations = face_recognition.face_locations(image, model=model)

        # Encodage de chaque visage détecté :
        face_encodings = face_recognition.face_encodings(image, face_locations)

        # AJout des noms et visages encodés aux listes "names" et "encodings" :
        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    # Enregistrement de l'encodage des images et de leurs noms dans un dictionnaire et sauvegarde dans un fichier "pickle" :
    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

# Générer un fichier "encodings.pkl" :
encode_known_faces()








# *************************************************************************************************************
# ************************* Méthode pour reconnaitre les visages sur une image donnée *************************
# *************************************************************************************************************

def recognize_faces(image_location: str, model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    # Chargement des images encodées dans le fichier pickle :
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    # Chargement de l'image à partir du fichier spécifié :
    input_image = face_recognition.load_image_file(image_location)

    # Détection des visages dans l'image et leur encodage :
    input_face_locations = face_recognition.face_locations(input_image, model=model)

    # Encodage des visages détectés dans l'image d'entrée :
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

    # Création d'un objet Image :
    pillow_image = Image.fromarray(input_image)

    # Création d'un objet ImageDraw associé à l'objet Image :
    draw = ImageDraw.Draw(pillow_image)

    # Comparaison des visages encodés enregistrés avec des visages encodés inconnus :
    for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
            # Appel de la fonction _recognize_face pour reconnaître le visage :
            name = _recognize_face(unknown_encoding, loaded_encodings)
            # Attribution d'un nom par défaut si le visage n'est pas reconnu :
            if not name:
                name = "Unknown"
            print(name, bounding_box)
            _display_face(draw, bounding_box, name)
    # Suppression de l'objet ImageDraw :
    del draw
    # Affichage de l'image :
    pillow_image.show()








# *****************************************************************************************************************************************************************************************************************************
# ************************** Méthode de reconnaissance d'un visage en comparant son encodage (encodage du visage inconnu) avec une liste d'encodages de visages connus préalablement enregistrés ******************************
# *****************************************************************************************************************************************************************************************************************************

def _recognize_face(unknown_encoding, loaded_encodings):

    # Compare le visage inconnu passé en paramtère avec chaque visage précédement enregistré :
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    # Comptage des votes pour chaque correspondance visage encodé / visage inconnu :
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    # Renvoie du visage qui a le plus de votes :
    if votes:
        return votes.most_common(1)[0][0]

# Tout visage que le script localise mais ne peut pas identifier
# recognize_faces("unknown.jpg")








# *********************************************************************************************************************************************
# **************************************** Méthode qui dessine une boite englobante autour des visages ****************************************
# *********************************************************************************************************************************************

def _display_face(draw, bounding_box, name):

    # Dessin de la boîte englobante autour du visage :
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)

    # Identifier les coordoonées de la boite englobante :
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name
    )
    # Dessin de la boite :
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill="blue",
        outline="blue",
    )
    # Dessin du texte :
    draw.text(
        (text_left, text_top),
        name,
        fill="white",
    )








# *************************************************************************************************************
# ************************************** Méthode pour valider le modèle ***************************************
# *************************************************************************************************************

def validate(model: str = "hog"):
    # Boucle sur les fichiers dans le répertoire validation :
    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            # Appel à la méthode recognize_faces :
            recognize_faces(
                image_location=str(filepath.absolute()), model=model
            )
# Remove validate :
validate()






# *************************************************************************************************************
# ********************** Code pour appeler la méthode souhaitée en fonction des arguments *********************
# *************************************************************************************************************

if __name__ == "__main__":
    if args.train:
        encode_known_faces(model=args.m)
    if args.validate:
        validate(model=args.m)
    if args.test:
        recognize_faces(image_location=args.f, model=args.m)





"""










































































# Detector
'''
URL :
https://realpython.com/face-recognition-with-python/#project-overview
'''




# BOITE ENGLOBANTES / BOUNDING BOXES :
"""
Les boîtes englobantes, souvent appelées "bounding boxes" en anglais,
sont des rectangles qui encadrent un objet ou une région d'intérêt dans une image.
Dans le contexte de la reconnaissance d'objets ou de visages,
les boîtes englobantes sont utilisées pour délimiter l'emplacement d'un objet ou
d'un visage détecté dans une image.
"""




# VOTE :
"""
-Qu’est-ce qu’un vote, et qui vote ?
Lorsque vous appelez compare_faces(), votre visage inconnu est comparé
à tous les visages connus pour lesquels vous disposez d'encodages.
Chaque match fait office de vote pour la personne au visage connu. 
"""






