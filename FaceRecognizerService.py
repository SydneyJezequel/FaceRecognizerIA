from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw
import face_recognition
import pickle






class FaceRecognizerService:
    """ Service de Reconnaissance Faciale  """





    """ ******************************** Attributs ******************************** """

    """ Fichier contenant les images encodées """
    DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

    """ Fichier contenant les images encodées """
    DEFAULT_IDENTIFY_PATH = Path("identifyFace/identifyFace.jpg")

    """ Couleur de la Bounding Box (rectangle qui contient les visages) """
    BOUNDING_BOX_COLOR = "blue"

    """ Couleur du texte """
    TEXT_COLOR = "white"

    """ Création du répertoire "training" """
    Path("training").mkdir(exist_ok=True)

    """ Création du répertoire "output" """
    Path("output").mkdir(exist_ok=True)

    """ Création du répertoire "validation" """
    Path("validation").mkdir(exist_ok=True)





    """ ******************************** Constructeur ******************************** """

    _instance = None

    def __new__(cls):
        """ Constructeur """
        if cls._instance is None:
            cls._instance = super(FaceRecognizerService, cls).__new__(cls)
        return cls._instance





    """ ******************************** Méthodes ******************************** """

    def encode_known_faces(self, model: str) -> None:
        """ Méthode qui charges les photos, identifie et enregistre les visages dans un dictionnaire """
        names = []
        encodings = []
        encodings_location: Path = self.DEFAULT_ENCODINGS_PATH
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



    def recognize_faces(self, image_location: str, model:str) -> None:
        """ Méthode pour reconnaitre les visages sur une image donnée """
        # Récupération du path des images encodées :
        encodings_location: Path = self.DEFAULT_ENCODINGS_PATH
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
                name = self._recognize_face(unknown_encoding, loaded_encodings)
                # Attribution d'un nom par défaut si le visage n'est pas reconnu :
                if not name:
                    name = "Unknown"
                print(name, bounding_box)
                self._display_face(draw, bounding_box, name)
        # Suppression de l'objet ImageDraw :
        del draw
        # Affichage de l'image :
        pillow_image.show()



    def _recognize_face(self, unknown_encoding, loaded_encodings):
        """ Méthode de reconnaissance d'un visage """
        """ Compare l'encodage du visage inconnu avec la liste des visages encodés lors de l'entrainement. """
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



    def _display_face(self, draw, bounding_box, name):
        """" Méthode qui dessine la boite englobante autour des visages """
        # Dessin de la boîte englobante autour du visage :
        top, right, bottom, left = bounding_box
        draw.rectangle(((left, top), (right, bottom)), outline=self.BOUNDING_BOX_COLOR)
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



    def validate(self, model: str = "hog"):
        """ Méthode de validation du modèle """
        # Boucle sur les fichiers dans le répertoire validation :
        for filepath in Path("validation").rglob("*"):
            if filepath.is_file():
                # Appel à la méthode recognize_faces :
                self.recognize_faces(
                    image_location=str(filepath.absolute()), model=model
                )




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

