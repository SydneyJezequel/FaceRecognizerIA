L'objectif de ce projet est de manipuler des modèles de reconnaissance faciale.

2 modèles sont mis à disposition : 
- HOG (histogram of oriented gradients)
- CNN (convolutional neural network)

Ce projet rassemble les fonctionnalités suivantes :
- Sélection et entrainement du modèle.
- Validation du modèle.
- Identification d'une image.

Ces fonctionnalités sont mises à disposition via Fast API.

Le Front et le Backend pour manipuler ce modèle sont mis à disposition dans les projets suivants :
- https://github.com/SydneyJezequel/applicationIABackend
- https://github.com/SydneyJezequel/applicationIAFrontend

Commande pour lancer le projet :
uvicorn FaceRecognizerController:app --reload --workers 1 --host 0.0.0.0 --port 8009




# NOTIONS :

URL : https://realpython.com/face-recognition-with-python/#project-overview

# BOITE ENGLOBANTES / BOUNDING BOXES :
Les boîtes englobantes, souvent appelées "bounding boxes" en anglais,
sont des rectangles qui encadrent un objet ou une région d'intérêt dans une image.
Dans le contexte de la reconnaissance d'objets ou de visages,
les boîtes englobantes sont utilisées pour délimiter l'emplacement d'un objet ou
d'un visage détecté dans une image.

# VOTE :
-Qu’est-ce qu’un vote, et qui vote ?
Lorsque vous appelez compare_faces(), votre visage inconnu est comparé
à tous les visages connus pour lesquels vous disposez d'encodages.
Chaque match fait office de vote pour la personne au visage connu. 

