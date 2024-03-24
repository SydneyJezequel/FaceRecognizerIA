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

