from django.shortcuts import render
from mtcnn import MTCNN
import cv2
from keras_facenet import FaceNet
from scipy.spatial import distance
import numpy as np
from PIL import Image

# Initialize MTCNN and FaceNet
detector = MTCNN()
model = FaceNet()

def extract_face(image):
    # Check if the image is grayscale
    if len(image.shape) == 2:
        # Convert grayscale to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        # Convert RGBA to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    faces = detector.detect_faces(image)
    if len(faces) == 0:
        return None

    bounding_box = faces[0]['box']
    x, y, w, h = bounding_box
    face_region = image[y:y+h, x:x+w]
    face_region = cv2.resize(face_region, (160, 160))
    return face_region


def compare_faces(request):
    if request.method == 'POST':
        im1 = request.FILES['image1']
        im2 = request.FILES['image2']

        # Open images using PIL
        img1 = Image.open(im1).convert('RGB')
        img2 = Image.open(im2).convert('RGB')

        # Convert images to numpy arrays
        img1 = np.array(img1)
        img2 = np.array(img2)

        # Extract faces directly from color images
        face1 = extract_face(img1)
        face2 = extract_face(img2)

        if face1 is None or face2 is None:
            return render(request, 'api/compare_faces.html', {'error': 'Face not detected'})

        # Extract embeddings using FaceNet
        embeddings1 = model.embeddings([face1])[0]
        embeddings2 = model.embeddings([face2])[0]

        # Normalize embeddings
        embeddings1 = embeddings1 / np.linalg.norm(embeddings1)
        embeddings2 = embeddings2 / np.linalg.norm(embeddings2)

        # Calculate cosine similarity
        similarity = 1 - distance.cosine(embeddings1, embeddings2)
        similarity_percentage = similarity * 100

        return render(request, 'api/compare_faces.html', {
            'similarity': f'{similarity_percentage:.2f}%',
            'message': 'Faces are similar' if similarity_percentage >= 70 else 'Faces are not similar',
        })

    return render(request, 'api/compare_faces.html')




# New view for About page
def about(request):
    return render(request, 'api/about.html')  # This will render an about.html template
