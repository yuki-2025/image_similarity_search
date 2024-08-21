import numpy as np
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from qdrant_client import QdrantClient, models
import os
import shutil

def extract_features(model, img):
    """
    Preprocess the image and extract features using the given model.
    """
    # Resize and preprocess the image
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Extract features
    features = model.predict(img_array)
    return features.flatten()

def main():
    # Initialize Qdrant client
    # client = QdrantClient(url="http://localhost:6333")
    client = QdrantClient(
    url=  ["db_url"],
    api_key= ["db_api_key"] ,
    )
    collection_name = "CIFAR"
    images_dir = "images1"

    # Recreate the collection in Qdrant
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=2048, distance=models.Distance.COSINE),
    )

    # Load the pretrained ResNet50 model
    base_model = ResNet50(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

    # Load CIFAR-10 dataset
    (x_train, y_train), _ = cifar10.load_data()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    y_train = y_train.flatten()
    train_labels = [class_names[label] for label in y_train]

    # Create a directory to store the images
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
    os.makedirs(images_dir)

    # Process and upload each image
    for index, (image, label) in enumerate(zip(x_train, train_labels)):
        image_id = f"{label}_{index}"
        cv2.imwrite(os.path.join(images_dir, image_id + ".jpg"), image)
        feature_vector = extract_features(model, image)
        
        # Insert feature vector and metadata into Qdrant
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=index,
                    payload={"label": label, "image_id": image_id},
                    vector=feature_vector,
                ),
            ],
        )
        print(f"Index: {index}, Label: {label}")

if __name__ == "__main__":
    main()
