import streamlit as st
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from qdrant_client import QdrantClient, models
import numpy as np
from PIL import Image
import cv2
import requests
from pathlib import Path
 
# Function to preprocess image and extract features
def extract_features(model, img):
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

# Function to search for similar images using feature vectors
def search_by_vector(feature_vector):
    collection_name = "CIFAR"
    qdrant_client = QdrantClient(
    url= st.secrets["db_url"],
    api_key=st.secrets["db_api_key"] ,
    )
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=feature_vector,
        query_filter=None, 
        limit=10  
    )
    return search_result

# Streamlit app
def main():
    st.title('Similarity Image Search')
    st.markdown(
            "- Built with Streamlit and Qdrant VectorDB.\n"\
            "- The app allows users to search for similar images in the CIFAR dataset using a provided query image.\n" \
            "- You can search airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck."
        ) 
    # Load ResNet50 model
    base_model = ResNet50(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

    # Image upload and feature extraction
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    examples=list(Path("sample").glob("*.jpg"))

    # Display "Sample" on top
    st.write("### Sample Images")

    # Display sample images and their names
    num_cols = 3
    num_images = len(examples)
    num_rows = (num_images + num_cols - 1) // num_cols

    for row in range(num_rows):
        col1, col2, col3 = st.columns(3)
        for col, image_path in zip((col1, col2, col3), examples[row * num_cols: (row + 1) * num_cols]):
            with col:
                st.image(str(image_path), width=200,caption=image_path.stem)
                #st.text()  # Display filename without extension

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        image = Image.open(uploaded_file)
        # Convert the PIL Image to a NumPy array
        image = np.array(image, dtype=np.uint8)

        # Extract feature vector from image
        feature_vector = extract_features(model, image)

        # Search using feature vector
        search_results = search_by_vector(feature_vector)
        
        if search_results:
            st.write("Search Results:")
            # Collect all image paths and labels
            images = []
            captions = []
            for result in search_results:
                image_path = "images1/" + result.payload['image_id'] + ".jpg"
                images.append(image_path)
                captions.append(result.payload['image_id'])
            
            # Display images in a grid
            st.image(images, caption=captions, width=120)

        else:
            st.write("No similar images found.")

if __name__ == '__main__':
    main()
