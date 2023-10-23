from deepface import DeepFace
import pickle
import os
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import cv2
import streamlit as st
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn import MTCNN
import os
from src.utils.al_utils import read_yaml,create_directory

config = read_yaml('config/config.yaml')
params = read_yaml('params.yaml')

artifacts = config['artifacts']
artifacts_dir = artifacts['artifacts_dir']

#creating upload folder for the user to upload images
upload_image_dir = artifacts['upload_image_dir']
upload_dir_path = os.path.join(artifacts_dir,upload_image_dir)

#pickle format_data dir
pickle_format_data_dir = artifacts['pickle_format_data_dir']
img_pickle_file_name = artifacts['img_pickle_file_name']

raw_local_dir_path = os.path.join(artifacts_dir, pickle_format_data_dir)
pickle_file = os.path.join(raw_local_dir_path, img_pickle_file_name)

#feature path
feature_extraction_dir = artifacts['feature_extraction_dir']
extracted_features_name = artifacts['extracted_features_name']

feature_extraction_path = os.path.join(artifacts_dir, feature_extraction_dir)

features_name = os.path.join(feature_extraction_path, extracted_features_name)

detector = MTCNN()
model = DeepFace.build_model("VGG-Face")
filenames = pickle.load(open(pickle_file,'rb'))
feature_list = pickle.load(open(features_name,'rb'))


def extract_feature(img_path,model,detector):
   img =  cv2.imread(img_path)
   result = detector.detect_faces(img)

   if result:
    x,y,width,height = result[0]['box']

    face = img[y:y+height, x:x+width]

    #extract features
    image = Image.fromarray(face)
    image = image.resize((224,224))

    face_array = np.asarray(image)
    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array,axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result
   else:
       return None


def save_upload_image(upload_image):

    try:
        create_directory(dirs=[upload_dir_path])
        with open(os.path.join(upload_dir_path,upload_image.name), 'wb') as f:
            f.write(upload_image.getbuffer())
        
        return True
    except:
        return False

def recommend(feature_list,fetures):
    similarity=[]

    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1,-1),feature_list[i].reshape(1,-1))[0][0])
    indexpos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]

    return indexpos



#streamlit
st.title('What Celebrity you look like?!')

upload_image = st.file_uploader('Choose an Image', type=['jpg', 'jpeg', 'png'])


if upload_image is not None:
    #Saving the image
    if save_upload_image(upload_image):
        #load the image
        load_image = Image.open(upload_image)
        st.image(load_image, caption='Uploaded Image', use_column_width=True)
        
        #extract feature
        features= extract_feature(os.path.join(upload_dir_path,upload_image.name),model,detector)

        #recommend
        if features is not None:
            indexpos= recommend(feature_list,features)

            predictor_actor = " ".join(filenames[indexpos].split('/')[1].split('_'))

            col1,col2 = st.columns(2)

            with col1:
                st.header('You Uploaded')
                st.image(load_image)
            
            with col2:
                st.header('You look like '+ predictor_actor)
                st.image(filenames[indexpos],width=300)
        else:
            st.error('No face detected try different Image')
