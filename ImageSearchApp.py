import streamlit as st
from pymilvus import MilvusClient, DataType
import numpy as np
from PIL import Image
import feature_extractor as fe
import milvus_helper
from datetime import datetime
from pathlib import Path
import os
from urllib.request import urlopen
from io import BytesIO
import pandas as pd


target_size = 768
TOP_K = 10
DATA_FOLDER = [
        Path('/home/public/image_search/image_data/test_image/bedroom'),
        Path('/home/public/image_search/image_data/test_image/downlight'),
        Path('/home/public/image_search/image_data/test_image/guide_board'),
        Path('/home/public/image_search/image_data/test_image/kitchen'),
        Path('/home/public/image_search/image_data/test_image/livingroom'),
        Path('/home/public/image_search/image_data/test_image/mall'),
        Path('/home/public/image_search/image_data/test_image/office'),
        Path('/home/public/image_search/image_data/test_image/outdoor_desk'),
        Path('/home/public/image_search/image_data/test_image/porch_cabinet'),
        Path('/home/public/image_search/image_data/test_image/sofa'),
        Path('/home/public/image_search/image_data/test_image/test_baseimage'),
        Path('/home/public/image_search/image_data/test_image/toilet'),
        Path('/home/public/image_search/image_data/test_image/western_restaurant'),
        Path('/home/public/image_search/image_data/3d_su_2000'),
        Path('/home/public/image_search/image_data/temp_image'),
    ]

@st.cache_resource
def get_client():
    client = MilvusClient(uri="/home/public/image_search/database/src_images_v2.db")
    return client
MILVUS_CLIENT = get_client()


def GenSimilar(img, collection_name, metric_type):  #Run search
    if collection_name == 'Histogram_Features_Collection':
        query = fe.hist_features(img)
    elif collection_name == 'image_sketch':
        query = fe.sketch_features(img)
    elif collection_name == 'image_depth':
        query = fe.depth_features(img)
    elif collection_name == 'image_seg':
        query = fe.segmentation_features(img)
    else:
        query = fe.original_image_features(img)
    
    print('*********** Query **********', query.shape)
    # print('*********** Features *********', features.shape)
    items_k = milvus_helper.query_vector(client=MILVUS_CLIENT, collection_name=collection_name, topk=TOP_K, 
                               query_vector=query, metric_type=metric_type, output_fields=['image_name'])[0]
    print(pd.DataFrame(items_k))
    scores = []
    for item in items_k:
        scores.append((item['distance'], item['entity']['image_name']))
    
    columns = st.columns(3)
    for i, (score, image_path) in enumerate(scores):
        print(f'{i}: {score}, {image_path}')
        folder_idx = 0
        if image_path.startswith('sketch_'):
            image_path = image_path[7:]

        col = columns[i % 3]
        with col:
            while folder_idx < len(DATA_FOLDER):
                try:
                    image = Image.open(DATA_FOLDER[folder_idx] / image_path)
                    width, height = image.size
                    print(DATA_FOLDER[folder_idx] / image_path)
                    print(width, height)
    
                    if width > height:
                        scale_factor = target_size / width  
                    else:
                        scale_factor = target_size / height 
                    
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    
                    resized_img = image.resize((new_width, new_height))
                except OSError as e:
                    folder_idx += 1
                else:
                    st.image(resized_img, caption=f'Score is {score},\n{DATA_FOLDER[folder_idx] / image_path}', use_container_width=True)
                    break


#Load the image from the path
image_path = "./banner.png"
banner = Image.open(image_path)
st.image(banner, use_container_width=True)

#Page Config
st.title('**:blue[Image Search]**')

#Take User Input
option = st.selectbox('How would you like to search?',('image_db', 'Histogram_Features_Collection', 'image_sketch', 'image_depth', 'image_seg', '搜原图', '搜结构相似', '搜色彩相似'))
option2 = st.selectbox('How would you like to search?',('COSINE', 'IP'))

if option in ['image_db', 'Histogram_Features_Collection', 'image_sketch', 'image_depth', 'image_seg']:
    file = st.file_uploader(label='Upload image to search', type=['jpg','png','jpeg'], key='FileInput')
    if file:
        st.image(file, caption='Uploaded image', use_container_width=True)
        img = Image.open(file)  # PIL image
        width, height = img.size
        
        if width > height:
            scale_factor = target_size / width  
        else:
            scale_factor = target_size / height 
                        
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        resized_img = img.resize((new_width, new_height))
        GenSimilar(resized_img, collection_name=option, metric_type=option2)

else:
    st.markdown('敬请期待')







    

    
