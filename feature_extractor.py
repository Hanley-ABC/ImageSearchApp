import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio.v3 as iio
import requests
import base64
from io import BytesIO
import clip
import open_clip
import torch
import torch.nn.functional as F
import streamlit as st
Image.MAX_IMAGE_PIXELS = None


@st.cache_resource
class Image2Vec_v1(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.model, self.preprocess = clip.load('RN50x4', self.device)

    def forward(self, image):
        with torch.no_grad():
            inputs = self.preprocess(image).unsqueeze(0).to(self.device)
            outputs = self.model.encode_image(inputs).squeeze(0).cpu().numpy()

            return outputs


@st.cache_resource
class Image2Vec_v2(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-B-16-SigLIP-512", pretrained="webli", cache_dir="./static/models", device=device)
    
    def forward(self, image):
        with torch.no_grad():
            inputs = self.preprocess(image).unsqueeze(0).to(self.device)
            outputs = self.model.encode_image(inputs, normalize=True).squeeze(0).cpu().numpy()

            return outputs


net_image2vec = Image2Vec_v2(device="cuda:3")


def hist_features(img):
    img = img.convert('RGB')
    img = np.array(img)
    # 将图像从 BGR 转换为 HSV 颜色空间
    hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # 计算颜色直方图
    h_hist = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
    s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
    v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])

    # 将所有通道的直方图特征拼接成一个向量
    hist_feature = np.concatenate((h_hist, s_hist, v_hist)).flatten()

    # L2 归一化
    norm = np.linalg.norm(hist_feature)
    if norm > 0:
        hist_feature = hist_feature / norm

    return hist_feature.astype(np.float32)


def sketch_features(image, low_threshold=100, high_threshold=200):
    # image = cv2.imread(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
    sketch = cv2.bitwise_not(edges)
    sketch_pil = Image.fromarray(sketch)

    image_vec = net_image2vec(sketch_pil)
    image_vec = np.array(image_vec, dtype=np.float32)
    return image_vec


def depth_features(image):
    api_url = "http://1.180.12.34:8002/leres_plus/"
    boost = 0

    buffered = BytesIO()
    image = image.convert('RGB')
    image.save(buffered, format="JPEG")
    buffered.seek(0)
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    payload = {
        "image_base64": image_base64,
        "boost": boost,
        "input_size":512
    }
    response = requests.post(api_url, json=payload)
    if response.status_code == 200:
        data = response.json()
        decoded_bytes = base64.b64decode(data['img_base64'])
        img_data = Image.open(BytesIO(decoded_bytes))

        image_vec = net_image2vec(img_data)
        image_vec = np.array(image_vec, dtype=np.float32)
        return image_vec
    else:
        print(f"请求失败，状态码: {response.status_code}")
        return None


def segmentation_features(image):
    api_url = "http://1.180.12.34:8002/oneformer/"

    buffered = BytesIO()
    image = image.convert('RGB')
    image.save(buffered, format="JPEG")
    buffered.seek(0)
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    payload = {
        "image_base64": image_base64,
    }
    response = requests.post(api_url, json=payload)
    if response.status_code == 200:
        data = response.json()
        decoded_bytes = base64.b64decode(data['img_base64'])
        img_data = Image.open(BytesIO(decoded_bytes))

        image_vec = net_image2vec(img_data)
        image_vec = np.array(image_vec, dtype=np.float32)
        return image_vec
    else:
        print(f"请求失败，状态码: {response.status_code}")
        return None


def original_image_features(image):
    image_vec = net_image2vec(image)
    image_vec = np.array(image_vec, dtype=np.float32)
    return image_vec


def cosine_similarity(hist_feature1, hist_feature2):
    hist_feature1 = np.array(hist_feature1)
    hist_feature2 = np.array(hist_feature2)
    
    dot_product = np.dot(hist_feature1, hist_feature2)
    
    norm1 = np.linalg.norm(hist_feature1)
    norm2 = np.linalg.norm(hist_feature2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


