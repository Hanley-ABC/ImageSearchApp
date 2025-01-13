# Visual Search of Fashion Products using Streamlit

This Github repository contains the code for implementing visual search of fashion products, especially clothes. 

**The primary task is to implement a fashion product search against the existing image library when a user uploads an image of a fashion product.** 
The interface additionally allows a user who does not have an image to make a custom fashion product by describing it. The code uses the **OpenAI model DALL-E** to generate a custom image using the description passed by the user. This image is then used to search for any similar product in the existing inventory. The interface also has a third option to search for images using **Pinecone vector database** to speed up the similarity matching.

**The model used for generation of vectors in this code is VGG16**, and the metric for similarity matching is Euclidean, and in Pinecone database, it is cosine similarity. The model for vector generation can be easily replaced with any other image recognition model, such as SqueezeNet, by altering the _feature_extractor.py_ file. The _offline.py_ file is used to generate vectors from existing image inventory and store it locally or on the Github repo. The _image-upsert_pinecode notebook_ is used to set up the Pinecone database and upload vectors to Pinecone. 

## Architecture 

![image](https://user-images.githubusercontent.com/81140802/229304113-20ebf6d1-67f8-4fa2-805a-b3942a092fd6.png)

**How to run ->**
The main file _imagesearchapp.py_ contains code for Streamlit development and can be run locally when the requirements are installed, followed by running the _offline.py_ file, followed by the notebook, and lastly by command 
```
streamlit run imagesearchapp.py.
```

## Files
1. **feature_extractor.py:** This file contains the implementation of VGG16 for extracting features from the image.
2. **offline.py:** This file is used to generate vectors from existing image inventory and store them locally or on the Github repo.
3. **image-upsert_pinecode.ipynb:** This notebook is used to set up the Pinecone database and upload vectors to Pinecone.
4. **imagesearchapp.py:** This file contains the Streamlit code for the user interface and for searching for similar images.
5. **requirements.txt:** This file contains the dependencies required to run the code.

## Running the Code

To run the code, you will need to clone this repository and install the required dependencies. You can use the following command to install the dependencies:

```
pip install -r requirements.txt
```

After installing the dependencies, you can generate vectors from the existing image inventory using the offline.py file. You can then upload these vectors to the Pinecone database using the image-upsert_pinecode.ipynb notebook. Finally, you can run the Streamlit app using the following command:

```
streamlit run imagesearchapp.py
```

## Streamlit App

[ImageSearch](https://pyashishmhatre-imagesearchapp-imagesearchapp-01u1wm.streamlit.app/)

## Documentation
Codelabs - [ImageSearch Documentation](https://codelabs-preview.appspot.com/?file_id=1CqBX4UQU1nVZaIw0REM76YEVNKpbQzJAk41JK48PuT8#0)
Docs - [ImageSearch Docx](https://docs.google.com/document/d/e/2PACX-1vTR4SRjlXpo6G3uqZe7IwzHlHZN_WNxy2p47njK0OvpwC6SqeJy-YpLmv4AeJSxY8WGZSmEiNMQuBoP/pub)

## Contact Information
If you have any questions or feedback regarding the project, please feel free to contact us at [Ashish Mhatre](https://www.linkedin.com/in/ashishmhatre927/) and [Ashwin Kadam](https://www.linkedin.com/in/ashwinkadam07/).
