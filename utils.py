import os
import requests
import zipfile
from io import BytesIO
import pandas as pd 
from IPython.display import Markdown, display



def kaggle_to_dataframe(base_path): 
    data = {'image_path': [], 'label': []} 
    for folder in os.listdir(base_path): 
        folder_path = os.path.join(base_path, folder) 
        if os.path.isdir(folder_path): 
            for img in os.listdir(folder_path): 
                if img.endswith(('.jpg', '.jpeg', '.png')): 
                    data['image_path'].append(os.path.join(folder_path, img)) 
                    data['label'].append(folder) 
    return pd.DataFrame(data)
