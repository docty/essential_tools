import os
import requests
import zipfile
from io import BytesIO

from IPython.display import Markdown, display

def download_data(data_url=None, data_format='pdf'):
    os.makedirs("data", exist_ok=True)
    response = requests.get(data_url)
    
    if response.status_code != 200:
        raise Exception("Failed to download file, try again")

    if data_format == 'pdf':
        pdf_path = "data/sample.pdf"
        with open(pdf_path, "wb") as f:
            f.write(response.content)
    elif data_format == 'zip':
         with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
            zip_ref.extractall('data')
        
    print("Downloaded and saved into the data directory")




 
def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        text_md = f"**Prompt Key**: {k}" f"**Text:** "
        display(Markdown(text_md))
        print(p.get_template())
        display(Markdown(""))
