# Essential Tools

This repo contains easy to use functions to handle common tasks.

### Convert Folder to HuggingFace Dataset Format
```python
#==============================================
#       FOLDER TO HUGGINGFACE DATASET
#==============================================
from essential_tools.folderToHFDataset import FolderToHFDataset

config = folderToHFDataset(
    trainDataset='/kaggle/input/brain-tumor-mri-data/brain-tumor-mri-dataset', 
    valDataset=None, 
    testDataset=None 
    )

dataset = config.getDataset()
```

### Upload to HuggingFace Hub
```python
#================================================
#        DEPLOY DATASET TO HUGGINGFACE HUB
#================================================

from huggingface_hub import notebook_login

notebook_login()

dataset.push_to_hub("Docty/repo_name")
```

### Convert Folder with CSV to HuggingFace Dataset Format
```python
#==============================================
#       FOLDER TO HUGGINGFACE DATASET
#==============================================
from essential_tools.folderwithCSVToHFDataset import FolderWithCSVToHFDataset

config = FolderWithCSVToHFDataset(
    train_folder="/content/images",
    train_csv="/content/metadata.csv",

)

dataset = config.getDataset()
```



### Package a python application
```bash
packagePython.ipynb
```

### Google Cloud Platform Template
```python
!python /content/essential-tools/gcptemplate.py \
--project_id  'gcp-deployment-example-454318' \
--bucket_uri 'gs://gcp-deployment-example-454318-bucket' \
--script_path '/content/task.py' \
--job_name 'sentiment-job' \
--framework 'sklearn'
```
