## Convert Folder to HuggingFace Dataset Format
```python
#==============================================
#       FOLDER TO HUGGINGFACE DATASET
#==============================================

config = folderToHFDataset(
    trainDataset='/kaggle/input/brain-tumor-mri-data/brain-tumor-mri-dataset', 
    valDataset=None, 
    testDataset=None 
    )

dataset = config.getDataset()
```
