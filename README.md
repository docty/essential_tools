```code python```
#==============================================
#           INITIALIZATION
#==============================================

config = toHuggingFaceDataset(
    trainDataset='/kaggle/input/brain-tumor-mri-data/brain-tumor-mri-dataset', 
    valDataset=None, 
    testDataset=None 
    )

dataset = config.getDataset()
