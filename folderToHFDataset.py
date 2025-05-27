#==============================================
#           IMPORT LIBRARIES
#==============================================

from datasets import Dataset, DatasetDict, Image, features
import numpy as np
import os

#==============================================
#           UTILITIES FUNCTIONS
#==============================================


class FolderToHFDataset:
    def __init__(self, trainDataset, valDataset= None, testDataset = None):
        self.trainDataset = trainDataset
        self.valDataset = valDataset
        self.testDataset = testDataset

    def createDataset(self, imagesPath, labelsPath, categories):
    
        dataColumns = {
            "image": imagesPath,
            "label": labelsPath
        }
        ds =  Dataset.from_dict(dataColumns)
    
        ds = ds.cast_column("image",  Image())
        ds = ds.cast_column("label", features.ClassLabel(names=categories))
        
        return ds

    def extractData(self, link):
    
        dataUrl = os.path.realpath(link)
        
        categories = os.listdir(dataUrl)
        imagePath = [] 
        labelPath = []
        
        for i in categories:
            folderPath = os.path.join(dataUrl, i)
            imageDir = os.listdir(folderPath)
            for j in imageDir:
                imagePath.append(os.path.join(dataUrl, i,j))
                labelPath.append(i)
    
        return imagePath, labelPath, categories

    def getDataset(self):
        # Train
        image_paths_train, label_paths_train, categories = self.extractData(self.trainDataset)
        train_dataset = self.createDataset(image_paths_train, label_paths_train, categories)

        # Validation
        if self.valDataset != None:
            image_paths_validation, label_paths_validation, _ = self.extractData(self.valDataset)
            validation_dataset = self.createDataset(image_paths_validation, label_paths_validation, categories)

        # Test
        if self.testDataset != None:
            image_paths_test, label_paths_test, _ = self.extractData(self.testDataset)
            test_dataset = self.createDataset(image_paths_test, label_paths_test, categories)

        dataset = DatasetDict({
            "train": train_dataset,
            "validation": validation_dataset if self.valDataset != None else None,
            "test": test_dataset if self.testDataset != None else None,
            })
        
        return dataset
