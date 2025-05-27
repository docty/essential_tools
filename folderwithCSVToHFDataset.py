#==============================================
#           IMPORT LIBRARIES
#==============================================

from datasets import Dataset, DatasetDict, Image, Value
import pandas as pd
import os

#==============================================
#           UTILITIES CLASS
#==============================================

class FolderWithCSVToHFDataset:
    def __init__(self, train_folder, train_csv, val_folder=None, val_csv=None, test_folder=None, test_csv=None):
        self.train_folder = train_folder
        self.train_csv = train_csv
        self.val_folder = val_folder
        self.val_csv = val_csv
        self.test_folder = test_folder
        self.test_csv = test_csv

    def create_dataset(self, image_folder, metadata_csv):
        df = pd.read_csv(metadata_csv)

        df["image"] = df["file_name"].apply(lambda x: os.path.join(image_folder, x))
        df = df[["image", "text"]]

        ds = Dataset.from_pandas(df)
        ds = ds.cast_column("image", Image())
        ds = ds.cast_column("text", Value('string') )
        return ds


    def get_dataset(self):
        dataset_dict = {}

        dataset_dict["train"] = self.create_dataset(self.train_folder, self.train_csv)

        if self.val_folder and self.val_csv:
            dataset_dict["validation"] = self.create_dataset(self.val_folder, self.val_csv)


        if self.test_folder and self.test_csv:
            dataset_dict["test"] = self.create_dataset(self.test_folder, self.test_csv)

        return DatasetDict(dataset_dict)
