import pandas as pd
from sklearn.model_selection import train_test_split


class DataPreparation:
    """Class for loading and preparing dataset."""

    def __init__(self, csv_file: str, data_dir: str):
        self.csv_file = csv_file
        self.data_dir = data_dir
        self.df = pd.read_csv(self.csv_file)

    def split_data(self):
        """Split the data into training and validation sets."""
        return train_test_split(self.df, test_size=0.2, random_state=42)
