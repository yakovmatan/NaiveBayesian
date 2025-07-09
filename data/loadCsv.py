import pandas as pd
import os
from loadData import DataLoader

class CSVLoader(DataLoader):
    def __init__(self,path):
        self.path = path

    def load(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"the file not found: {self.path}")
        return pd.read_csv(self.path)