import pandas as pd
import os
from logger.Logger import logger


class CSVLoader:
    def __init__(self,path):
        self.path = path

    def load(self):
        logger.info(f"Loading csv from path: {self.path}")
        if not os.path.exists(self.path):
            logger.error(f"file not found: {self.path}")
            raise FileNotFoundError(f"the file not found: {self.path}")
        logger.info("csv loaded successfully")
        return pd.read_csv(self.path)