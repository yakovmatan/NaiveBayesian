from logger.Logger import logger
class Cleaner:

    def __init__(self,df):
        self.df = df

    def clean_data(self):
        logger.info("Cleaning data")
        return self.df