import numpy as np

class Preprocessor():
    def __init__(self):
        pass

    def log_diff(self, df):
        return np.log(df).diff(1)
