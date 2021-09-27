
import sys
import os
sys.path.append('./../')
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import pandas as pd
from pathlib import Path

class Downloader():
    def __init__(self):
        pass

    '''def download(self, code, ):
        my_share = share.Share('MSFT')
        symbol_data = None
        
        try:
            symbol_data = my_share.get_historical(
                share.PERIOD_TYPE_DAY, 10,
                share.FREQUENCY_TYPE_DAY, 1)
        except YahooFinanceError as e:
            print(e.message)
            sys.exit(1)
        
        df = pd.DataFrame(symbol_data)
        df["datetime"] = pd.to_datetime(df.timestamp, unit="ms")
        df.head()
        print(df)'''

    def download_csv(self, name):
        print(os.getcwd(), __file__)
        file_path = Path(os.path.dirname(__file__)).joinpath('../dataset/{}'.format(name)).resolve()
        df = pd.read_csv(file_path)
        return df

if __name__ == "__main__":
    dl = Downloader()
    df = dl.download_csv("USD_JPY_D.csv")
    print(df.head())