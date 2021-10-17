import pandas as pd
def _read_file(filename):
    from os.path import dirname, join

    return pd.read_csv(join(dirname(__file__), filename),
                       index_col=0, parse_dates=True, infer_datetime_format=True)

def create_usdjpy():
    data = _read_file('USD_JPY_H1.csv')
    data.columns = ["Volume", "High", "Low", "Open", "Close"]
    return data

USDJPY = create_usdjpy()

if __name__ == "__main__":
    data = create_usdjpy()
    print(data)