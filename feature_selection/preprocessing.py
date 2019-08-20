import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    path_Excel = "../data/spot-rCsvSpotProb.xlsx"
    path_CSV = "../data/spot-rCsvSpotProb.csv"

    df = pd.read_excel(path_Excel, usecols=['A', 'C', 'StF', 'SoF', 'N', 'MG', 'BC'])
    df.to_csv(path_CSV)
    df.hist()
    plt.show()
    descriptions = df.describe(include='all')
    for d in descriptions:
        print(d, '\n', descriptions[d])

    df.to_csv(path_CSV)
    df.hist()
    plt.show()
