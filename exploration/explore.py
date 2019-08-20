import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    path_Excel = "../data/spot-rCsvSpotProb.xlsx"
    path_CSV = "../data/spot-rCsvSpotProb.csv"
    path_CSV_sorted_ID = "../data/sorted_ID.csv"

    df = pd.read_excel(path_Excel)
    df.to_csv(path_CSV)
    df.hist()
    plt.show()
    descriptions = df.describe(include='all')
    for d in descriptions:
        print(d, '\n', descriptions[d])

    sorted_ID=df.sort_values('ID')
    sorted_ID.to_csv(path_CSV_sorted_ID)


