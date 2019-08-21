import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import radviz

if __name__ == '__main__':
    path_Excel = "../data/spot-rCsvSpotProb.xlsx"
    path_CSV = "../data/spot-rCsvSpotProb.csv"
    target_map = {2: 0, 5: 1}

    df = pd.read_excel(path_Excel, usecols=['A', 'C', 'StF', 'SoF', 'N', 'MG', 'BC'])
    df['BC'] = df['BC'].map(target_map)
    df.to_csv(path_CSV)

    descriptions = df.describe(include='all')
    for d in descriptions:
        print(d, '\n', descriptions[d])

    df.to_csv(path_CSV)

    # df.hist()
    # df.plot(subplots=True)
    # radviz(df, 'BC', color=['red', 'green'])
    # plt.show()

    df_subset = df[['A', 'C', 'N', 'SoF', 'MG', 'BC']]
    radviz(df_subset, 'BC', color=['red', 'green'])
    plt.show()
