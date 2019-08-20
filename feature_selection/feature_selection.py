import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm


def select_pearson_correlation(df: pd.DataFrame):
    # plt.figure(figsize=(12, 10))
    cor = df.corr()

    # Correlation with output variable
    cor_target = abs(cor["BC"])
    # Selecting highly correlated features
    relevant_features = cor_target[cor_target > 0.01]
    print('relevant_features by parson', relevant_features)

    relevant_features_names = ['C', 'SoF', 'N']

    for cor1 in relevant_features_names:
        for cor2 in relevant_features_names:
            if cor1 != cor2:
                print("Correlation between:", cor1, cor2, df[[cor1, cor2]].corr())

    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()


def select_wrapper(df: pd.DataFrame):
    features = df.drop(target_name)
    target = df[target_name]

    feature_1 = sm.add_constant(features)
    model = sm.OLS(target, feature_1).fit()
    print("Ordinary least squares:", model.pvalues)


if __name__ == '__main__':
    path_CSV = "../data/spot-rCsvSpotProb.csv"
    target_name = 'BC'

    df = pd.read_csv(path_CSV)

    # select_pearson_correlation(df)
    select_wrapper(df)
