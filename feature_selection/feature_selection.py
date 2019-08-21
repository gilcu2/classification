import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE


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


def backward_elimination(df: pd.DataFrame, features_names, taget_name):
    features = df[features_names]
    target = df[target_name]

    pmax = 1
    cols = list(features.columns)
    while (len(cols) > 0):
        p = []
        X_1 = features[cols]
        X_1 = sm.add_constant(X_1)
        model = sm.OLS(target, X_1).fit()
        p = pd.Series(model.pvalues.values[1:], index=cols)
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if (pmax > 0.05):
            cols.remove(feature_with_p_max)
        else:
            break

    selected_features_BE = cols
    print('Selected_features', selected_features_BE)


def recursive_elimination(df: pd.DataFrame, features_names, taget_name):
    X = df[features_names]
    y = df[target_name]

    nof_list = np.arange(1, 6)
    high_score = 0
    # Variable to store the optimum features
    nof = 0
    score_list = []
    for n in range(len(nof_list)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        model = LinearRegression()
        rfe = RFE(model, nof_list[n])
        X_train_rfe = rfe.fit_transform(X_train, y_train)
        X_test_rfe = rfe.transform(X_test)
        model.fit(X_train_rfe, y_train)
        score = model.score(X_test_rfe, y_test)
        score_list.append(score)
        if (score > high_score):
            high_score = score
            nof = nof_list[n]
    print("Optimum number of features: %d" % nof)
    print("Score with %d features: %f" % (nof, high_score))

    cols = list(X.columns)
    model = LinearRegression()
    # Initializing RFE model
    rfe = RFE(model, nof)
    # Transforming data using RFE
    X_rfe = rfe.fit_transform(X, y)
    # Fitting the data to model
    model.fit(X_rfe, y)
    temp = pd.Series(rfe.support_, index=cols)
    selected_features_rfe = temp[temp == True].index
    print(selected_features_rfe)


if __name__ == '__main__':
    path_CSV = "../data/spot-rCsvSpotProb.csv"
    features_names = ['A', 'C', 'StF', 'SoF', 'N', 'MG']
    target_name = 'BC'

    df = pd.read_csv(path_CSV)

    # select_pearson_correlation(df)
    # backward_elimination(df, features_names, target_name)
    recursive_elimination(df, features_names, target_name)