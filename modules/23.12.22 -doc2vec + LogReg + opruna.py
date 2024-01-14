import pandas as pd
import numpy as np
import optuna
import webbrowser

from coding_sklearn import Doc2VecCoding
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import f1_score, roc_auc_score, classification_report


def get_same_count_rating(df_x, size=1000, col_x='sentence', col_y='rating'):
    """Выделяет равномерную по рейтингу выборку. Если size=-1 - максимальную по объему."""
    X = np.array([])
    y = np.array([])
    if size == -1:
        size = df_x[col_y].value_counts().min() # определяем максимальный объем
    for rating_x in df_x[col_y].unique():
        df_tmp = df_x[df_x[col_y] == rating_x].iloc[:size]  # строки с нужным рейтингом
        X = np.concatenate((X, df_tmp[col_x]))              # добавляем строки в X
        y = np.concatenate((y, df_tmp[col_y]))              # добавляем строки в y
    return X, y


def transform_to_binary(df_x):
    df_x['rating_bin'] = df_x['rating'].apply(lambda x: True if x > 3 else False)


def print_and_save_optuna_study(model, study_tmp, name_tmp, params_tmp, save_html=True):

    print(model.__name__, " -- лучший скор -- ", round(study_tmp.best_value, 5))
    print(study_tmp.best_params)

    if save_html:
        fig = optuna.visualization.plot_slice(study_tmp, params=params_tmp).to_html(f'{name_tmp}.html')
        with open(f'{name_tmp}.html', 'w', encoding='utf8') as f:
            f.write(fig)
        webbrowser.open(f'{name_tmp}.html', new=2)


def objective(trial):
    vector_size = trial.suggest_int('vector_size', 200, 900)
    min_count = trial.suggest_int('min_count', 2, 20)
    epochs = trial.suggest_int('epochs', 2, 30)
    transf = Doc2VecCoding(vector_size=vector_size, min_count=min_count, epochs=epochs)
    X_new = transf.fit_transform(X)

    C = trial.suggest_float('C', 1, 300, log=True)
    model = LogisticRegression(C=C, random_state=42, max_iter=1000)
    score = cross_val_score(model, X_new, y).mean()
    return score


df = pd.read_csv("./data/sentences.csv")
transform_to_binary(df)
X, y = get_same_count_rating(df, col_y='rating_bin', size=10_000)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=75, n_jobs=-1)
params = study.best_params
print_and_save_optuna_study(LogisticRegression, study, 'LogRegression', params)
print(params)
# transf = Doc2VecCoding()
# X = transf.fit_transform(X)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
#
# model = LogisticRegression()
# model.fit(X_train, y_train)
#
# pred = model.predict(X_test)
print("Все")