import pandas as pd
from sklearn import metrics

import optuna
from optuna.visualization import plot_slice, plot_contour

from options import *
from func_img import get_img, get_img_3d, get_img_cloud

import mlflow
from time import time

# для расчета когерентности
from gensim import corpora
from gensim.models import CoherenceModel


def objective(trial):

    coding_dict = coding_optincs[coding_select]
    params_coding = {}
    for float_name, float_values in coding_dict['float'].items():
        params_coding[float_name] = trial.suggest_float(float_name, float_values[0], float_values[1], step=float_values[2])
    for int_name, int_values in coding_dict['int'].items():
        params_coding[int_name] = trial.suggest_int(int_name, int_values[0], int_values[1])
    for cat_name, cat_values in coding_dict['categorical'].items():
        params_coding[cat_name] = trial.suggest_categorical(cat_name, cat_values)

    coding = coding_dict['func'](**params_coding)

    model_dict = model_options[model_select]
    params_model = {}
    for float_name, float_values in model_dict['float'].items():
        params_model[float_name] = trial.suggest_float(float_name, float_values[0], float_values[1], step=float_values[2])
    for int_name, int_values in model_dict['int'].items():
        params_model[int_name] = trial.suggest_int(int_name, int_values[0], int_values[1])
    for cat_name, cat_values in model_dict['categorical'].items():
        params_model[cat_name] = trial.suggest_categorical(cat_name, cat_values)

    model = model_dict['func'](**params_model)

    data_obj = coding.fit_transform(data).toarray()
    model.fit(data_obj)

    if len(set(model.labels_)) < 2:
        silhouette = -1
    else:
        silhouette = round(metrics.silhouette_score(data_obj, model.labels_, metric=SULOUETTE_METRIC), 5)
    return silhouette


def save_results(model, study_tmp, name_tmp):

    print(model.__name__, " -- лучший скор -- ", round(study_tmp.best_value, 5))
    print(study_tmp.best_params)

    fig = plot_slice(study_tmp).to_html(f'{name_tmp}.html')
    with open(f'./tmp/plot_slice.html', 'w', encoding='utf8') as f:
        f.write(fig)

    fig_2 = plot_contour(study_tmp).to_html(f'{name_tmp}.html')
    with open(f'./tmp/plot_contour.html', 'w', encoding='utf8') as f:
        f.write(fig_2)
    #webbrowser.open(f'{name_tmp}.html', new=2)


def get_metrics(best_params, coding_select_x, model_select_x):
    coding_dict = coding_optincs[coding_select_x]
    model_dict = model_options[model_select_x]

    # выделяем параметры кодировщика
    coding_params_set = set()
    for type_x in ('int', 'float', 'categorical'):
        for name_x in coding_dict[type_x].keys():
            coding_params_set.add(name_x)

    # разделяем параметры кодировщика и модели
    params_coding = {}
    params_model = {}
    for param_name, param_val in best_params.items():
        if param_name in coding_params_set:
            params_coding[param_name] = param_val
        else:
            params_model[param_name] = param_val

    coding = coding_dict['func'](**params_coding)
    model = model_dict['func'](**params_model)

    t0 = time()
    data_obj = coding.fit_transform(data).toarray()
    model.fit(data_obj)
    fit_time = round(time() - t0, 2)

    results = {'fit_time': fit_time, 'n_clusters': len(set(model.labels_))}
    if len(set(model.labels_)) < 2:
        for metric_x in ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']:
            results[f'silhouette_{metric_x}'] = -1
    else:
        for metric_x in ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']:
            silhouette_tmp = round(metrics.silhouette_score(data_obj, model.labels_, metric=metric_x), 3)
            results[f'silhouette_{metric_x}'] = silhouette_tmp

    # сохранение картинок и html из study
    name_html = f"{coding_select}_{model_select}"
    save_results(model_dict['func'], study, name_html)



    # texts = [[word for word in doc.split()] for doc in data]
    # dictionary = corpora.Dictionary(texts)
    # feature_names = [dictionary[i] for i in range(len(dictionary))]
    #
    # topics = []
    # n_top_words = 20
    # for label_x in set(model.labels_):
    #     index_top_x = pd.DataFrame(data_obj[model.labels_ == label_x].sum(axis=0)) \
    #                       .transpose().sort_values(by=0, ascending=False).index[:n_top_words]
    #     topics.append([feature_names[i] for i in index_top_x])
    #
    # result_tmp = {'labels': model.labels_}
    # for type_coh in ('c_v'):  # , 'c_uci', 'u_mass'
    #     coh = CoherenceModel(topics=topics,
    #                          texts=texts,
    #                          dictionary=dictionary,
    #                          coherence=type_coh)
    #     coh_score = coh.get_coherence()
    #     result_tmp[type_coh] = coh_score

    """MLFLOOOOOW"""
    with mlflow.start_run(run_name=f"{coding_select_x}_{model_select_x}") as run:
        mlflow.log_metrics(results)

        mlflow.log_params(coding.get_params())
        mlflow.sklearn.log_model(sk_model=coding, artifact_path="sklearn-coding")
        mlflow.log_params(model.get_params())
        mlflow.sklearn.log_model(sk_model=model, artifact_path="sklearn-model")

        # построение картинок
        get_img(model, data_obj)
        mlflow.log_artifact("./tmp/pca.png", artifact_path="images")
        mlflow.log_artifact("./tmp/tnse.png", artifact_path="images")
        # get_img_3d(model, data_obj)
        # mlflow.log_artifact("pca_3D.png", artifact_path="images")
        # mlflow.log_artifact("tnse_3D.png", artifact_path="images")


        mlflow.log_artifact("./tmp/plot_slice.html", artifact_path="result_optuna")
        mlflow.log_artifact("./tmp/plot_contour.html", artifact_path="result_optuna")

        for i in set(model.labels_):
            df_x = data[model.labels_ == i]
            df_x.to_csv(f"./tmp/{i}.csv", index=False)
            mlflow.log_artifact(f"./tmp/{i}.csv", artifact_path="df_clusters")
            get_img_cloud(data_x=df_x, name_file=f"cloud_{i}_cluster")
            mlflow.log_artifact(f"./tmp/cloud_{i}_cluster.png", artifact_path="images")

        mlflow.set_tags({'type_model': 'kmeans'})
    return 0

print("Подключение", end='... ')
mlflow.set_tracking_uri(uri="http://192.168.0.4:8080")
mlflow.set_registry_uri(uri='http://192.168.0.4:8080')
exp_id = mlflow.set_experiment(f'test_avalon_sentences_word')
print("ОК")
input_select = 'sentences'
input_size = 5000
SULOUETTE_METRIC = 'manhattan'   # ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']


"""_____________ *** ЧТЕНИЕ ДАННЫХ *** ________________"""
print("Чтение данных", end='... ')

data = pd.read_csv(input_options[input_select])
if input_size == 'all': input_size = data.shape[0]
data = data.iloc[:input_size]['sentence'].dropna()
print("ОК")

"""_____________ *** ПОДБОР ВСЕХ СОЧЕТАНИЙ *** ________________"""
for coding_select in coding_optincs.keys():
    for model_select in model_options.keys():
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100, n_jobs=-1)
        best_params = study.best_params
        get_metrics(best_params, coding_select, model_select)

# coding_select = 'count'
# model_select = 'KMeans'
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=10, n_jobs=-1)
# print(study.best_value)
# print(study.best_params)

print("End of the program")
