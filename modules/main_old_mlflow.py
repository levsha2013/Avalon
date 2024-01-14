import mlflow
import numpy as np
import pandas as pd
from time import time
from datetime import datetime
from sklearn.datasets import load_iris

from sklearn import metrics
from sklearn.pipeline import make_pipeline

from sklearn.cluster import MeanShift, AgglomerativeClustering, SpectralClustering, KMeans, DBSCAN, HDBSCAN, OPTICS, \
                            AffinityPropagation, Birch
from sklearn.mixture import GaussianMixture

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from func_img import get_img, get_img_3d
from mpl_toolkits import mplot3d

from tqdm import tqdm
import optuna
from optuna.visualization import plot_slice, plot_contour
import webbrowser

# для расчета когерентности
from gensim import corpora
from gensim.models import CoherenceModel


MAX_CLUSTERS = 20
SULOUETTE_METRIC = 'manhattan'   #['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
RANDOM_STATE = 42


def save_results(model, study_tmp, name_tmp):

    print(model.__name__, " -- лучший скор -- ", round(study_tmp.best_value, 5))
    print(study_tmp.best_params)

    fig = plot_slice(study_tmp).to_html(f'{name_tmp}.html')
    with open(f'plot_slice.html', 'w', encoding='utf8') as f:
        f.write(fig)

    fig_2 = plot_contour(study_tmp).to_html(f'{name_tmp}.html')
    with open(f'plot_contour.html', 'w', encoding='utf8') as f:
        f.write(fig_2)
    #webbrowser.open(f'{name_tmp}.html', new=2)


def get_metrics(model, data, metric='only_silhiuette', study=None, name_html='k_means', exp_id=None):

    t0 = time()
    model.fit(data)
    fit_time = round(time() - t0, 2)

    if len(set(model.labels_)) < 2:
        silhouette = -1
    else:
        silhouette = round(metrics.silhouette_score(data, model.labels_, metric=SULOUETTE_METRIC, sample_size=300), 3)
    if metric == 'all':
        results = {'fit_time': fit_time}

        clustering_metrics = {
            'homogeneity': metrics.homogeneity_score,
            'completeness': metrics.completeness_score,
            'v_measure': metrics.v_measure_score,
            'adjusted_rand': metrics.adjusted_rand_score,
            'adjusted_mutual_info': metrics.adjusted_mutual_info_score,
        }
        #for name_metric_x, metric_x in clustering_metrics.items():
        #    results[name_metric_x] = round(metric_x(labels, model.labels_), 3)

        results['silhouette_max'] = silhouette
        results['n_clusters'] = len(set(model.labels_))  # залогируем количество кластеров, когда нет явного параметра

        for metric_x in ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']:
            silhouette_tmp = round(metrics.silhouette_score(data, model.labels_, metric=metric_x, sample_size=300), 3)
            results[f'silhouette_{metric_x}'] = silhouette_tmp

        # построение картинок
        get_img(model, data)
        # get_img_3d(model, data)
        save_results(KMeans, study, name_html)

        # LDA
        LDA_model = LatentDirichletAllocation(n_components=1,
                                              learning_method='online',
                                              random_state=42,
                                              n_jobs=-1)

        from gensim.test.utils import common_corpus, common_dictionary
        from gensim.models.coherencemodel import CoherenceModel
        topics = [
            ['human', 'computer', 'system', 'interface'],
            ['graph', 'minors', 'trees', 'eps']
        ]
        cm = CoherenceModel(topics=topics, corpus=common_corpus, dictionary=common_dictionary, coherence='u_mass')
        coherence = cm.get_coherence()  # get coherence value
        """MLFLOOOOOW"""
        with mlflow.start_run(run_name=f"{model}") as run:
            mlflow.log_metrics(results)
            mlflow.sklearn.log_model(sk_model=model, artifact_path="sklearn-model")
            mlflow.log_params(model.get_params())

            mlflow.log_artifact("pca.png", artifact_path="images")
            mlflow.log_artifact("tnse.png", artifact_path="images")
            # mlflow.log_artifact("pca_3D.png", artifact_path="images")
            # mlflow.log_artifact("tnse_3D.png", artifact_path="images")
            mlflow.log_artifact("plot_slice.html", artifact_path="result_optuna")
            mlflow.log_artifact("plot_contour.html", artifact_path="result_optuna")

            mlflow.set_tags({'type_model': 'kmeans'})
    return silhouette


def objective_kmeans(trial):
    n_clusters = trial.suggest_int('n_clusters', 2, MAX_CLUSTERS)
    algorithm = trial.suggest_categorical('algorithm', ['lloyd', 'elkan'])
    init = trial.suggest_categorical('init', ['k-means++', 'random'])

    params = {
        'n_clusters': n_clusters,
        'algorithm': algorithm,
        'init': init,
        'random_state': 42,
        'n_init': 'auto'
    }

    model = KMeans(**params)
    silouette = get_metrics(model, data)
    return silouette


def objective_spectral(trial):
    n_clusters = trial.suggest_int('n_clusters', 2, MAX_CLUSTERS)
    eigen_solver = trial.suggest_categorical('eigen_solver', ['arpack', 'lobpcg'])
    gamma = trial.suggest_float('gamma', 0.001, 100)
    affinity = trial.suggest_categorical('affinity', ['nearest_neighbors', 'rbf'])# , 'precomputed' , 'precomputed_nearest_neighbors'
    n_neighbors = trial.suggest_int('n_neighbors', 2, 20)
    assign_labels = trial.suggest_categorical('assign_labels', ['kmeans', 'discretize', 'cluster_qr'])
    degree = trial.suggest_int('degree', 1, 10)
    coef0 = trial.suggest_float('coef0', 0.1, 2)

    params = {
        'n_clusters': n_clusters,
        'eigen_solver': eigen_solver,
        'gamma': gamma,
        'affinity': affinity,
        'n_neighbors': n_neighbors,
        'assign_labels': assign_labels,
        'degree': degree,
        'coef0': coef0,
        'random_state': 42,
    }

    model = SpectralClustering(**params)
    silouette = get_metrics(model, data)
    return silouette


def objective_MeanShift(trial):
    bandwidth = trial.suggest_float('bandwidth', 0.001, 100)
    cluster_all = trial.suggest_categorical('cluster_all', [True, False])
    bin_seeding = trial.suggest_categorical('bin_seeding', [True, False])

    params = {
        'bandwidth': bandwidth,
        'cluster_all': cluster_all,
        'bin_seeding': bin_seeding,
    }

    model = MeanShift(**params)
    silouette = get_metrics(model, data)
    return silouette


def objective_AgglomerativeClustering(trial):
    n_clusters = trial.suggest_int('n_clusters', 2, MAX_CLUSTERS)
    linkage = trial.suggest_categorical('linkage', ['ward', 'complete', 'average', 'single'])
    if linkage == 'ward':
        affinity = 'euclidean' #trial.suggest_categorical('affinity', ['euclidean'])
    else:
        affinity = trial.suggest_categorical('affinity', ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'])

    params = {
        'n_clusters': n_clusters,
        'linkage': linkage,
        'affinity': affinity,
    }

    model = AgglomerativeClustering(**params)
    silouette = get_metrics(model, data)
    return silouette


def objective_DBSCAN(trial):
    eps = trial.suggest_float('eps', 0.001, 10)   # 0.5
    metric = trial.suggest_categorical('metric', ['cityblock',  'euclidean', 'l1', 'l2', 'manhattan']) # 'cosine',
    algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
    p = trial.suggest_int('p', 1, 10)


    params = {
        'eps': eps,
        'metric': metric,
        'algorithm': algorithm,
        'p': p,
    }

    model = DBSCAN(**params)
    silouette = get_metrics(model, data)
    return silouette


def objective_HDBSCAN(trial):
    min_cluster_size = trial.suggest_int('min_cluster_size', 5, 100)
    min_samples = trial.suggest_int('min_samples', 5, 100)
    cluster_selection_epsilon = trial.suggest_float('cluster_selection_epsilon', 0, 10)   # 0.5

    # metric = trial.suggest_categorical('metric',
    #                                    ['precomputed', 'pyfunc', 'hamming',  'sokalsneath',
    #                                     'chebyshev', 'canberra', 'infinity', 'p', 'cityblock', 'seuclidean',
    #                                     'minkowski', 'l1', 'manhattan', 'l2', 'jaccard', 'braycurtis', 'mahalanobis',
    #                                     'rogerstanimoto', 'russellrao', 'euclidean', 'sokalmichener'])

    metric = trial.suggest_categorical('metric', ['cityblock',  'euclidean', 'l1', 'l2',  'manhattan'])
    alpha = trial.suggest_float('alpha', 0, 10)   # 0.5
    algorithm = trial.suggest_categorical('algorithm', ['auto', 'balltree', 'kdtree', 'brute'])
    # leaf_size = trial.suggest_int('min_samples', 20, 100) -- для ускорения алгоритма
    cluster_selection_method = trial.suggest_categorical('cluster_selection_method', ['eom', 'leaf'])
    store_centers = trial.suggest_categorical('store_centers', [None, 'centroid', 'medoid', 'both'])

    params = {
        'min_cluster_size': min_cluster_size,
        'min_samples': min_samples,
        'cluster_selection_epsilon': cluster_selection_epsilon,
        'metric': metric,
        'alpha': alpha,
        'algorithm': algorithm,
        'cluster_selection_method': cluster_selection_method,
        'store_centers': store_centers,
    }

    model = HDBSCAN(**params)
    silouette = get_metrics(model, data)
    return silouette


def objective_OPTICS(trial):
    min_samples = trial.suggest_int('min_samples', 5, 100)
    p = trial.suggest_int('p', 1, 10)

    cluster_method = trial.suggest_categorical('cluster_method', ['xi', 'dbscan'])
    metric = trial.suggest_categorical('metric', ['cityblock',  'euclidean', 'l1', 'l2', 'manhattan']) # 'cosine',
    algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])

    eps = trial.suggest_float('eps', 0.001, 10)   # 0.5
    # leaf_size = trial.suggest_int('min_samples', 20, 100) -- для ускорения алгоритма

    params = {
        'min_samples': min_samples,
        'cluster_method': cluster_method,
        'metric': metric,
        'p': p,
        'eps': eps,
        'algorithm': algorithm,
    }

    model = OPTICS(**params)
    silouette = get_metrics(model, data)
    return silouette


def objective_AffinityPropagation(trial):
    damping = trial.suggest_float('damping', 0.5, 0.999)   # 0.5
    #affinity = trial.suggest_categorical('affinity', ['euclidean', 'precomputed']) # precomputed нужна квадратная матрица

    params = {
        'damping': damping,
        'random_state': RANDOM_STATE,
    }

    model = AffinityPropagation(**params)
    silouette = get_metrics(model, data)
    return silouette


def objective_Birch(trial):
    threshold = trial.suggest_float('threshold', 0.01, 5)   # 0.5
    branching_factor = trial.suggest_int('branching_factor', 5, 100)
    n_clusters =trial.suggest_int('n_clusters', 2, MAX_CLUSTERS)

    params = {
        'threshold': threshold,
        'branching_factor': branching_factor,
        'n_clusters': n_clusters,
    }

    model = Birch(**params)
    silouette = get_metrics(model, data)
    return silouette


def objective_GaussianMixture(trial):
    n_components =trial.suggest_int('n_components', 2, MAX_CLUSTERS)
    covariance_type = trial.suggest_categorical('covariance_type', ['full', 'tied', 'diag', 'spherical'])
    init_params = trial.suggest_categorical('init_params', ['kmeans', 'k-means++', 'random'])

    params = {
        'n_components': n_components,
        'covariance_type': covariance_type,
        'init_params': init_params,
        'random_state': RANDOM_STATE
    }

    model = GaussianMixture(**params)
    silouette = get_metrics(model, data)
    return silouette


def get_coherence_mean(model, coding_text, texts, n_top_words=20):
    # колчиество тем
    #topics = model.components_

    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import CountVectorizer

    df = pd.read_csv('./data/Articles_medical.csv')
    df_text = df['text']

    vector = CountVectorizer()
    text_coding = vector.fit_transform(df_text)

    model = KMeans(n_clusters=2, random_state=42)
    pred = model.fit_transform(text_coding)

    from gensim import corpora
    from gensim.models import CoherenceModel


    texts = [[word for word in doc.lower().split()] for doc in df_text]
    dictionary = corpora.Dictionary(texts)
    feature_names = [dictionary[i] for i in range(len(dictionary))]

    topics = []
    for label_x in set(model.labels_):
        index_top_x = pd.DataFrame(coding_text[model.labels_ == label_x].sum(axis=0))\
                          .transpose().sort_values(by=0, ascending=False).index[:n_top_words]
        topics.append([feature_names[i] for i in index_top_x])


    coh = CoherenceModel(topics=topics,
                         dictionary=dictionary,
                         coherence='c_v')


    # самые 20 популярных слов для модели kmeans класса 1
    top_index = pd.DataFrame(coding_text[model.labels_ == 1].sum(axis=0))\
                    .transpose().sort_values(by=0, ascending=False).index[:n_top_words]

    topics = set(model.labels_)

    # получение токенов
    texts = [[word for word in doc.split()] for doc in texts]
    dictionary = corpora.Dictionary(texts)

    feature_names = [dictionary[i] for i in range(len(dictionary))]

    top_words = []
    for topic in topics:
        top_words.append(
            [feature_names[i] for i in topic.argsort()[:-n_top_words - 1: -1]]
        )

    coh = CoherenceModel(topics=top_words,
                         texts=texts,
                         dictionary=dictionary,
                         coherence='c_v')
    res = coh.get_coherence()
    return res

# data, labels = load_iris(return_X_y=True)
# (n_samples, n_features), n_digits = data.shape, np.unique(labels).size
text = pd.read_csv("./data/Articles_medical.csv")['text']
vector = CountVectorizer()
data = vector.fit_transform(text)

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_registry_uri(uri='http://127.0.0.1:8080')
exp_id = mlflow.set_experiment(f'test_avalon_text')

models = {
    'KMeans': {
        'func2opt': objective_kmeans,
        'model_name': KMeans,
        'html_name': 'KMeans'},
    #
    # 'Spectral': {
    #     'func2opt': objective_spectral,
    #     'model_name': SpectralClustering,
    #     'html_name': 'SpectralClustering'},
    #
    # 'MeanShift': {
    #     'func2opt': objective_MeanShift,
    #     'model_name': MeanShift,
    #     'html_name': 'MeanShift'},

    # 'AgglomerativeClustering': {
    #     'func2opt': objective_AgglomerativeClustering,
    #     'model_name': AgglomerativeClustering,
    #     'html_name': 'AgglomerativeClustering'},
    #
    # 'DBSCAN': {
    #     'func2opt': objective_DBSCAN,
    #     'model_name': DBSCAN,
    #     'html_name': 'DBSCAN'},
    #
    # 'HDBSCAN': {
    #     'func2opt': objective_HDBSCAN,
    #     'model_name': HDBSCAN,
    #     'html_name': 'HDBSCAN'},
    #
    # 'OPTICS': {
    #     'func2opt': objective_OPTICS,
    #     'model_name': OPTICS,
    #     'html_name': 'OPTICS'},
    #
    # 'AffinityPropagation': {
    #     'func2opt': objective_AffinityPropagation,
    #     'model_name': AffinityPropagation,
    #     'html_name': 'AffinityPropagation'},
    #
    # 'Birch': {
    #     'func2opt': objective_Birch,
    #     'model_name': Birch,
    #     'html_name': 'Birch'},
}

for model_x in models.values():
    func2optimize = model_x['func2opt']
    model_name = model_x['model_name']
    html_name = model_x['html_name']

    study = optuna.create_study(direction='maximize')
    study.optimize(func2optimize, n_trials=100, n_jobs=-1)
    best_mode = model_name(**study.best_params)
    get_metrics(best_mode, data, metric='all', study=study, name_html=html_name)
