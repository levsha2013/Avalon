import pandas as pd
import mlflow
import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from func_img import get_img_cloud
from pipeline_tamplate import fit_model, show_topics, get_divergention
import os

from collections import defaultdict


def create_img_hist(df, name):
    plt.figure(1)
    plt.clf()
    plt.hist(df)
    plt.grid()
    plt.ylabel('count')
    plt.savefig(f'./tmp/{name}.png', dpi=DPI)


def get_img_LDA(pred, data_x):
    # pca_model = PCA(n_components=2)
    # pca_model.fit(data_x.toarray())
    # reduced_data_pca = pca_model.transform(data_x.toarray())
    #
    #
    # plt.figure(1)
    # plt.clf()
    # for i in set(pred):
    #     to_print = reduced_data_pca[pred == i]
    #     plt.scatter(to_print[:, 0], to_print[:, 1])
    # plt.savefig('./tmp/pca.png', dpi=DPI)

    reduced_data_tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(data_x)
    plt.figure(1)
    plt.clf()
    for i in set(pred):
        to_print = reduced_data_tsne[pred == i]
        plt.scatter(to_print[:, 0], to_print[:, 1])
    plt.savefig('./tmp/tnse.png', dpi=DPI)


if __name__ == '__main__':
    if not os.path.exists('./data'): os.chdir('..')
    mlflow.set_tracking_uri(uri="http://192.168.0.4:8080")
    mlflow.set_registry_uri(uri='http://192.168.0.4:8080')

    RANDOM_STATE = 42
    MAX_N_COMP = 50
    MIN_N_COMP = 2
    # NGRAM_RANGE = (2, 3)
    file_name = 'phrases'
    DPI = 75

    path = './data/'
    csvs = {
        'coffee': 'coffee.csv',
        'coffee_prep': 'coffee_prep.csv',
        'sentences': 'sentences.csv',
        'sentences_prep': 'sentences_prep.csv',
        'phrases': 'phrases.csv',
        'phrases_prep': 'phrases_prep.csv',
    }
    file = csvs['coffee_prep']

    df = pd.read_csv(path+file)['text']
    df = df.apply(lambda x: x.replace(" n", ' '))
    for NGRAM_RANGE in [(1, 3), (2, 2), (2, 3), (3, 3)]:  # (1, 1), (1, 2), (1, 3),
        exp_id = mlflow.set_experiment(f'LDA_{file_name}_{NGRAM_RANGE}')

        my_stop_words = ['не', 'на', 'что', 'за', 'но', 'для', 'это', 'то', 'по', 'нет', 'как', 'от', 'был', 'кафе',
                         'всё', 'так', 'все', 'тоже', 'из', 'раз', 'очень', 'самый', 'же', 'мы', 'нам', 'были', 'просто',
                         'заказ', 'если', 'бы', 'только', 'даже', 'там', 'уже', 'была', 'при', 'или', 'нас', 'этом', 'до']
        vectorizer = CountVectorizer(analyzer='word',
                                     min_df=10,
                                     max_df=0.95,
                                     ngram_range=NGRAM_RANGE,
                                     dtype='int16',
                                     stop_words=my_stop_words)
        data_x = vectorizer.fit_transform(df)
        # логирование стартовых параметров
        with mlflow.start_run(run_name=f"start_options") as run:
            mlflow.log_params(vectorizer.get_params())
            mlflow.sklearn.log_model(sk_model=vectorizer, artifact_path="sklearn-coding")
            with open("./tmp/my_stop_words.txt", 'w+', encoding='utf8') as f:
                f.write(", ".join(my_stop_words))
            mlflow.log_artifact("./tmp/my_stop_words.txt")

        result_metrics = defaultdict(list)
        for n_components in range(MIN_N_COMP, MAX_N_COMP):
            LDA_model = LatentDirichletAllocation(n_components=n_components, random_state=RANDOM_STATE, n_jobs=-1)
            LDA_model, vectorizer, metrics = fit_model(vectorizer_x=vectorizer, model_type=LDA_model, verbose=True,
                                                       res_vectorizer=data_x)

            # result = pd.DataFrame(show_topics(vectorizer, LDA_model, 10))

            pred = pd.DataFrame(LDA_model.transform(data_x))
            pred = pred.apply(lambda x: x.sort_values().index[-1], axis=1)

            result_metrics['n_comp'].append(n_components)

            print("Расчет когерентности - ", end='')
            start_coher = time.time()
            metrics['coherence'] = get_divergention(LDA_model, vectorizer, df)
            print(f"{round(time.time() - start_coher)} сек.")

            for i, j in metrics.items():
                print(f"\t{i}\t{round(j, 3)}")
                result_metrics[i].append(j)

            with mlflow.start_run(run_name=f"LDA_model_{n_components}") as run:
                mlflow.log_metrics(metrics)

                mlflow.log_params(LDA_model.get_params())
                mlflow.sklearn.log_model(sk_model=LDA_model, artifact_path="sklearn-model")

                with open("./tmp/topics.txt", 'w+', encoding='utf8') as f:
                    feature_names = vectorizer.get_feature_names_out()
                    for topic_idx, topic in enumerate(LDA_model.components_):
                        f.write(f"Topic #{topic_idx}:")
                        f.write(", ".join([feature_names[i] for i in topic.argsort()[:-25:-1]]))
                        f.write(f"\n")
                mlflow.log_artifact(f"./tmp/topics.txt", artifact_path="topics")

                # построение картинок
                create_img_hist(pred,  name='hist_topics')
                mlflow.log_artifact(f"./tmp/hist_topics.png", artifact_path="images")
                # data_x_img = data_x.sample(2000, random_state=RANDOM_STATE)
                # get_img_LDA(pred=pred, data_x=data_x_img)
                # mlflow.log_artifact("./tmp/pca.png", artifact_path="images")
                # mlflow.log_artifact("./tmp/tnse.png", artifact_path="images")
                # get_img_3d(model, data_obj)
                # mlflow.log_artifact("pca_3D.png", artifact_path="images")
                # mlflow.log_artifact("tnse_3D.png", artifact_path="images")

                # mlflow.log_artifact("./tmp/plot_slice.html", artifact_path="result_optuna")
                # mlflow.log_artifact("./tmp/plot_contour.html", artifact_path="result_optuna")

                start_tmp = time.time()
                print("Создание и логирование облака слов - ", end="")
                for i in set(range(n_components)):
                    df_x = df[pred == i]
                    if df_x.shape[0] >= 2000:
                        df_x = df_x.sample(2000, random_state=RANDOM_STATE)
                    else: continue
                    # df_x.to_csv(f"./tmp/{i}.csv", index=False)
                    # mlflow.log_artifact(f"./tmp/{i}.csv", artifact_path="df_clusters")
                    get_img_cloud(data_x=df_x, name_file=f"cloud_{i}_cluster", ngram_range=NGRAM_RANGE)
                    mlflow.log_artifact(f"./tmp/cloud_{i}_cluster.png", artifact_path="images")
                print(f"{round(time.time() - start_tmp)} сек")

        # финальное логирование
        with mlflow.start_run(run_name=f"result_images") as run:
            df_tmp = pd.DataFrame(result_metrics)
            for col_x in df_tmp.columns:
                if col_x == 'n_comp': continue
                plt.figure(1)
                plt.clf()
                plt.plot(df_tmp['n_comp'], df_tmp[col_x])
                plt.xlabel('n_clusters')
                plt.ylabel(col_x)
                plt.grid()
                plt.savefig(f'./tmp/metrics_{col_x}.png', dpi=DPI)
                mlflow.log_artifact(f"./tmp/metrics_{col_x}.png", artifact_path="images")
    print("Всё")