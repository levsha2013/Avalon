from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def get_img(kmeans, data_x):
    reduced_data_pca = PCA(n_components=2).fit_transform(data_x)
    pred = kmeans.fit_predict(data_x)

    plt.figure(1)
    plt.clf()
    for i in set(pred):
        to_print = reduced_data_pca[pred == i]
        plt.scatter(to_print[:, 0], to_print[:, 1])
    plt.savefig('./tmp/pca.png')

    reduced_data_tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(data_x)
    plt.figure(1)
    plt.clf()
    for i in set(pred):
        to_print = reduced_data_tsne[pred == i]
        plt.scatter(to_print[:, 0], to_print[:, 1])
    plt.savefig('./tmp/tnse.png')


def get_img_3d(kmeans, data_x):
    reduced_data_pca = PCA(n_components=3).fit_transform(data_x)
    pred = kmeans.fit_predict(data_x)

    plt.figure(1)
    plt.clf()
    ax = plt.axes(projection='3d')
    for i in set(pred):
        to_print = reduced_data_pca[pred == i]
        ax.scatter3D(to_print[:, 0], to_print[:, 1], to_print[:, 2], s=5)
    plt.savefig('./tmp/pca_3D.png')

    reduced_data_tsne = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=3).fit_transform(data_x)
    plt.figure(1)
    plt.clf()
    ax = plt.axes(projection='3d')
    for i in set(pred):
        to_print = reduced_data_tsne[pred == i]
        ax.scatter3D(to_print[:, 0], to_print[:, 1], to_print[:, 2], s=5)
    plt.savefig('./tmp/tnse_3D.png')


def get_img_cloud(data_x, name_file, ngram_range=(1, 3)):
    # если df маленький - не ограничивать tfidf в частоте термов, если большой - нормально делать
    #if len(data_x) < 80:
    #    coding = TfidfVectorizer(ngram_range=(1, 2))
    #else:
    #    coding = TfidfVectorizer(min_df=0.1, max_df=0.9, ngram_range=(1, 2))
    # coding = TfidfVectorizer(ngram_range=ngram_range)
    coding = CountVectorizer(ngram_range=ngram_range, min_df=5, max_df=0.99)

    values = coding.fit_transform(data_x).toarray().sum(axis=1)
    keys = coding.get_feature_names_out()
    freq_x = {key_x: val_x for key_x, val_x in zip(keys, values)}

    cloud = WordCloud(width=1000, height=600, margin=40, max_words=80, random_state=42, colormap='Pastel1',
                      prefer_horizontal=0.99,   # процент перевернутых слов
                      min_word_length=4,        # минимальная длина слова
                      collocation_threshold=10  # минимальное значение вероятности Даннинга, чтобы было биграммой
                      ).generate_from_frequencies(freq_x)
    cloud.to_file(f'./tmp/{name_file}.png')
