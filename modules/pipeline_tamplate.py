import re
import nltk
import time
import pymorphy2
import numpy as np
import pandas as pd
from gensim.models import CoherenceModel
import gensim.corpora as corpora
from nltk.stem.snowball import SnowballStemmer

# nltk.download('punkt')


def preprocess_text(data, morph=pymorphy2.MorphAnalyzer(), stemmer=SnowballStemmer(language="russian")):
    """
    Применяет функцию preprocess_text к входным данным (просто элементу текста)
    Лемматизация закомментирована, потому что есть стемминг (приведение слов в одной форме (муж. р, им. пад.)
    :param data: - данные для преобразования
    :param morph:
    :return:
    """
    if data is not np.NaN:
        """Очистка текста и удаление стоп-слов"""
        # замена ё на е // оставляем буквы, пробелы и знаки препинания // удаление пробелов
        text = re.sub('ё', 'е', data.lower())
        text = re.sub(r"([.,!?])", r'\1', text)
        text = re.sub(r"[^а-яА-Я\s]+", r'', text)
        text = text.strip()

        # удаление стоп слов и слов меньше длины 3
        # text = " ".join([w for w in text.split() if w not in stop_words])
        # text = [w for w in text.split() if len(w) >= 3]

        # result = " ".join([morph.parse(word_x)[0].normal_form for word_x in data.split()])  # лемматизация
        result = " ".join([stemmer.stem(word) for word in nltk.word_tokenize(text)])    # стемминг
        return result
    return ""


def transform_data(data: pd.Series) -> list:
    """
    Вызов preprocess_text и удаление None, '' для всех элементов Series
    :param data: - pd.Series с текстами для обработки
    :return: - список преобразованных текстов
    """
    transform_lst = [preprocess_text(data=i) for i in data]
    transform_lst = [x for x in transform_lst if x not in [None, '']]
    return transform_lst


def fit_model(input_data=None, vectorizer_x=None, model_type=None, verbose=False, res_vectorizer=None):
    """
    На основании переданного vectorizer и model_type строит тематическую модель
    :param input_data: - входной pd.Series с текстами
    :param vectorizer_x: - векторизатор (с указанием параметров)
    :param model_type: - модель, принимающая результаты векторайзера (с указанием параметров)
    :param verbose: - стоит ли выводить процессы в консоль
    :return: обученая модель и векторайзер
    """
    start_fit = time.time()
    # result = transform_data(input_data)
    result = input_data
    if res_vectorizer is None:
        #if verbose: print("Конец трансформации. Начало векторизации")

        #res_vectorizer = vectorizer_x.fit_transform(result)
        if verbose: print("Конец векторизации. ", end="")
    print("Начало обучения", end=" -- ")

    model_type.fit(res_vectorizer)
    if verbose: print("Конец обучения.", end=" -- ")

    metrics = {
        'Правдоподобие': model_type.score(res_vectorizer),
        'Perplexity': model_type.perplexity(res_vectorizer)
    }
    if verbose:
        # for i, j in metrics.items():
        #     print(i, '--', round(j, 2))
        print(round(time.time() - start_fit), "сек")
    return model_type, vectorizer_x, metrics


def show_topics(vectorizer_x=None, model=None, n_words=20) -> list[list]:
    """
    На основе векторизатора и модели показывает топ 20 слов для каждой модели
    :param vectorizer_x: - векторайзер ()
    :param model: - модель (LDA например)
    :param n_words: - какой топ слов выводить?
    :return: - для каждой темы топ слов
    """
    feature_names = np.array(vectorizer_x.get_feature_names_out())
    top_words = []

    for topic_weights in model.components_:
        top_keywords_locs = (-topic_weights).argsort()[:n_words]
        top_words.append(feature_names.take(top_keywords_locs))
    return top_words


def get_divergention(model, vectorizer, texts: list[str], n_top_words: int = 20, verbose=False) -> float:
    """
    На основе построенной модели, текста строит в окне n_top_words дивергенцию Кульбака-Лейбница
    :param model: - построенная LDA модель
    :param texts: - входной текст (iterable[text])
    :param n_top_words: - окно для вычисления дивергенции (20 по умолчанию)
    :return: Значение дивергенции Кульбака-Лейбница
    """
    if verbose:
        print("В функции расчета дивергенции ...")
    # колчиество тем
    topics = model.components_

    # получение токенов
    texts = [[word for word in doc.split()] for doc in texts]

    dictionary = corpora.Dictionary(texts)
    # dictionary = list(vectorizer.vocabulary_.keys())
    # corpus = [dictionary.doc2bow(text) for text in texts]

    feature_names = [dictionary[i] for i in range(len(dictionary))]

    top_words = []
    for topic in topics:
        top_words.append(
            [feature_names[i] for i in topic.argsort()[:-n_top_words - 1: -1]]
        )
    if verbose:
        print("\tПостроение модели")
    coh = CoherenceModel(topics=top_words,
                         texts=texts,
                         dictionary=dictionary,
                         coherence='c_v'
                         )
    if verbose:
        print("\tРасчет когерентности ... ", end="")
    res = coh.get_coherence()
    if verbose:
        print("OK")
    return res
