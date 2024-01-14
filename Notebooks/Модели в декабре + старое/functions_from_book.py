import nltk
import string
from collections import defaultdict
from nltk import pos_tag, sent_tokenize, wordpunct_tokenize

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer

import gensim


def words(text):
    '''Выделение слов из текста'''
    for token in wordpunct_tokenize(text):
        yield token


def tokenize_part(text):
    '''Выделение частей речи для каждого слова в тексте
    [ВНИМАНИЕ!] необходимо download punct'''
    yield [pos_tag(wordpunct_tokenize(sent)) for sent in sent_tokenize(text)]


def tokenize(text):
    '''Выделение смысловой части слова'''
    stem = nltk.stem.SnowballStemmer('russian')
    text = text.lower()
    for token in words(text):
        if token in string.punctuation: continue
        yield stem.stem(token)


def template(corpus, mode='skl'):
    if mode == 'nltk':
        pass
    elif mode == 'skl':
        pass
    elif mode == 'gen':
        pass
    else:
        raise "В функции template выбран неверный режим"


def vectorize(corpus, mode='skl', features=None):
    if mode == 'nltk':
        if features is None:
            features = defaultdict(int)
        features = defaultdict(int)
        for token in tokenize(corpus[0]):
            features[token] += 1
        return features
    elif mode == 'skl':
        vectorizer = HashingVectorizer()  # CountVectorizer()- обычный, HashingVectorizer - хешируемый
        vectors = vectorizer.fit_transform(corpus)
        return vectors, vectorizer
    elif mode == 'gen':
        """Не получается векторизовать"""
        corpus = [tokenize(doc) for doc in corpus]
        id2word = gensim.corpora.Dictionary(corpus)
        vectors = [id2word.doc2bow(doc) for doc in corpus]
        return vectors
    else:
        raise "В функции vecrotize выбран неверный режим"


def front_coding(corpus: list[str], mode='skl', return_model=False):
    """Просто наличие/отсутствие слова в документе"""
    if mode == 'nltk':
        return {token: True for token in corpus[0]}
    elif mode == 'skl':
        """Просто наличие/отсутствие слова в документе
        Можно использовать CountVectorizer(binary = True)"""
        freq = CountVectorizer(binary=True)
        corpus = freq.fit_transform(corpus)
        if return_model:
            return corpus, freq
        return corpus
    elif mode == 'gen':
        corpus = [tokenize(doc) for doc in corpus]
        id2word = gensim.corpora.Dictionary(corpus)
        vectors = [[(token[0], 1) for token in id2word.doc2bow(doc)] for doc in corpus]
        return vectors
    return 0


def tfidf_coding_2(corpus, mode='skl', return_model=False):
    if mode == 'nltk':
        from nltk.text import TextCollection
        corpus = [list(words(doc)) for doc in corpus]

        texts = TextCollection(corpus)
        for doc in corpus:
            return {term: texts.tf_idf(term, doc) for term in doc}
    elif mode == 'skl':
        """skl_tfidf(list[text])"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer()
        corpus = tfidf.fit_transform(corpus)
        if return_model:
            return corpus, tfidf
        return corpus
    elif mode == 'gen':
        corpus = [tokenize(doc) for doc in corpus]
        lexicon = gensim.corpora.Dictionary(corpus)
        tfidf = gensim.models.TfidfModel(dictionary=lexicon, normalize=True)
        vectors = [tfidf[lexicon.doc2bow(doc)] for doc in corpus]
        return vectors


