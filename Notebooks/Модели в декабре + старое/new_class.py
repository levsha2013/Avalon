import os
import string
import pandas as pd

import nltk
from nltk import pos_tag, sent_tokenize, wordpunct_tokenize

from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader

CAT_PATTERN = r'([a-z_\s]+)/.*'
DOC_PATTERN = r'(?!\.)[a-z_\s]+/[a-f0-9]+\.json'
TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'p', 'li']


class HTMLCorpusReader(CategorizedCorpusReader, CorpusReader):
    def __init__(self, root, fileids=DOC_PATTERN, encoding='utf8',
                 tags=TAGS, **kwargs):
        """
        Инициализирует объект чтения корпуса.
        Аргументы, управляющие классификацией
        (``cat_pattern``, ``cat_map`` и ``cat_file``), передаются
        в конструктор ``CategorizedCorpusReader``. остальные аргументы
        передаются в конструктор ``CorpusReader``.
        """
        # Добавить шаблон категорий, если он не был передан в класс явно.
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_pattern'] = CAT_PATTERN

        # Инициализировать объекты чтения корпуса из NLTK
        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, root, fileids, encoding)

        # Сохранить теги, подлежащие извлечению.
        self.tags = tags

    def resolve(self, fileids, categories):
        """
        Возвращает список идентификаторов файлов или названий категорий,
        которые передаются каждой внутренней функции объекта чтения корпуса.
        Реализована по аналогии с ``CategorizedPlaintextCorpusReader`` в NLTK.
        """

        if fileids is not None and categories is not None:
            raise ValueError(
                "Specify fileids or categories, not both")
        if categories is not None:
            return self.fileids(categories)
        return fileids

    def sizes(self, fileids=None,
              categories=None):
        """
        Возвращает список кортежей, идентификатор файла и его размер.
        Эта функция используется для выявления необычно больших файлов
        в корпусе.        """
        # Получить список файлов
        fileids = self.resolve(fileids, categories)
        # Создать генератор, возвращающий имена и размеры файлов
        for path in self.abspaths(fileids):
            yield path, os.path.getsize(path)

    def sents(self, fileids=None, categories=None):
        """
        Использует встроенный механизм для выделения предложений из
        абзацев. Обратите внимание, что для парсинга разметки HTML
        этот метод использует BeautifulSoup.
        """
        for sentence in sent_tokenize(self):
            yield sentence

    def words(self, fileids=None, categories=None):
        """Выделяе слова из текста"""
        for token in wordpunct_tokenize(self):
            yield token

    def tokenize(self, fileids=None, categories=None):
        """
        Сегментирует, лексемизирует и маркирует документ в корпусе.
        """
        yield [
            pos_tag(wordpunct_tokenize(sent))
            for sent in sent_tokenize(self)
        ]