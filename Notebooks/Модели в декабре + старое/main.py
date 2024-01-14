import pandas as pd
from collections import defaultdict
import numpy as np
from functions_from_book import tfidf_coding_2, front_coding, tokenize
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import LatentDirichletAllocation

'''Выделение df coffee'''
# df = pd.read_csv('data/data.csv')
#
# """Рубрики с максимальным количеством ФИРМ"""
# # rubric_set_firm = defaultdict(set)
# # for rub_all_x, name_x in zip(df['rubrics'], df['name_ru']):
# #     for rub_x in set(rub_all_x.split(";")):
# #         rubric_set_firm[rub_x].add(name_x)
# #
# # df_tmp = pd.DataFrame(rubric_set_firm.items())
# # df_tmp.columns = ['Рубрика', 'Кол-во фирм']
# # df_tmp['Кол-во фирм'] = df_tmp['Кол-во фирм'].apply(lambda x: len(x))
# # df_tmp = df_tmp.set_index('Кол-во фирм').sort_index(ascending=False)
#
# """Рубрики с максимальным количеством ОТЗЫВОВ"""
# rubric_name_list = defaultdict(list)
# for rub_all_x, name_x in zip(df['rubrics'], df['name_ru']):
#     for rub_x in set(rub_all_x.split(";")):
#         rubric_name_list[rub_x].append(name_x)
#
#
# df_tmp = pd.DataFrame(rubric_name_list.items())
# df_tmp.columns = ['Рубрика', 'Кол-во отзывов']
# df_tmp['Кол-во отзывов'] = df_tmp['Кол-во отзывов'].apply(lambda x: len(x))
# df_tmp = df_tmp.set_index('Кол-во отзывов').sort_index(ascending=False)
#
# interesting_rub_list = ['Кафе', 'Ресторан', 'Бар, паб', 'Быстрое питание', 'Кофейня', 'Пиццерия',
#                         'Пекарня', 'Столовая', 'Суши-бар', 'Кофе с собой']
#
# # coffee = df[df['rubrics'].isin(interesting_rub_list)] только уникальные включения ("Кафе, Кофейня" не будет)
# mask = df['rubrics'].apply(lambda x: True if any(rub_x in x for rub_x in interesting_rub_list) else False)
# coffee = df[mask]
# coffee.index = np.arange(coffee.shape[0])
# coffee.info()
#
# # сохранение датафрейма
# coffee.to_csv('data/coffee.csv', index=False)

coffee = pd.read_csv('data/coffee.csv')

# возьмем в корпус первые 10к отзывов
corpus = [coffee['text'][i] for i in range(10000)]

# построим doc2vec
corpus = [list(tokenize(doc)) for doc in corpus]
corpus = [
    TaggedDocument(words, ['d{}'.format(idx)])
    for idx, words in enumerate(corpus)]
model = Doc2Vec(corpus, vector_size=20, min_count=20)
print("Модель Doc2Vec построена")

vectorize_docs = model.docvecs.vectors

# избавимся от минимальных значений - мин-макс преобразование
min_max = MinMaxScaler()
vectorize_docs = min_max.fit_transform(vectorize_docs)


# Возьмем LDA как базу
LDA = LatentDirichletAllocation(n_components=5,
                                random_state=42,
                                n_jobs=-1)
LDA.fit(vectorize_docs)
print("Модель LDA построена")

a = LDA.transform(vectorize_docs)
b = pd.DataFrame(LDA.transform(vectorize_docs))


def func(x):
    return x.sort_values(ascending=False).index[0]

b.apply(func, axis=1)

print("Конец")