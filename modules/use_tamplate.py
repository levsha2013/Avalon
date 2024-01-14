from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from modules.pipeline_tamplate import *

if __name__ == '__main__':
    df = pd.read_csv('../data/coffee_prep.csv')['text'].iloc[:10000]

    #vectorizer = TfidfVectorizer(analyzer='word',
    vectorizer = CountVectorizer(analyzer='word',
                                 min_df=10,
                                 max_df=0.95,
                                 ngram_range=(1, 3))


    LDA_model = LatentDirichletAllocation(n_components=3,
                                          learning_method='online',
                                          random_state=42,
                                          #n_jobs=-1
                                          )

    LDA_model, vectorizer = fit_model(df, vectorizer_x=vectorizer, model_type=LDA_model, verbose=True)
    result = pd.DataFrame(show_topics(vectorizer, LDA_model, 10))

    feature_names = vectorizer.get_feature_names_out()

    for topic_idx, topic in enumerate(LDA_model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i] for i in topic.argsort()[:-16:-1]]))
        print()

    converg = get_divergention(LDA_model, vectorizer, df)
    print(f"Дивергенция KL = {converg}")
