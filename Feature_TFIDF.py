from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import logging as log
import Utilities
from joblib import Parallel, delayed

class Feature_TFIDF():
    def __init__(self, train_df, columnName):
        start_time = time.time()
        log.info("Init TF IDF")
        self.tfidf = TfidfVectorizer(tokenizer=Utilities.CVTokeniser, smooth_idf=True, use_idf=True, sublinear_tf=True,
                                stop_words=None, lowercase=True, ngram_range=(1, 1), norm='l2')
        # print("TfidfVectorizer init took: %s minutes" % round(((time.time() - start_time) / 60), 2))
        train_df['tmpColumn'] = train_df[columnName].map(lambda  x: self.__word_joiner(x))

        self.tfidf.fit_transform(train_df['tmpColumn'])
        train_df = train_df.drop(['tmpColumn'], axis=1, inplace=True)
        log.debug('TFIDF fit completed')

    def __word_joiner(self, s):
        return " ".join([word for word in s])

    def computeScoreHelper(self, chunk):
        source = [s for s, _ in chunk]
        target = [t for _, t in chunk]

        score_queries = self.tfidf.transform(source)
        score_target = self.tfidf.transform(target)
        cos_result = cosine_similarity(score_queries, score_target)

        result = []
        for i in range(cos_result.shape[0]):
            result.append(cos_result[i][i])

        return result

    def getCosineSimilarity(self, df, source_columnName, target_columnName, nWorker):
        df['tmpSource'] = df[source_columnName].map(lambda x: self.__word_joiner(x))
        df['tmpTarget'] = df[target_columnName].map(lambda x: self.__word_joiner(x))

        my_list = [(row['tmpSource'], row['tmpTarget']) for _, row in df.iterrows()]
        my_list = list(Utilities.chunks(my_list, 5000))

        target_vectors = Parallel(n_jobs=nWorker)(
            delayed(self.computeScoreHelper)(chunk)
            for chunk in my_list)

        result = [r for chunk in target_vectors for r in chunk]
        df = df.drop(['tmpTarget', 'tmpSource'], axis=1, inplace=True)

        return result
