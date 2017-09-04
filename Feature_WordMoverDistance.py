from gensim.models.keyedvectors import KeyedVectors
import logging as log
from joblib import Parallel, delayed
import Utilities

class Feature_WordMoverDistance():
    def __init__(self):
        # Load word2vec model using google pretrained vectors
        log.info("Loading Pre-trained vector")
        # Download link for GoogleNews-vectors-negative300.bin
        # https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
        self.model = KeyedVectors.load_word2vec_format('model/GoogleNews-vectors-negative300.bin', binary=True)
        # log.info("Loaded Pre-trained vector. Normalising now")
        # Normalize the embedding vectors
        self.model.init_sims(replace=True)

    def getDistance(self, df, source_columnName, target_columnName):
        log.info("Computing WordMoverDistance now")
        my_list = [(row[source_columnName], row[target_columnName]) for _, row in df.iterrows()]
        my_list = list(Utilities.chunks(my_list, 2500))
        log.debug('Number of chunk: %d' %len(my_list))

        # target_vectors = [self.model.wmdistance(row[target_columnName], row[source_columnName])
        #                   for _, row in df.iterrows()]

        # wmdistance is damn talkative. Disable the logging for now
        l = log.getLogger()
        l.setLevel(log.ERROR)
        target_vectors = Parallel(n_jobs=30)(
            delayed(self.computeDistanceHelper)(self.model, chunk)
            for chunk in my_list)
        l.setLevel(log.DEBUG)
        # target_vectors = Parallel(n_jobs=55)(
        #     delayed(self.model.wmdistance)(row[target_columnName], row[source_columnName]) for _, row in source_df.iterrows())
        result = [r for chunk in target_vectors for r in chunk]
        # print(result[:100])
        return result

    def computeDistanceHelper(self, model, chunk):
        result = []
        for source, target in chunk:
            result.append(model.wmdistance(source, target))
        return result


