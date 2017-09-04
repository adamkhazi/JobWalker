from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
import Utilities
import logging as log

class Feature_Doc2Vec:

    def getCosineSimilarity(self, source_df, source_columnName, target_df, target_columnName):
        '''
        Compute the cosine similarity between 2 column
        :param source_df: A dataframe
        :param source_columnName: Column to use
        :param target_df: B dataframe
        :param target_columnName: Column to use
        :return: Similarity score
        '''
        start_time = time.time()
        try:
            model = Doc2Vec.load('model/doc2vec_trainedvocab.d2v')
        except FileNotFoundError:
            print("File not found. Do training. Takes > 20min")
            model = self.__trainModel(source_df, source_columnName, target_df, target_columnName)

        log.debug("Feature_Doc2Vec load model took: %s minutes" % round(((time.time() - start_time) / 60), 2))



        if str('doc2vec_' + source_columnName + '_vector') in source_df.columns:
            search_vectors = source_df['doc2vec_' + source_columnName + '_vector'].tolist()
        else:
            search_vectors = [np.array(model.infer_vector(Utilities.CVTokeniser(row),
                                                          alpha=0.025, min_alpha=0.001, steps=40))
                              for _, row in source_df[source_columnName].iteritems()]
        print("Feature_Doc2Vec search_vectors took: %s minutes" % round(((time.time() - start_time) / 60), 2))

        if str('doc2vec_' + target_columnName + '_vector') in source_df.columns:
            target_vectors = source_df['doc2vec_' + target_columnName + '_vector'].tolist()
        else:
            target_vectors = [np.array(model.infer_vector(
                Utilities.CVTokeniser(str(target_df[target_columnName].iloc[row].values[0])),
                alpha=0.025, min_alpha=0.001, steps=40))
                              for _, row in source_df.product_idx.iteritems()]
        log.debug("Feature_Doc2Vec target_vectors took: %s minutes" % round(((time.time() - start_time) / 60), 2))

        result = []
        batch_size = 2000
        # Batch it
        for i in range(int(len(source_df)/batch_size)):
            inter_result = cosine_similarity(search_vectors[i * batch_size:(i * batch_size) + batch_size],
                                             target_vectors[i * batch_size:(i * batch_size) + batch_size])

            for i in range(inter_result.shape[0]):
                result.append(inter_result[i][i])

        # Compute the remaining
        start = int(len(source_df)/batch_size)*batch_size
        end = len(source_df)
        inter_result = cosine_similarity(search_vectors[start: end], target_vectors[start: end])

        for j in range(inter_result.shape[0]):
            result.append(float(inter_result[j][j]))
        print("Feature_Doc2Vec Cosine Similarity took: %s minutes" % round(((time.time() - start_time) / 60), 2))

        return result
