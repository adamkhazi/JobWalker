import pandas as pd
import Utilities
import logging as log
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
import Feature_WordMoverDistance
import Feature_TFIDF
import Feature_BM25
import Feature_PMI
from joblib import Parallel, delayed


class FeatureEngineering():
    def __init__(self, opp_df):
        '''
        Init the object
        :param opp_df: Opportunity dataframe
        '''

        # Init Word Mover Distance
        self.wmd = Feature_WordMoverDistance.Feature_WordMoverDistance()
        self.wmd_max = float('Inf')

        # Init TF-IDF
        self.tfidf = Feature_TFIDF.Feature_TFIDF(opp_df, 'content')

        # Init BM25
        self.bm25 = Feature_BM25.Feature_BM25(opp_df)
        self.bm25_max = float('Inf')

        # Init PMI
        self.pmi = Feature_PMI.Feature_PMI(opp_df)
        self.pmi_max = float('Inf')

        self.cv_len_max = float('Inf')
        self.jobdes_len_max = float('Inf')


    def processFeatures(self, df, nWorker = 55):
        '''
        Create the features based on the dataframe
        :param df: Dataframe to use
        :param nWorker: Number of worker thread to use to create the feature
        :return: Dataframe with additional feature columns
        '''
        log.debug('Generating Features')
        # log.debug(df.info())

        log.debug('Computing doc2vec score')
        result = Parallel(n_jobs=nWorker)(delayed(cosine_similarity)
                                                  (np.array(row['app_doc2vec']).reshape(1, 300),
                                                   np.array(row['opp_doc2vec']).reshape(1, 300))
                                                  for index, row in df.iterrows())

        df['doc2vec_score'] = [r[0] for chunk in result for r in chunk]
        df['doc2vec_sqrt_score'] = df.doc2vec_score.map(lambda x: math.sqrt(math.fabs(x)))
        df['doc2vec_sq_score'] = df.doc2vec_score.map(lambda x: math.pow(x, 2))

        log.debug('Computing TF-IDF score')
        df['tfidf_score'] = self.tfidf.getCosineSimilarity(df, 'cv_data', 'content', nWorker=nWorker)
        df['tfidf_sqrt_score'] = df.tfidf_score.map(lambda x: math.sqrt(math.fabs(x)))
        df['tfidf_sq_score'] = df.tfidf_score.map(lambda x: math.pow(x, 2))

        log.debug('Computing BM25 score')
        df['bm25_score'] = self.bm25.getBM25_Score(df, 'cv_data')
        if self.bm25_max == float('Inf'):
            self.bm25_max = df['bm25_score'].max()
            # print('max: ', self.bm25_max)

        df['bm25_score'] = df.bm25_score.map(lambda x: x/self.bm25_max)
        df['bm25_sqrt_score'] = df.bm25_score.map(lambda x: math.sqrt(math.fabs(x)))
        df['bm25_sq_score'] = df.bm25_score.map(lambda x: math.pow(x, 2))

        log.debug('Compute Word Mover Distance')
        df['wmd_score'] = self.wmd.getDistance(df, 'content', 'cv_data')
        if self.wmd_max == float('Inf'):
            self.wmd_max = df['wmd_score'].max()
        df['wmd_score'] = df.wmd_score.map(lambda x: float(x / self.wmd_max))
        df['wmd_score_sqrt'] = df.wmd_score.map(lambda x: math.sqrt(math.fabs(x)))
        df['wmd_score_sq'] = df.wmd_score.map(lambda x: math.pow(x, 2))

        log.debug('Computing CV length')
        df['cv_len'] = df.cv_data.map(lambda x: len(x))
        if self.cv_len_max == float('Inf'):
            self.cv_len_max = df['cv_len'].max()
            # print('cv_len max: ', self.cv_len_max)
        df['cv_len'] = df.cv_len.map(lambda x: float(x/self.cv_len_max))
        df['cv_len_sqrt'] = df.cv_len.map(lambda x: math.sqrt(math.fabs(x)))
        df['cv_len_sq'] = df.cv_len.map(lambda x: math.pow(x, 2))

        log.debug('Computing job description length')
        df['jobdes_len'] = df.content.map(lambda x: len(x))
        if self.jobdes_len_max == float('Inf'):
            self.jobdes_len_max = df['jobdes_len'].max()
            # print('jobdes_len max: ', self.jobdes_len_max)
        df['jobdes_len'] = df.jobdes_len.map(lambda x: float(x / self.jobdes_len_max))
        df['jobdes_len_sqrt'] = df.jobdes_len.map(lambda x: math.sqrt(math.fabs(x)))
        df['jobdes_len_sq'] = df.jobdes_len.map(lambda x: math.pow(x, 2))

        log.debug('Computing PMI score')
        df['pmi_score'] = self.pmi.computePMIColumn(df, targetColname='cv_data')
        if self.pmi_max == float('Inf'):
            self.pmi_max = df['pmi_score'].max()
        df['pmi_score'] = df.pmi_score.map(lambda x: x/self.pmi_max)
        df['pmi_score_sqrt'] = df.pmi_score.map(lambda x: math.sqrt(math.fabs(x)))
        df['pmi_score_sq'] = df.pmi_score.map(lambda x: math.pow(x, 2))

        # log.debug(df.info())
        return df

    def parseOpp(self, data):
        return Utilities.processOpp(data)

def main():
    app_filename = 'data/app_331_doc2vec.csv'
    opp_filename = 'data/opp_331_doc2vec.csv'
    cv_filename = 'data/cv_331.csv'

    app_df = pd.read_csv(app_filename)
    opp_df = pd.read_csv(opp_filename)
    cv_df = pd.read_csv(cv_filename)

    # app_df = app_df.head(10)

    log.debug('Converting str to list')
    opp_df['opp_doc2vec'] = Parallel(n_jobs=55)(delayed(eval)(row.opp_doc2vec) for _, row in opp_df.iterrows())
    app_df['app_doc2vec'] = Parallel(n_jobs=55)(delayed(eval)(row.app_doc2vec) for _, row in app_df.iterrows())
    cv_df['cv_data'] = Parallel(n_jobs=55)(delayed(eval)(row.cv_data) for _, row in cv_df.iterrows())
    log.debug('Completed conversion')

    log.info('Creating job description "content" column')
    opp_df['content'] = opp_df['title'].map(str) + " " + \
                        opp_df['role_specifics'].map(str) + " " + \
                        opp_df['skills'].map(str)
    # df['title'] = df['title'].map(lambda x: Utilities.processOpp(x))
    # df['role_specifics'] = df['role_specifics'].map(lambda x: Utilities.processOpp(x))
    # df['skills'] = df['skills'].map(lambda x: Utilities.processOpp(x))
    # df['content'] = df['content'].map(lambda x: Utilities.processOpp(x))
    opp_df['content'] = Parallel(n_jobs=55)(delayed(Utilities.processOpp)(row.content) for _, row in opp_df.iterrows())
    # log.debug(cv_df.head(2))

    df = pd.merge(app_df, opp_df, how='left', on='opp_id')
    df = pd.merge(df, cv_df, how='left', on='cv name')

    fe = FeatureEngineering(opp_df)
    fe.processFeatures(df)



if __name__ == "__main__":
    main()