from gensim.models import Doc2Vec
import Utilities
import collections
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging as log
import DataReader
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import Evaluator
from joblib import Parallel, delayed
import Feature_Engineering
from xgboost import *

class ContentBasedDoc2Vec():
    def __init__(self, train_app_df, test_app_df, opp_df, train_model_name):
        '''
        Init the object
        :param train_app_df: Training dataframe
        :param test_app_df: Test dataframe
        :param opp_df: Opportunity dataframe
        :param train_model_name: Doc2vec model filename
        '''
        log.info('## Init Content-based Recommender ##')

        self.train_app_df = train_app_df
        self.test_app_df = test_app_df
        self.opp_df = opp_df
        self.train_model_name = train_model_name

    def getRecommendationResult(self, topN=10):
        '''
        Get personalised recommendation all candidates in test_app_df

        :param topN: Number of recommendation per candidate
        :return:
        '''

        log.info('## Computing Content-based Recommendation ##')

        # List of candidates that recommendation is required
        candidate_id_list = self.test_app_df['candidate_id'].unique()

        try:
            doc2vecmodel = Doc2Vec.load(self.train_model_name)
        except FileNotFoundError:
            log.error('%s model not found. Please create the model first.' %self.train_model_name)

        # Infer vector for all the open opportunities description
        log.info('  Inferring Opp vector')
        self.opp_df['content'] = self.opp_df['title'].map(str) + " " + \
                                 self.opp_df['role_specifics'].map(str) + " " + \
                                 self.opp_df['skills'].map(str)

        # Create DF for open opportunity
        open_opp_df = self.opp_df[self.opp_df['is_open'] == 1][['opp_id', 'content']].copy()
        open_opp_vectors = [np.array(
            doc2vecmodel.infer_vector(Utilities.processOpp(row),
                                      alpha=0.025, min_alpha=0.001, steps=15))
            for _, row in open_opp_df['content'].iteritems()]

        open_opp_df['content_vector'] = open_opp_vectors



        # Infer vector for all the candidate's CV
        log.info('  Inferring candidate CV vector')
        recommendationDict = collections.defaultdict(list)
        for target_candidate_id in candidate_id_list:
            # Assume candidate can have multiple CVs
            CVs = self.test_app_df[self.test_app_df.candidate_id == target_candidate_id]['cv name'].unique()
            CVs = np.append(CVs, self.train_app_df[self.train_app_df.candidate_id == target_candidate_id]['cv name'].unique())
            CVs = np.unique(CVs)
            # print(target_candidate_id)
            # print(CVs)
            cv_folders = [Utilities.convertCVName2FolderStructure(cv) for cv in CVs]

            candiate_vectors = []
            for fname in cv_folders:
                try:
                    with open(fname, encoding='utf8') as cv:
                        data = cv.read()

                except UnicodeDecodeError:
                    # print('UnicodeDecodeError Error')
                    try:
                        with open(fname, encoding='ISO-8859-1') as cv:
                            data = cv.read()
                    except Exception:
                        log.error(Exception)
                except FileNotFoundError:
                    log.error('  CV not found error: %s' %fname)
                    continue


                if len(data) > 500:
                    vector = doc2vecmodel.infer_vector(Utilities.processCV(data),
                                                       alpha=0.025, min_alpha=0.001, steps=15)

                    candiate_vectors.append(vector)

                else:
                    log.debug('Candidate %f.0 CV length is shorter than 100 bytes: %s' %(target_candidate_id, fname))

            result = cosine_similarity(candiate_vectors, open_opp_vectors)

            # If candidate has more than 1 CV, we use the highest similarity score
            maxed = np.amax(result, axis=0)

            topN_index = maxed.argsort()[-topN:][::-1]

            for index in topN_index:
                recommendationDict[target_candidate_id].append(open_opp_df['opp_id'].iloc[index])

        log.debug('Recommendation Result: %s' % recommendationDict)
        return recommendationDict

class ContentBasedClassifier:
    def __init__(self, train_app_df, test_app_df, opp_df, cv_df):
        self.train_app_df = train_app_df
        self.test_app_df = test_app_df
        self.opp_df = opp_df
        self.cv_df = cv_df


        log.debug('Converting str to list')
        self.opp_df['opp_doc2vec'] = Parallel(n_jobs=55)(delayed(eval)(row.opp_doc2vec)
                                                         for _, row in self.opp_df.iterrows())
        self.train_app_df['app_doc2vec'] = Parallel(n_jobs=55)(delayed(eval)(row.app_doc2vec)
                                                               for _, row in self.train_app_df.iterrows())
        self.test_app_df['app_doc2vec'] = Parallel(n_jobs=55)(delayed(eval)(row.app_doc2vec)
                                                              for _, row in self.test_app_df.iterrows())
        self.cv_df['cv_data'] = Parallel(n_jobs=55)(delayed(eval)(row.cv_data) for _, row in self.cv_df.iterrows())
        log.debug('Completed conversion')

        log.info('Creating job description "content" column')
        opp_df['content'] = opp_df['title'].map(str) + " " + \
                            opp_df['role_specifics'].map(str) + " " + \
                            opp_df['skills'].map(str)

        opp_df['content'] = Parallel(n_jobs=55)(
            delayed(Utilities.processOpp)(row.content) for _, row in opp_df.iterrows())

        self.featureHelper = Feature_Engineering.FeatureEngineering(self.opp_df)

    def getRecommendationResult(self, topN=10):
        log.info('## Computing Content-based Classifier (Keras) ##')

        # Create model train data
        x_train, y_train = self.__prepareTrainData(isBalance=False)

        # Open Opp DF
        open_opp_df = self.opp_df[self.opp_df.is_open == 1].copy()

        # List of candidates that recommendation is required
        candidate_id_list = self.test_app_df['candidate_id'].unique()
        cv_list = self.test_app_df['cv name'].unique()

        self.cv_df = self.cv_df[self.cv_df['cv name'].isin(cv_list)]
        self.train_app_df = ''
        self.opp_df = ''
        log.debug('Length of cv_df: %d' % self.cv_df.shape[0])

        candidate_df_list = self.__prepareTestData(open_opp_df, candidate_id_list)
        # log.debug('No of unique candidate: %d'% len(candidate_id_list))
        # log.debug('No of candidate_df_list: %d'% len(candidate_df_list))

        log.info('Training model now. ')
        self.model = self.trainModel(x_train, y_train, modelType='keras')
        log.info('Model created. Computing recommendation')

        recommendationDict = collections.defaultdict(list)
        for df in candidate_df_list:
            can_id = float(df.head(1).candidate_id.values[0])
            # log.debug('can_id: %d' %can_id)
            # log.debug(type(can_id))

            # Pick the relevant column from the DF
            nparr = self.getModelDataFormat(df, False)

            pred_score = self.model.predict(nparr)

            df['pred_score'] = pred_score
            df = df.sort_values('pred_score', ascending=False)

            # log.debug(df[['candidate_id', 'opp_id', 'tfidf_score', 'pred_score']].head(5))
            opp_list = []
            for _, row in df.iterrows():
                if len(opp_list) < topN:
                    if row.opp_id not in opp_list:
                        opp_list.append(row.opp_id)
                else:
                    break

            recommendationDict[can_id] = opp_list

        return recommendationDict, candidate_df_list


    def getModelDataFormat(self, df, isTrain=True):
        # Pick the relevant column from the DF
        if isTrain:
            # tmp = [np.concatenate([[row.doc2vec_score], [row.doc2vec_sqrt_score], [row.doc2vec_sq_score],
            #                        [row.tfidf_score], [row.tfidf_sqrt_score], [row.tfidf_sq_score], [row.app_state]])
            #        for index, row in df.iterrows()]
            tmp = [np.concatenate([[row.doc2vec_score], [row.tfidf_score], [row.bm25_score], [row.cv_len], [row.cv_len_sqrt], [row.cv_len_sq],
                                   [row.pmi_score], [row.pmi_score_sqrt], [row.pmi_score_sq], [row.jobdes_len], [row.jobdes_len_sqrt], [row.jobdes_len_sq],
                                   [row.app_state]])
                   for index, row in df.iterrows()]
        else:
            # tmp = [np.concatenate([[row.doc2vec_score], [row.doc2vec_sqrt_score], [row.doc2vec_sq_score],
            #                        [row.tfidf_score], [row.tfidf_sqrt_score], [row.tfidf_sq_score]])
            #        for index, row in df.iterrows()]
            tmp = [np.concatenate([[row.doc2vec_score], [row.tfidf_score], [row.bm25_score], [row.cv_len], [row.cv_len_sqrt], [row.cv_len_sq],
                                   [row.pmi_score], [row.pmi_score_sqrt], [row.pmi_score_sq], [row.jobdes_len], [row.jobdes_len_sqrt], [row.jobdes_len_sq]])
                   for index, row in df.iterrows()]

        return np.array(tmp)

    def __prepareTestData(self, open_opp_df, candidate_id_list):
        log.info('Creating test feature.')
        candidate_df_list = Parallel(n_jobs=20)(delayed(self.computeCandidateModelData)
                                                (self.test_app_df, self.cv_df, open_opp_df,
                                                 self.featureHelper, candidate_id)
                                                for candidate_id in candidate_id_list)
        return candidate_df_list


    def __prepareTrainData(self, isBalance):
        log.info('Creating train feature. Takes a while.')
        df = pd.merge(self.train_app_df, self.opp_df, how='left', on='opp_id')
        df = pd.merge(df, self.cv_df, how='left', on='cv name')

        count_dict = collections.Counter(df['app_state'])
        log.debug('Original App_state distribution: %s' % str(count_dict))

        # Rebalance between apply vs interview/offer
        if isBalance:
            applied_app_id_list = list(df[df['app_state'] == 1].app_id)
            interview_offer_app_id_list = list(df[df['app_state'] > 1].app_id)

            num_applied = (count_dict[2] + count_dict[3]) * 8
            np.random.seed(18)
            test_app_id_list = list(np.random.choice(applied_app_id_list, num_applied, replace=False))
            test_app_id_list = test_app_id_list + interview_offer_app_id_list
            df = df[df.app_id.isin(test_app_id_list)].copy()
            count_dict = collections.Counter(df['app_state'])
            log.debug('After Balancing App_state distribution: %s' % str(count_dict))

        df = self.featureHelper.processFeatures(df)

        # log.debug(df[['doc2vec_score', 'tfidf_score', 'app_state']].head(1))
        # Pick the relevant column from the DF
        nparr = self.getModelDataFormat(df, True)

        # log.debug(nparr.shape)
        x_train = nparr[:, :(nparr.shape[1] - 1)]
        y_train = nparr[:, (nparr.shape[1] - 1):]
        # y_train = (y_train-1.0)/2.0

        # Remap the app_state to 0 and 1
        # 0 == candidate was not preferred by employer as not offered or interviewed
        # 1 == candidate preferred by employer as offered or interviewed
        # palette must be given in sorted order
        palette = [1, 2, 3]
        # key gives the new values you wish palette to be mapped to.
        key = np.array([0, 1.0, 1.0])
        index = np.digitize(y_train.ravel(), palette, right=True)
        y_train = key[index].reshape(y_train.shape)

        log.debug('x_train.shape: ' + str(x_train.shape))
        log.debug('y_train.shape: ' + str(y_train.shape))

        return x_train, y_train


    # def __prepareTrainData_old(self):
    #     # Prepare training data
    #     groubyCanID = self.train_app_df.groupby('candidate_id')['opp_id']
    #     oppLookup_df = self.opp_df[['opp_id', 'opp_doc2vec']]
    #     oppLookup_dict = oppLookup_df.set_index('opp_id')['opp_doc2vec'].to_dict()
    #
    #     def averageVec(opp_list):
    #         if len(opp_list) > 1:
    #             vec_list = []
    #             for opp_id in opp_list:
    #                 vec = oppLookup_dict[opp_id]
    #                 vec_list.append(vec)
    #             vec_array = array(vec_list)
    #             vec_array = vec_array.mean(axis=0)
    #             return vec_array
    #         elif len(opp_list) == 1:
    #             return [0] * 300
    #         else:
    #             log.error('Fatal##############################################')
    #
    #     candidateAveOppVecDict = {canID: averageVec(opp_list) for canID, opp_list in groubyCanID}
    #
    #     log.debug('Total unique candidate: %d' % len(self.train_app_df['candidate_id'].unique()))
    #     log.debug('Total length of candidateAveOppVecDict: %d' % len(candidateAveOppVecDict))
    #
    #     appliedOpp_df = pd.DataFrame(columns=['candidate_id', 'appliedAveOppVec'])
    #     appliedOpp_df['candidate_id'] = candidateAveOppVecDict.keys()
    #     appliedOpp_df['appliedAveOppVec'] = candidateAveOppVecDict.values()
    #
    #     final_df = pd.merge(self.train_app_df, appliedOpp_df, how='left', on='candidate_id')
    #     final_df = pd.merge(final_df, self.opp_df, how='left', on='opp_id')
    #
    #     # log.debug(final_df.info())
    #     final_df['doc2vec_score'] = [cosine_similarity(np.array(row['app_doc2vec']).reshape(1,-1),
    #                                                    np.array(row['opp_doc2vec']).reshape(1,-1))
    #                                  for index, row in final_df.iterrows()]
    #     log.debug(final_df.info())
    #     log.debug(final_df.head(5).doc2vec_score)
    #
    #     # Rebalance between apply vs interview/offer
    #     if False:
    #         applied_app_id_list = list(final_df[final_df['app_state'] == 1].app_id)
    #         interview_offer_app_id_list = list(final_df[final_df['app_state'] > 1].app_id)
    #
    #         count_dict = collections.Counter(final_df['app_state'])
    #         log.debug('Original App_state distribution: %s' % str(count_dict))
    #
    #         num_applied = (count_dict[2]+count_dict[3])*8
    #         np.random.seed(18)
    #         test_app_id_list = list(np.random.choice(applied_app_id_list, num_applied, replace=False))
    #         test_app_id_list = test_app_id_list + interview_offer_app_id_list
    #         final_df = final_df[final_df.app_id.isin(test_app_id_list)].copy()
    #         count_dict = collections.Counter(final_df['app_state'])
    #         log.debug('After Balancing App_state distribution: %s' % str(count_dict))
    #
    #
    #     # final_df = final_df[['app_id', 'appliedAveOppVec', 'app_doc2vec', 'opp_doc2vec', 'app_state']]
    #     # log.debug(final_df.head(5))
    #     # log.debug(final_df.info())
    #
    #     # index_list = []
    #     # for index, row in final_df.iterrows():
    #     #     if list(row['opp_doc2vec'])==list(row['appliedAveOppVec']):
    #     #         index_list.append(index)
    #     #         row['appliedAveOppVec'] = [0.0] * 300
    #     #         print('No more###############################################################')
    #     # print(final_df[final_df['opp_doc2vec']==final_df['appliedAveOppVec']].head(3))
    #
    #     # for index in index_list:
    #     #     final_df.set_value(index, 'appliedAveOppVec', [0.0] * 300)
    #
    #     # Pick the relevant column from the DF
    #     tmp = [np.concatenate([row.opp_doc2vec, row.appliedAveOppVec, row.app_doc2vec,
    #                            row.doc2vec_score[0], [row.app_state]])
    #            for index, row in final_df.iterrows()]
    #     # tmp = [np.concatenate([row.opp_doc2vec, row.appliedAveOppVec, row.app_doc2vec, [row.app_state]])
    #     #        for index, row in final_df.iterrows()]
    #     # tmp = [np.concatenate([row.opp_doc2vec, row.app_doc2vec, [row.app_state]])
    #     #        for index, row in final_df.iterrows()]
    #
    #     # log.debug(tmp[:5])
    #
    #     # count = 0
    #     # for index, row in final_df.iterrows():
    #     #     if row.app_id == 107828:
    #     #         print('############# Found index: %d' %count)
    #     #         break
    #     #     count += 1
    #     # print(tmp[count])
    #
    #     # print(tmp[0])
    #     nparr = np.array(tmp)
    #     print(nparr.shape)
    #     x_train = nparr[:,:(nparr.shape[1]-1)]
    #     y_train = nparr[:,(nparr.shape[1]-1):]
    #     # y_train = (y_train-1.0)/2.0
    #
    #     # Remap the app_state to 0 and 1
    #     # 0 == candidate was not preferred by employer as not offered or interviewed
    #     # 1 == candidate preferred by employer as offered or interviewed
    #     # palette must be given in sorted order
    #     palette = [1, 2, 3]
    #     # key gives the new values you wish palette to be mapped to.
    #     key = np.array([0, 1.0, 1.0])
    #     index = np.digitize(y_train.ravel(), palette, right=True)
    #     y_train = key[index].reshape(y_train.shape)
    #
    #     log.debug('x_train.shape: ' + str(x_train.shape))
    #     log.debug('y_train.shape: ' + str(y_train.shape))
    #     # print(y_train[:100])
    #     # print(y_train.sum())
    #
    #     return x_train, y_train

    def trainModel(self, x_train, y_train, modelType):
        if modelType == 'keras':
            log.debug('Creating Keras Model')
            # x_train, y_train = self.__prepareTrainData()

            # create model
            model = Sequential()

            model.add(Dense(x_train.shape[1], input_dim=x_train.shape[1], activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(6, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(4, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))

            # Compile model
            optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

            # Fit the model
            model.fit(x_train, y_train, epochs=15, batch_size=16, verbose=2, shuffle=True)

            # evaluate the model
            scores = model.evaluate(x_train, y_train)
            log.info("\nModel %s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        elif modelType == 'xgboost':
            log.debug('Creating XGBoost Model')
            model = XGBRegressor(learning_rate=0.1, silent=True, objective='binary:logistic', nthread=-1,
                                       gamma=0.9,
                                       min_child_weight=2, max_delta_step=0, subsample=0.9, colsample_bytree=0.7,
                                       colsample_bylevel=1, reg_alpha=0.0009, reg_lambda=1, scale_pos_weight=1,
                                       base_score=0.5, seed=0, missing=None, max_depth=5, n_estimators=100)

            model.fit(x_train, y_train)

        return model

    def computeCandidateModelData(self, test_app_df, cv_df, open_opp_df, featureHelper, candidate_id):
        log.debug('Computing Candidate: %d' %candidate_id)
        df = test_app_df[test_app_df.candidate_id == candidate_id].copy()
        df = pd.merge(df, cv_df, how='left', on='cv name')
        df = Utilities.df_crossjoin(df, open_opp_df)

        df = featureHelper.processFeatures(df, nWorker=1)

        return df

log.basicConfig(level=log.DEBUG,
                # filemode='w',
                # filename='01logging.log',
                format='%(asctime)s; %(module)s; %(funcName)s; %(levelname)s; %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')

def main():
    client_id = '422'
    datareader = DataReader.CSVReader('data/app_'+client_id+'_doc2vec.csv',
                                      'data/opp_'+client_id+'_doc2vec.csv')
    cv_filename = 'data/cv_'+client_id+'.csv'
    train_app_df, gold_app_df, opp_df = datareader.getTrainNTestData_RandomCandidate(number_of_test_opp=30,
                                                                                     number_of_test_candidate=30,
                                                                                     seed=3)

    cv_df = pd.read_csv(cv_filename)
    # Drop the 'answer' columns
    test_app_df = gold_app_df.drop(['opp_id', 'app_state', 'app_id', 'app_source_json'], axis=1)

    model = ContentBasedClassifier(train_app_df, test_app_df, opp_df, cv_df)
    result,_ = model.getRecommendationResult(30)

    log.info('Computing score on the recommendation.')
    evaluate = Evaluator.RecommenderEvaluator()
    scoreAll_df = evaluate.computeResult(result, gold_app_df, onlyInterviewOffer=False)
    scoreInterviewOffer_df = evaluate.computeResult(result, gold_app_df, onlyInterviewOffer=True)

    log.info('========= Overall Candidate Score =========')
    summaryAll_df = evaluate.printResult(scoreAll_df)
    summaryInterviewOffer_df = evaluate.printResult(scoreInterviewOffer_df)

    log.info('Client ID: %s' % client_id)

    # opp_df = pd.read_csv('data/opp_200.csv', delimiter=',', low_memory=False, header=0, encoding='utf_8')
    # trainingData = model.prepareTrainingDoc('/u01/bigdata/vX/CV/CV/200', opp_df)
    # model.trainDoc2VecModel(trainingData, 30)


if __name__ == "__main__":
    main()