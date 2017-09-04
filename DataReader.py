import pandas as pd
import numpy as np
import logging as log

class CSVReader:
    def __init__(self, app_filename, opp_filename):
        self.app_filename = app_filename
        self.opp_filename = opp_filename

    def getTrainNTestData_RandomCandidate(self, number_of_test_opp=500, number_of_test_candidate=500, seed=0):
        '''
        Prepare the CSV data into training and test set
        :param: number_of_test_opp: Number of open opportunity in test set
        :param: number_of_test_candidate: Number of candidate in test set
        :param seed: Use a diff Seed number for N-fold testing
        :return: 3 pandas dataframe. [train_app, test_app, opp DF]
        '''
        app_df = pd.read_csv(self.app_filename, delimiter=',', low_memory=True, header=0, encoding='utf_8')
        opp_df = pd.read_csv(self.opp_filename, delimiter=',', low_memory=False, header=0, encoding='utf_8')
        # app_df = self.__removeNoCVRows(app_df)
        # app_df = self.__removeAnonymisedRows(app_df)

        # basic_df = pd.merge(app_df,opp_df,how='left', on='opp_id')

        # Get all unique opportunity id as a list
        opp_id_unique = opp_df['opp_id'].unique()
        # can_id_unique = app_df['candidate_id'].unique()

        np.random.seed(seed)
        # Get random set of opportunity for testing
        test_opp_id_list = np.random.choice(opp_id_unique, number_of_test_opp, replace=False)
        log.info('Number of unique Opportunity: %d' %len(opp_id_unique))
        log.info('Number of Opportunity for test: %d' %len(test_opp_id_list))

        # Find who are the candidates that have applied to the test opp id list
        unique_candiate_list = app_df[app_df['opp_id'].isin(test_opp_id_list)].candidate_id.unique()
        log.info('Unique candidate list that applied to test opp: %d' %len(unique_candiate_list))
        assert (len(unique_candiate_list) > number_of_test_candidate)

        # Choose a random set of candidate as test set
        test_candidate_index_list = np.random.choice(unique_candiate_list, number_of_test_candidate, replace=False)

        gold_app_df = app_df[app_df['candidate_id'].isin(test_candidate_index_list)
                             & app_df['opp_id'].isin(test_opp_id_list)]
        train_app_df = app_df.loc[set(app_df.index.values) - set(gold_app_df.index.values)]

        self.__basic_statistic(train_app_df, gold_app_df)

        # Add a column to opportunity to simulate open and closed opportunites
        # Recommder will only recommend open opportunity
        opp_df.set_value(opp_df[opp_df['opp_id'].isin(test_opp_id_list)].index.values,'is_open', 1)


        return [train_app_df, gold_app_df, opp_df]

    def getTrainNTestData_ColdOpportunity(self, number_of_test_opp=500, number_of_test_candidate=50, seed=0):
        '''
        Prepare the CSV data into training and test set

        Simulate cold opp in the test set.
        number_of_candidate will be applying to cold opp in test set

        :param number_of_cold_opp:
        :param total_opp_in_test:
        :param seed:
        :return: 3 pandas dataframe. [train_app, test_app, opp DF]
        '''
        app_df = pd.read_csv(self.app_filename, delimiter=',', low_memory=True, header=0, encoding='utf_8')
        opp_df = pd.read_csv(self.opp_filename, delimiter=',', low_memory=False, header=0, encoding='utf_8')
        app_df = self.__removeNoCVRows(app_df)
        app_df = self.__removeAnonymisedRows(app_df)

        # Get a list of candidate with 2 or more application
        # Get a list of unique opp that the candidates above applied to
        # From the unique opp list randomly choose number_of_cold_opp


        # Get a list of candidate with 2 or more application
        # Choose 50 candidate
        # Take one application from each candidate == gold
        # Remove all other application with opp_id in gold set

        # Group candidate by their application count
        bygroup_candidate = app_df.groupby('candidate_id').opp_id.count()
        # print(bygroup_candidate[bygroup_candidate>=min_application])

        potential_candidate_list = bygroup_candidate[bygroup_candidate >= 2].index
        # print('######################### %s' %potential_candidate_list)

        if len(potential_candidate_list) < number_of_test_candidate:
            log.error('Data set is impossible to construct. number_of_test_candidate needs to be reduced.')
            assert 0

        np.random.seed(seed)
        # Get random set of candidate for testing
        test_can_id_list = np.random.choice(potential_candidate_list, number_of_test_candidate, replace=False)
        # print(len(test_can_id_list))

        # For each candidate, choose min_application - application_train application to be in test set
        test_application_list = []
        for candidate in test_can_id_list:
            # Find all the applications made by the candidate id
            app_list = app_df[app_df['candidate_id'] == candidate].app_id.unique()
            # Select one of the application as test set
            test_application_list.append(np.random.choice(app_list, 1, replace=False))

        test_application_list = set(np.concatenate(test_application_list).ravel())
        # log.debug('Application ID: %s' % test_application_list)

        # Form the test and train app dataframe
        gold_app_df = app_df[app_df['app_id'].isin(test_application_list)]
        # train_app_df = app_df.loc[set(app_df.index.values) - set(gold_app_df.index.values)]
        # Find all the opp_id that candidates in gold that candidate id list applied
        gold_opp_id_list = list(gold_app_df['opp_id'].unique())
        log.info('Unique opp_id in test set: %d' % len(gold_opp_id_list))

        # Make test opp_id cold by dropping all application to those opp in train
        drop_index = app_df[app_df['opp_id'].isin(gold_opp_id_list)].index
        train_app_df = app_df.drop(drop_index)


        # Get all unique id as a list
        opp_id_unique = list(opp_df['opp_id'].unique())
        log.debug('sample opp_id: %s' % opp_id_unique[:5])

        while len(gold_opp_id_list) < number_of_test_opp:
            new_opp = np.random.choice(opp_id_unique, 1, replace=False)
            if new_opp not in gold_opp_id_list:
                gold_opp_id_list.append(new_opp)

        log.info('Unique opp_id after adding more: %d' % len(gold_opp_id_list))
        log.debug(gold_opp_id_list[:5])
        opp_df.set_value(opp_df[opp_df['opp_id'].isin(gold_opp_id_list)].index.values, 'is_open', 1)

        # log.debug(opp_df[opp_df['opp_id'] == 5575][['opp_id', 'is_open']])
        # log.debug(opp_df[opp_df.opp_id == 6331][['opp_id', 'is_open']])
        # log.debug(opp_df[opp_df.opp_id == 6316][['opp_id', 'is_open']])
        # log.debug(opp_df[opp_df.opp_id == 6430][['opp_id', 'is_open']])
        # log.debug(opp_df.head(5))

        # log.debug(opp_df['is_open'].sum())
        # assert 0
        self.__basic_statistic(train_app_df, gold_app_df)

        return [train_app_df, gold_app_df, opp_df]

    def getTrainNTestData_ColdCandidate(self, number_of_test_opp=500, number_of_test_candidate=50, seed=0):
        '''
        Prepare the CSV data into training and test set
        Focus on getting only cold candidates in test set

        :param number_of_test_opp:
        :param number_of_test_candidate:
        :param seed:
        :return: 3 pandas dataframe. [train_app, test_app, opp DF]
        '''
        app_df = pd.read_csv(self.app_filename, delimiter=',', low_memory=True, header=0, encoding='utf_8')
        opp_df = pd.read_csv(self.opp_filename, delimiter=',', low_memory=False, header=0, encoding='utf_8')
        app_df = self.__removeNoCVRows(app_df)
        app_df = self.__removeAnonymisedRows(app_df)

        # Get all unique id as a list
        opp_id_unique = opp_df['opp_id'].unique()
        can_id_unique = app_df['candidate_id'].unique()

        np.random.seed(seed)
        # Get random set of application for testing
        test_can_id_list = np.random.choice(can_id_unique, number_of_test_candidate, replace=False)
        log.info('Number of unique candidate_id: %d' % len(can_id_unique))
        log.info('Number of candidate_id for test: %d' % len(test_can_id_list))

        # Find all the application detail by candidates in test_can_id_list to create test_app
        gold_app_df = app_df[app_df['candidate_id'].isin(test_can_id_list)]
        train_app_df = app_df.loc[set(app_df.index.values) - set(gold_app_df.index.values)]

        # Find all the opp_id that candidates in gold can id list applied
        test_opp_list = list(gold_app_df.opp_id.unique())
        log.info('Unique opp_id in test set: %d' %len(test_opp_list))

        while len(test_opp_list) < number_of_test_opp:
            new_opp = np.random.choice(opp_id_unique, 1, replace=False)
            if new_opp not in test_opp_list:
                test_opp_list.append(new_opp)

        log.info('Unique opp_id after adding more: %d' % len(test_opp_list))

        opp_df.set_value(opp_df[opp_df['opp_id'].isin(test_opp_list)].index.values, 'is_open', 1)

        self.__basic_statistic(train_app_df, gold_app_df)

        return [train_app_df, gold_app_df, opp_df]


    def getTrainNTestData_WarmCandidate(self, number_of_test_opp=500, number_of_test_candidate=10, min_application=7,
                                        application_train=4, seed=0):
        '''
        Prepare the CSV data into training and test set
        Focus on getting warm candidates with at least x applications in test set

        :param: number_of_test_opp: Number of open opportunity in test set
        :param: number_of_test_candidate: Number of candidate in test set
        :param: min_application: Test candidate needs to have at least x application
        :param: application_train: Test candidate needs to have at x application in train
        :param seed: Use a diff Seed number for N-fold testing
        :return: 3 pandas dataframe. [train_app, test_app, opp DF]
        '''
        if number_of_test_opp < number_of_test_candidate * (min_application - application_train):
            log.error('Data set is impossible to construct. number_of_test_opp needs to be greater.')
            assert(0)

        app_df = pd.read_csv(self.app_filename, delimiter=',', low_memory=True, header=0, encoding='utf_8')
        opp_df = pd.read_csv(self.opp_filename, delimiter=',', low_memory=False, header=0, encoding='utf_8')
        app_df = self.__removeNoCVRows(app_df)
        app_df = self.__removeAnonymisedRows(app_df)

        # Find all candidates with application greater than min_application_train + min_application_test
        bygroup_candidate = app_df.groupby('candidate_id').opp_id.count()
        # print(bygroup_candidate[bygroup_candidate>=min_application])

        potential_candidate_list = bygroup_candidate[bygroup_candidate>=min_application].index
        # print(potential_candidate_list)

        if len(potential_candidate_list) < number_of_test_candidate:
            log.error('Data set is impossible to construct. number_of_test_candidate needs to be reduced.')
            assert(0)

        np.random.seed(seed)
        # Get random set of candidate for testing
        test_can_id_list = np.random.choice(potential_candidate_list, number_of_test_candidate, replace=False)
        # print(len(test_can_id_list))

        # For each candidate, choose min_application - application_train application to be in test set
        test_application_list = []
        for candidate in test_can_id_list:
            app_list = app_df[app_df['candidate_id']==candidate].app_id.unique()
            test_application_list.append(np.random.choice(app_list, min_application - application_train, replace=False))

        test_application_list = set(np.concatenate(test_application_list).ravel())

        # Form the test and train app dataframe
        gold_app_df = app_df[app_df['app_id'].isin(test_application_list)]
        train_app_df = app_df.loc[set(app_df.index.values) - set(gold_app_df.index.values)]

        # Add a column to opportunity to simulate open and closed opportunites
        # Recommender will only recommend open opportunity
        test_opp_id_list = gold_app_df['opp_id'].unique().tolist()

        # Add more opp if less than required
        if len(test_opp_id_list) < number_of_test_opp:
            # Get all unique opportunity id as a list
            opp_id_unique = opp_df['opp_id'].unique()

            opp_id_unique = set(opp_id_unique) - set(test_opp_id_list)
            # print(opp_id_unique)
            test_opp_id_list.extend(np.random.choice(list(opp_id_unique), number_of_test_opp - len(test_opp_id_list)
                                                     , replace=False))

        opp_df.set_value(opp_df[opp_df['opp_id'].isin(test_opp_id_list)].index.values,'is_open', 1)

        self.__basic_statistic(train_app_df, gold_app_df)

        return [train_app_df, gold_app_df, opp_df]

    def getBasicDataFrame(self):
        '''
        Get the entire dataset. Mainly used for data exploration
        :return: A pandas DF
        '''
        app_df = pd.read_csv(self.app_filename, delimiter=',', low_memory=False, header=0, encoding='utf_8')
        opp_df = pd.read_csv(self.opp_filename, delimiter=',', low_memory=False, header=0, encoding='utf_8')

        app_df = self.__removeNoCVRows(app_df)
        result_df = pd.merge(app_df,opp_df,how='left', on='opp_id')

        return result_df

    def __removeNoCVRows(self, df):
        # print(df.shape[0])
        df = df[df['cv name'].notnull()]
        # print(df.shape[0])
        return df

    def __removeAnonymisedRows(self, df):
        # print(df.shape[0])
        df = df[df['is_anonymized']!=1]
        # print(df.shape[0])
        return df

    def __basic_statistic(self, train_df, gold_df):


        # Cold user == only in test and not in train
        can_id_train_unique = train_df['candidate_id'].unique()
        can_id_gold_unique = gold_df['candidate_id'].unique()
        log.info('Number of unique Candidate in Train: %d' % len(can_id_train_unique))
        log.info('Number of unique Candidate in Test: %d' % len(can_id_gold_unique))
        log.info('Number of cold user in Test: %d' % len(set(can_id_gold_unique) - set(can_id_train_unique)))
        warm_candidate = set(can_id_gold_unique) - (set(can_id_gold_unique) - set(can_id_train_unique))
        log.info('Number of warm user in Test: %d ' %len(warm_candidate))

        # Cold opportunity == only in test and not in train
        opp_id_train_unique = train_df['opp_id'].unique()
        opp_id_gold_unique = gold_df['opp_id'].unique()
        log.info('Number of unique Opp in Test: %d' % len(opp_id_gold_unique))
        log.info('Number of cold opportunity in Test: %d' % len(set(opp_id_gold_unique) - set(opp_id_train_unique)))

        # DES.displayCandidateAppHistogram(gold_df)

def main():
    datareader = CSVReader('data/app_200_cv.csv', 'data/opp_200.csv')
    # train_app_df, gold_app_df, opp_df= datareader.getTrainNTestData2(5, 5, seed=3)
    train_app_df, gold_app_df, opp_df = datareader.getTrainNTestData_WarmCandidate(number_of_test_opp=500, number_of_test_candidate=10, min_application=7,
                                                                                   application_train=4, seed=0)
    # print(train_df.info())
    # print(test_df.info())

    log.info('Number of open opportunity: %d' %len(opp_df[opp_df['is_open']==1]))
    log.info('Size of test set: %d' %gold_app_df.shape[0])


if __name__ == "__main__":
    main()