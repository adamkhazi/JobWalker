import DataReader as dr
import CollaborativeFiltering
import ContentBasedRecommender
import GraphBasedRecommender
import Evaluator
import logging as log
from Enumeration import Test_Mode
import pandas as pd

def runCollaborativeFiltering(app_filename, opp_filename, test_mode, seed):
    log.info('Starting Collaborative Filtering Recommender')

    train_app_df, gold_app_df, opp_df = __getTestData(app_filename, opp_filename, test_mode, seed)

    # Drop the 'answer' columns
    test_app_df = gold_app_df.drop(['opp_id', 'app_state', 'app_id', 'app_source_json'], axis=1)

    recommender = CollaborativeFiltering.CollaborativeFiltering(train_app_df, test_app_df, opp_df)
    result = recommender.getRecommendationResult(topN=30)

    log.info('Computing score on the recommendation.')
    evaluate = Evaluator.RecommenderEvaluator()
    scoreAll_df = evaluate.computeResult(result, gold_app_df, onlyInterviewOffer=False)
    scoreInterviewOffer_df = evaluate.computeResult(result, gold_app_df, onlyInterviewOffer=True)

    log.info('========= Overall Candidate Score =========')
    summaryAll_df = evaluate.printResult(scoreAll_df)
    summaryInterviewOffer_df = evaluate.printResult(scoreInterviewOffer_df)

    return [summaryAll_df, summaryInterviewOffer_df, result]

def runContentBasedRecommender(app_filename, opp_filename, doc2vec_filename, test_mode, seed):
    log.info('Starting Content-based Recommender')
    train_app_df, gold_app_df, opp_df = __getTestData(app_filename, opp_filename, test_mode, seed)

    # Drop the 'answer' columns
    test_app_df = gold_app_df.drop(['opp_id', 'app_state', 'app_id', 'app_source_json'], axis=1)

    recommender = ContentBasedRecommender.ContentBasedDoc2Vec(train_app_df, test_app_df, opp_df,
                                                              train_model_name = doc2vec_filename)
    result = recommender.getRecommendationResult(topN=30)

    log.info('Computing score on the recommendation.')
    evaluate = Evaluator.RecommenderEvaluator()
    scoreAll_df = evaluate.computeResult(result, gold_app_df, onlyInterviewOffer=False)
    scoreInterviewOffer_df = evaluate.computeResult(result, gold_app_df, onlyInterviewOffer=True)

    log.info('========= Overall Candidate Score =========')
    summaryAll_df = evaluate.printResult(scoreAll_df)
    summaryInterviewOffer_df = evaluate.printResult(scoreInterviewOffer_df)

    return [summaryAll_df, summaryInterviewOffer_df, result]

def runGraphBasedRecommender(app_filename, opp_filename, cv_filename, test_mode, seed):
    log.info('Starting Reciprocal Recommender')

    train_app_df, gold_app_df, opp_df = __getTestData(app_filename, opp_filename, test_mode, seed)

    # Drop the 'answer' columns
    test_app_df = gold_app_df.drop(['opp_id', 'app_state', 'app_id', 'app_source_json'], axis=1)
    recommender = GraphBasedRecommender.GraphBasedBiDirectionalRecommender(train_app_df, test_app_df, opp_df, cv_filename)
    # result = recommender.getRecommendationResult_schulze(topN=30)
    # result = recommender.getRecommendationResult_lexicographic(topN=30)
    result = recommender.getRecommendationResult_harmonic(topN=30)

    log.info('Computing score on the recommendation.')
    evaluate = Evaluator.RecommenderEvaluator()
    scoreAll_df = evaluate.computeResult(result, gold_app_df, onlyInterviewOffer=False)
    scoreInterviewOffer_df = evaluate.computeResult(result, gold_app_df, onlyInterviewOffer=True)

    log.info('========= Overall Candidate Score =========')
    summaryAll_df = evaluate.printResult(scoreAll_df)
    summaryInterviewOffer_df = evaluate.printResult(scoreInterviewOffer_df)

    log.debug(app_filename)

    return [summaryAll_df, summaryInterviewOffer_df, result]

def __getTestData(app_filename, opp_filename, test_mode, seed):
    datareader = dr.CSVReader(app_filename, opp_filename)

    if test_mode == Test_Mode.Random_Candidate:
        train_app_df, gold_app_df, opp_df = datareader.getTrainNTestData_RandomCandidate(number_of_test_opp=500,
                                                                                         number_of_test_candidate=50,
                                                                                         seed=seed)
    elif test_mode == Test_Mode.Warm_Candidate:
        train_app_df, gold_app_df, opp_df = datareader.getTrainNTestData_WarmCandidate(number_of_test_opp=500,
                                                                                       number_of_test_candidate=50,
                                                                                       min_application=2,
                                                                                       application_train=1,
                                                                                       seed=seed)
    elif test_mode == Test_Mode.Cold_Candidate:
        train_app_df, gold_app_df, opp_df = datareader.getTrainNTestData_ColdCandidate(number_of_test_opp=500,
                                                                                       number_of_test_candidate=50,
                                                                                       seed=seed)
    elif test_mode == Test_Mode.Cold_Opportunity:
        train_app_df, gold_app_df, opp_df = datareader.getTrainNTestData_ColdOpportunity(number_of_test_opp=500,
                                                                                         number_of_test_candidate=50,
                                                                                         seed=seed)
    else:
        log.error('Invalid mode selected.')
        assert(False)

    return train_app_df, gold_app_df, opp_df

def main():
    # Run test multiple and average the result
    client_id = '422'

    all_result = []
    # Loop the test
    for seed in range(0, 5):
        # Use a different seed to choose different candidates and opporunity for each run

        # Comment or uncomment to test collaborative filtering
        # all_result.append(runCollaborativeFiltering('data/app_'+client_id+'.csv',
        #                                             'data/opp_'+client_id+'.csv',
        #                                             Test_Mode.Warm_Candidate, seed))

        # Comment or uncomment to test content-based
        # all_result.append(runContentBasedRecommender('data/app_'+client_id+'_doc2vec.csv',
        #                                              'data/opp_'+client_id+'_doc2vec.csv',
        #                                              'model/doc2vec_'+client_id+'.d2v',
        #                                              Test_Mode.Random_Candidate,
        #                                              seed))

        # Comment or uncomment to test reciprocal recommender
        all_result.append(runGraphBasedRecommender('data/app_'+client_id+'_doc2vec.csv',
                                                   'data/opp_'+client_id+'_doc2vec.csv',
                                                   'data/cv_'+client_id+'.csv',
                                                   Test_Mode.Random_Candidate,
                                                   seed))

    # Average the result from multiple run and display it
    if len(all_result) > 1:
        all = [row[0] for row in all_result]
        all_df = pd.concat(all)
        all_df = all_df.apply(pd.to_numeric, errors='ignore')
        all_df = all_df.groupby(all_df.index).mean()
        all_df = all_df.applymap(lambda x: '%.3f' % x)
        log.info('Average Result (All): \n%s' % all_df)

        offerInterview = [row[1] for row in all_result]
        offerInterview_df = pd.concat(offerInterview)
        offerInterview_df = offerInterview_df.apply(pd.to_numeric, errors='ignore')
        offerInterview_df = offerInterview_df.groupby(offerInterview_df.index).mean()
        offerInterview_df = offerInterview_df.applymap(lambda x: '%.3f' % x)
        log.info('Average Result (Interview/Offer): \n%s' % offerInterview_df)

if __name__ == "__main__":
    logger = log.getLogger()
    logger.setLevel(log.INFO)

    # create formatter and add it to the handlers
    formatter = log.Formatter('%(asctime)s; %(module)s; %(funcName)s; %(levelname)s; %(message)s')

    fh = log.FileHandler('01logging.log')
    fh.setFormatter(formatter)
    # add the handlers to logger
    # logger.addHandler(fh)

    main()