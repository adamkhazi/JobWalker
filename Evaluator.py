import pandas as pd
import math
import logging as log
import collections

class RecommenderEvaluator:
    def __init__(self):
        pass

    def computeResult(self, pred, gold_df, onlyInterviewOffer=False):
        '''
        Compute the result
        :param pred: Dict {candidate_id: [list of recommendation]}
        :param gold_df: Gold dataframe
        :param onlyInterviewOffer: True to score only recommendation that reach at least interview
        :return: Combined scoring result including Recall, MAP and NDCG
        '''
        if onlyInterviewOffer:
            gold_df = gold_df[gold_df.app_state >= 2]
            candidateID_list = list(gold_df.candidate_id.unique())
            pred = {key:value for key, value in pred.items() if key in candidateID_list}


        score_df = pd.DataFrame()
        score_df['candidate_id'] = pred.keys()
        score_df['recommendation'] = pred.values()

        recall = self.__recall(pred, gold_df, 5)
        score_df['recall_05'] = recall

        recall = self.__recall(pred, gold_df, 10)
        score_df['recall_10'] = recall

        recall = self.__recall(pred, gold_df, 20)
        score_df['recall_20'] = recall

        recall = self.__recall(pred, gold_df, 30)
        score_df['recall_30'] = recall

        precision = self.__precision(pred, gold_df, 5)
        score_df['precision_05'] = precision

        precision = self.__precision(pred, gold_df, 10)
        score_df['precision_10'] = precision

        precision = self.__precision(pred, gold_df, 20)
        score_df['precision_20'] = precision

        precision = self.__precision(pred, gold_df, 30)
        score_df['precision_30'] = precision

        ndcg = self.__ndcg(pred, gold_df, 5)
        score_df['ndcg_05'] = ndcg

        ndcg = self.__ndcg(pred, gold_df, 10)
        score_df['ndcg_10'] = ndcg

        ndcg = self.__ndcg(pred, gold_df, 20)
        score_df['ndcg_20'] = ndcg

        ndcg = self.__ndcg(pred, gold_df, 30)
        score_df['ndcg_30'] = ndcg

        # self.printResult(score_df)
        return score_df

    def __recall(self, pred, gold_df, topN):
        '''
        Compute Recall result
        :param pred: Dict {candidate_id: [list of recommendation]}
        :param gold_df: Gold dataframe
        :param topN: Top N recommendation used for scoring
        :return: Scoring result
        '''
        result = []
        for k, v in pred.items():
            gold_opp = gold_df[gold_df['candidate_id']==k].opp_id.values

            count = 0
            number_found = 0
            for pred_opp in v:
                if count == topN:
                    break
                count += 1
                if pred_opp in gold_opp:
                    number_found += 1
            recall = number_found/len(gold_opp)
            # print('Candidate %s recall: %f' %(k, recall))
            result.append(recall)

        return result

    def __precision(self, pred, gold_df, topN):
        '''
        Compute MAP result
        :param pred: Dict {candidate_id: [list of recommendation]}
        :param gold_df: Gold dataframe
        :param topN: Top N recommendation used for scoring
        :return: Scoring result
        '''
        result = []
        for k, v in pred.items():
            gold_opp = gold_df[gold_df['candidate_id']==k].opp_id.values

            count = 0
            number_found = 0
            score = 0.0
            for pred_opp in v:
                if count == topN:
                    break
                count += 1
                if pred_opp in gold_opp:
                    number_found += 1
                    score += (number_found/float(count))

            # print('topN: %d' %topN)
            # print('Gold: %s' %gold_opp)
            # print('Pred: %s' %v)
            # print(score)
            if number_found == 0:
                result.append(0)
            else:
                result.append(score/number_found)

        return result

    # def __dcg(self, pred, gold_df, topN):
    #     result = []
    #     for k, v in pred.items():
    #         gold_opp = gold_df[gold_df['candidate_id']==k].opp_id.values
    #
    #         score = 0
    #         count = 0
    #         for pred_opp in v:
    #             if count == topN:
    #                 break
    #             count += 1
    #
    #             if pred_opp in gold_opp:
    #                 relevance_score = (gold_df[(gold_df['candidate_id']==k) & (gold_df['opp_id']==pred_opp)].app_state)
    #                 gain = math.pow(2, relevance_score) - 1
    #                 discount = math.log2(1 + count)
    #                 score += gain/discount
    #         # print(gold_opp)
    #         # print(v)
    #         # print(score)
    #         result.append(score)
    #     return result

    def __ndcg(self, pred, gold_df, topN):
        '''
        Compute NDCG result
        :param pred: Dict {candidate_id: [list of recommendation]}
        :param gold_df: Gold dataframe
        :param topN: Top N recommendation used for scoring
        :return: Scoring result
        '''
        result = []
        for k, v in pred.items():
            gold_opp = gold_df[gold_df['candidate_id']==k].opp_id.values

            # Calculate 'perfect ranking score' for normalisation
            can_gold_df = gold_df[gold_df['candidate_id'] == k]
            # print(can_gold_df['app_state'])
            can_gold_df = can_gold_df.sort_values('app_state', ascending=False)
            # if k == 367838:
            #     print(can_gold_df[['app_state', 'opp_id']])

            best_score = 0
            best_count = 1
            for index, row in can_gold_df.iterrows():
                # if row.app_state == 3:
                #     relevance_score = 5
                # elif row.app_state == 2:
                #     relevance_score = 3
                # elif row.app_state == 1:
                #     relevance_score = 1
                relevance_score = row.app_state
                gain = math.pow(2, relevance_score) - 1
                discount = math.log2(1 + best_count)
                best_score += (gain / discount)
                best_count += 1
                if best_count == topN:
                    break
                # if k == 367838:
                #     print('best score: %f' %best_score)


            score = 0
            count = 0
            for pred_opp in v:
                if count == topN:
                    break
                count += 1

                if pred_opp in gold_opp:
                    relevance_score = (gold_df[(gold_df['candidate_id']==k) & (gold_df['opp_id']==pred_opp)].app_state)
                    # print(relevance_score)
                    # print(type(relevance_score))
                    # print('candidate: %f  Opp: %f' %(k, pred_opp))
                    gain = math.pow(2, relevance_score) - 1
                    discount = math.log2(1 + count)
                    score += gain/discount
            # print(gold_opp)
            # print(v)
            # print(score)
            # print(best_score)
            result.append(score/best_score)

            # if k == 367838:
            #     print('####################################')
            #     print('367838 best score: %f' %best_score)
            #     print('367838 score: %f' % score)
        return result

    def printResult(self, score_df):
        '''
        Process the statistical result and print it
        :param score_df:
        :return: None
        '''

        columns = list(score_df.columns.values)
        columns.remove('candidate_id')
        columns.remove('recommendation')

        stats = []
        for col in columns:
            # print('%s Mean: %f' %(col, score_df[col].mean()))
            stats.append(score_df[col].describe())
            # print(score_df[col].describe())

        df = pd.DataFrame(stats)
        df = df.applymap(lambda x: '%.4f' % x)
        log.info('\n%s' %df)

        return df

    def printRecommendationResult(self, pred, gold_df, opp_df, num_candidate=50, topN=10):
        '''
        Print the topN candidate recommendation
        :param pred: Dict {candidate_id: [list of recommendation]}
        :param gold_df: Gold label. Used to highlight which recommendation was correctly recommended
        :return: None
        '''

        outputString = '\n'
        for key, value in pred.items():
            if num_candidate == 0:
                break
            num_candidate -= 1

            outputString = outputString + '#############################\n'
            outputString = outputString + 'Candidate ID: ' + str(key) + '\n'
            outputString = outputString + '#############################\n'
            outputString = outputString + 'Opp_id\tRel\tTitle\n'
            # log.info('#################################')
            # log.info('Candidate ID: %s ' %key)
            # log.info('#################################')
            count = 0
            for rec in value:
                if count == topN:
                    continue
                count += 1

                title = opp_df[opp_df['opp_id']==rec].title.values[0]
                rel = gold_df[(gold_df['candidate_id']==key) & (gold_df['opp_id']==rec)].app_state
                if len(rel)>0:
                    rel = rel.values[0]
                else:
                    rel = '?'

                # log.info('Opp_ID: %s\t%s' %(rec, title))
                outputString = outputString + str(rec) + '\t' + str(rel) + '\t' + str(title) + '\n'


        log.info(outputString)

    def countApplyInterviewOffer(self, result, gold_df, topN):
        result_dict = collections.defaultdict(int)
        for can_id, opp_list in result.items():
            for opp in opp_list[:topN]:
                state = gold_df[(gold_df['candidate_id'] == can_id) & (gold_df['opp_id'] == opp)].app_state.values
                if len(state) > 0:
                    result_dict[state[0]] += 1
        return result_dict