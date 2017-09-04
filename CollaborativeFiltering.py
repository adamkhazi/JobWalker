import collections
from datetime import datetime
import logging as log

class CollaborativeFiltering():
    def __init__(self, train_app_df, test_app_df, opp_df):
        '''
        Init the object
        :param train_app_df: Training dataframe
        :param test_app_df: Test dataframe
        :param opp_df: Opportunity dataframe
        '''

        log.info('## Init Collaborative-Filtering ##')
        self.train_app_df = train_app_df
        self.test_app_df = test_app_df
        self.opp_df = opp_df
        self.total_opp = opp_df.shape[0]

    def getRecommendationResult(self, topN=10):
        '''
        Get personalised recommendation all candidates in test_app_df

        :param topN: Number of recommendation per candidate
        :return:
        '''
        log.info('## Computing Collaborative-Filtering Recommendation ##')

        # Find the list of open opportunity
        open_opp_id_list = self.opp_df[self.opp_df['is_open'] == 1].opp_id.values

        # List of candidates that recommendation is required
        candidate_id_list = self.test_app_df['candidate_id'].unique()

        # Find the list of most popular open opportunity
        popular_opp_list = self.__mostPopularOpp(open_opp_id_list)

        # Create the interaction lookup dict
        # interactionDict = self.__createCandidateInteractionLookupDict()
        interactionDict = self.__createCandidate2OpportunityLookupDict(self.train_app_df)

        recommendationDict = collections.defaultdict(list)
        for target_candidate_id in candidate_id_list:
            result = self.__computeSimilarity(target_candidate_id, interactionDict)

            if len(result) > 0:
                top_similar_list = sorted(result, key=result.get, reverse=True)

                for i in top_similar_list:
                    # Did top similar candidate applied for any open opportunity?
                    for opp in interactionDict[i]:
                        if opp in open_opp_id_list:
                            if opp not in recommendationDict[target_candidate_id]:
                                if len(recommendationDict[target_candidate_id]) < topN:
                                    recommendationDict[target_candidate_id].append(opp)

            # If recommendation less than topN, fill it up with popular opportunities
            for opp in popular_opp_list:
                if len(recommendationDict[target_candidate_id]) < topN:
                    if opp not in recommendationDict[target_candidate_id]:
                        recommendationDict[target_candidate_id].append(opp)
                else:
                    break

        log.debug('Recommendation Result: %s' % recommendationDict)
        return recommendationDict

    def __createCandidate2OpportunityLookupDict(self, train_app_df):
        bygroup_candidate_dict = {k: set(v) for k, v in train_app_df.groupby('candidate_id')['opp_id']}

        return bygroup_candidate_dict

    def __computeSimilarity(self, target_can_id, interactionDict):
        # Rank neighbour candidates by jaccard similarity
        result = {}
        if target_can_id in interactionDict:
            target_interaction = interactionDict[target_can_id]

            for k, v in interactionDict.items():
                if k != target_can_id:
                    if len(v) >= 2:
                        score = self.__jaccard_similarity(target_interaction, v, self.total_opp)
                        if score != 0:
                            result[k] = score
                        # else:
                        #     result[k] = score
            return result
        else:
            # print('Target Candidate ID (%d) not found.' %target_can_id)
            return result

    def __jaccard_similarity(self, x, y, union_cardinality):
        """ returns the jaccard similarity between two lists """

        intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
        return intersection_cardinality / float(union_cardinality)

    def __mostPopularOpp(self, open_opp_id_list):
        '''
        Rank open opportunity by popularity
        :param open_opp_id_list: List of open opportunity
        :return: ranked list of opportunity by popularity
        '''
        popularDic = {}
        for opp_id in open_opp_id_list:
            popularDic[opp_id] = len(self.train_app_df[self.train_app_df['opp_id']==opp_id])
        # print('__mostPopularOpp: %s' %popularDic)
        result = sorted(popularDic, key=popularDic.get, reverse=True)
        log.info(result)
        return result