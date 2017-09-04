import collections
import numpy as np
import networkx as nx
import logging as log
from joblib import Parallel, delayed
import ast
import math
from dateutil.parser import parse
import Feature_Engineering
import Utilities
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from xgboost import *
from py3votecore.schulze_npr import SchulzeNPR
from py3votecore.condorcet import CondorcetHelper
import operator


class GraphBasedOpportunityRecommender():
    '''
    Recommend opportunities to candidate
    '''
    def __init__(self, train_app_df, test_app_df, opp_df):
        '''
        Init the object
        :param train_app_df: Training dataframe
        :param test_app_df: Test dataframe
        :param opp_df: Opportunity dataframe
        '''
        log.info('## Init Candidate Preference Ranking Function ##')

        self.train_app_df = train_app_df
        self.test_app_df = test_app_df
        self.opp_df = opp_df

        # Drop unused column to reduce memory
        self.train_app_df = self.train_app_df.drop(['app_doc2vec', 'first_name', 'last_name',
                                                    'mid_name', 'email', 'phone', 'app_source_json'], axis=1)
        self.test_app_df = self.test_app_df.drop(['app_doc2vec', 'first_name', 'last_name',
                                                    'mid_name', 'email', 'phone'], axis=1)
        self.opp_df = self.opp_df.drop(['opp_doc2vec', 'opp_source_json'], axis=1)

    def getRecommendationResult(self, topN=10):
        '''
        Get personalised recommendation all candidates in test_app_df

        :param topN: Number of recommendation per candidate
        :return:
        '''

        log.info('## Computing Graph-based Opportunity Recommendation ##')

        # Prepare the graph
        graph = self.__createGraphType6()

        # List of candidates that recommendation is required
        candidate_id_list = self.test_app_df['candidate_id'].unique()

        # Find the list of open opportunity
        open_opp_id_list = self.opp_df[self.opp_df['is_open'] == 1].opp_id.values

        self.train_app_df = ''
        self.test_app_df = ''
        self.opp_df = ''

        can_chunks = list(Utilities.chunks(candidate_id_list, 3))
        log.info('  Computing Personalised Page Rank')
        # Split the computation to multiple threads. Each thread computing a chuck of candidate PPR
        results = Parallel(n_jobs=30)(delayed(self.computeCandidatePPR)(chunk, graph, topN, open_opp_id_list)
                                             for chunk in can_chunks)

        # Variable to store the result
        recommendationDict = {k: v for item in results for k, v in item.items()}

        log.debug('Recommendation Result: %s' %recommendationDict)
        return recommendationDict

    def computeCandidatePPR(self, chunk_candidateID, graph, topN, open_opp_id_list):
        '''
        Helper function for Joblib to compute PPR for a candidate
        :param chunk_candidateID: List of Candidate ID to compute PPR
        :param graph: The base graph
        :param topN: Number of recommendation to make
        :param open_opp_id_list: List of open opportunity
        :return: Dict {candidate_id: [list of opp_id recommended]}
        '''
        resultDict = {}
        for candidateID in chunk_candidateID:
            log.debug('Computing Pagerank for candidate: %d' % candidateID)

            # Create Personalisation dict required for personalised page rank
            personalised_dict = {n: 0 for n in graph.nodes()}

            # Personalise to this candidate
            personalised_dict['c_' + str(candidateID)] = 1
            ppr = nx.pagerank(graph, alpha=0.75, personalization=personalised_dict)

            # Sort the pagerank from highest to lowest
            sorted_keys = sorted(ppr, key=ppr.get, reverse=True)
            result = [int(i[2:]) for i in sorted_keys if i.startswith('o_')]

            # Only recommends opportunity that are still open
            result = [i for i in result if i in open_opp_id_list]

            resultDict[candidateID] = result[:topN]

        return resultDict

    def __createGraphType1(self):
        '''
        Create basic bi-directional graph weighted by state
        No similar user or opportunity edges are added
        No opportunity attribute are added
        :return:
        '''
        log.info('  Creating graph type 1')
        graph = nx.Graph()

        # Step 1: Add all candidates as nodes into graph
        candidate_id_list = self.train_app_df['candidate_id'].unique()
        candidate_id_list = np.concatenate([candidate_id_list, self.test_app_df['candidate_id'].unique()])
        candidate_nodes = [('c_'+str(candidate), {'type':'c'}) for candidate in candidate_id_list]
        graph.add_nodes_from(candidate_nodes);
        # log.debug('Number of unique candidate in train_app_df: %d' %len(self.train_app_df['candidate_id'].unique()))
        # log.debug('Number of unique candidate in test_app_df: %d' %len(self.test_app_df['candidate_id'].unique()))
        # log.debug('Number of candidate in candidate_id_list (Can have duplicate): %d' % len(candidate_id_list))

        # Step 2: Add all opportunities as nodes into graph
        opp_id_list = self.opp_df['opp_id'].unique()
        opp_nodes = [('o_' + str(opp), {'type': 'o'}) for opp in opp_id_list]
        graph.add_nodes_from(opp_nodes)
        log.debug('Number of unique Opportunity in opp_df: %d' %len(self.opp_df['opp_id'].unique()))

        # Step 3: Create all the edges between candidates and opportunities
        edges = [('o_'+str(row.opp_id),'c_'+str(row.candidate_id),{'weight':row.app_state}) for index, row in self.train_app_df.iterrows()]
        graph.add_edges_from(edges)
        log.debug(self.train_app_df.shape[0])

        self.__printGraphInformation(graph)

        return graph

    def __createGraphType2(self):
        '''
        Create basic bi-directional graph unweighted
        No similar user or opportunity edges are added
        No opportunity attribute are added
        :return:
        '''
        log.info('  Creating the graph')
        graph = nx.Graph()

        # Step 1: Add all candidates as nodes into graph
        candidate_id_list = self.train_app_df['candidate_id'].unique()
        candidate_id_list = np.concatenate([candidate_id_list, self.test_app_df['candidate_id'].unique()])
        candidate_nodes = [('c_'+str(candidate), {'type':'c'}) for candidate in candidate_id_list]
        graph.add_nodes_from(candidate_nodes);
        log.debug('Number of unique candidate in train_app_df: %d' %len(self.train_app_df['candidate_id'].unique()))
        log.debug('Number of unique candidate in test_app_df: %d' % len(self.test_app_df['candidate_id'].unique()))
        log.debug('Number of candidate in candidate_id_list (Can have duplicate): %d' % len(candidate_id_list))

        # Step 2: Add all opportunities as nodes into graph
        opp_id_list = self.opp_df['opp_id'].unique()
        opp_nodes = [('o_' + str(opp), {'type': 'o'}) for opp in opp_id_list]
        graph.add_nodes_from(opp_nodes)
        log.debug('Number of unique Opportunity in opp_df: %d' %len(self.opp_df['opp_id'].unique()))

        # Step 3: Create all the edges between candidates and opportunities
        edges = [('o_'+str(row.opp_id),'c_'+str(row.candidate_id),{'weight':1}) for index, row in self.train_app_df.iterrows()]
        graph.add_edges_from(edges)
        log.debug(self.train_app_df.shape[0])

        self.__printGraphInformation(graph)

        return graph

    def __createGraphType3(self):
        '''
        Create basic bi-directional graph weighted
        Top 2 similar users based on Doc2vec on their CV are added
        No opportunity attribute are added
        :return:
        '''
        log.info('  Creating graph type 3 using base graph 1')
        # Get base graph
        graph = self.__createGraphType1()
        # self.__printGraphInformation(graph)

        def addSimilarEdge(row):
            similar = row.app_similar_to_doc2vec
            similar = ast.literal_eval(similar)
            edges = [('c_' + str(row['candidate_id']), 'c_' + str(id), {'weight': weight}) for id, weight in similar[:1]]
            # log.debug('number of edges: %d' % len(edges))
            graph.add_edges_from(edges)

        self.test_app_df.apply(addSimilarEdge, axis=1)
        self.__printGraphInformation(graph)

        return graph

    def __createGraphType4(self):
        '''
        Create basic bi-directional graph weighted
        Top 2 similar users based on Doc2vec on their CV are added if they are cold candidate
        No opportunity attribute are added
        :return:
        '''
        log.info('  Creating graph type 4 using base graph 1')
        # Get base graph
        graph = self.__createGraphType1()
        # self.__printGraphInformation(graph)

        can_id_train_unique = self.train_app_df['candidate_id'].unique()

        def addSimilarEdge(row):
            if row['candidate_id'] in can_id_train_unique:
                log.debug('Not adding similar edge. Warm Candidate: %d' % row['candidate_id'])
            else:
                log.debug('Adding edges. Cold Candidate: %d ' % row['candidate_id'])
                similar = row.app_similar_to_doc2vec
                similar = ast.literal_eval(similar)

                edges = [('c_' + str(row['candidate_id']), 'c_' + str(id), {'weight': weight})
                         for id, weight in similar[:10]]
                # log.debug('number of edges: %d' % len(edges))
                graph.add_edges_from(edges)

        self.test_app_df.apply(addSimilarEdge, axis=1)
        self.__printGraphInformation(graph)

        return graph

    def __createGraphType5(self):
        '''
        Create basic bi-directional graph weighted
        Top N similar users based on Doc2vec on their CV are added if they are cold candidate
        Similar Edges has to be within a temporal window.
        No opportunity attribute are added
        :return:
        '''
        log.info('  Creating graph type 5 using base graph 2')
        # Get base graph
        graph = self.__createGraphType2()
        # self.__printGraphInformation(graph)

        can_id_train_unique = self.train_app_df['candidate_id'].unique()

        def addSimilarEdge(row, temporalWindow=15):
            target_candidate_id =row['candidate_id']
            log.debug('Target Candidate ID: %f' % target_candidate_id)

            if target_candidate_id in can_id_train_unique:
                log.debug('Not adding similar edge. Warm Candidate: %d' % target_candidate_id)
            else:
                log.debug('Adding edges within a temporal window. Cold Candidate: %d ' % target_candidate_id)
                target_candidate_date = parse(row['submitted_date'])
                similar_string = row['app_similar_to_doc2vec']
                similar_dict = dict(ast.literal_eval(similar_string))

                temporalGap_dict = collections.defaultdict(lambda: float('inf'))
                for similar_candidate_id, similar_score in similar_dict.items():
                    similar_df = self.train_app_df[self.train_app_df['candidate_id'] == similar_candidate_id]

                    for index_s, row_s in similar_df.iterrows():
                        similar_candidate_date = parse(row_s['submitted_date'])
                        # print(type(similar_candidate_date))
                        diff = math.fabs((target_candidate_date - similar_candidate_date).total_seconds()) / 86400.0
                        # print('Time diff: %.2f' %diff)

                        if diff < temporalGap_dict[similar_candidate_id]:
                            temporalGap_dict[similar_candidate_id] = diff
                # print(temporalGap_dict)
                temporalGap_dict = {k: v for k, v in temporalGap_dict.items() if v < temporalWindow}
                nearest_similar_candidate_list = sorted(temporalGap_dict, key=temporalGap_dict.get)

                log.debug('Number of nearest similar candidate: %d' % len(nearest_similar_candidate_list))

                edges = [('c_' + str(target_candidate_id), 'c_' + str(c_id), {'weight': similar_dict[c_id]})
                         for c_id in nearest_similar_candidate_list[:2]]

                if len(edges) == 0:
                    log.debug('Opening up temporal window to %d. '% int(temporalWindow*1.5))
                    addSimilarEdge(row, temporalWindow=int(temporalWindow*1.5))
                # print(edges)
                graph.add_edges_from(edges)

        self.test_app_df.apply(addSimilarEdge, axis=1)
        self.__printGraphInformation(graph)

        return graph

    def __createGraphType6(self):
        '''
        Create basic bi-directional graph weighted
        Top N similar users based on Doc2vec on their CV are added if they are cold candidate
        Top N similar opp based on Doc2vec on their job description are added if they are cold opportunity
        :return:
        '''
        log.info('  Creating graph type 6 using base graph 1 and 5')
        # Get base graph
        graph = self.__createGraphType5()

        opp_id_train_unique = self.train_app_df['opp_id'].unique()

        def addSimilarEdge(row, temporalWindow=50000):
            if row.is_open == 1:
                target_opp_id = row['opp_id']

                if target_opp_id not in opp_id_train_unique:
                    date_string = str(row['last_app_date'])
                    # print(date_string)
                    if date_string != 'nan':
                        log.debug('Target Opportunity ID: %f' % target_opp_id)
                        log.debug('Adding edges within a temporal window. Cold Opportunity: %d ' % target_opp_id)
                        target_opp_date = parse(date_string)
                        similar_string = row['opp_similar_to_doc2vec']
                        similar_dict = dict(ast.literal_eval(similar_string))

                        temporalGap_dict = collections.defaultdict(lambda: float('inf'))
                        for similar_opp_id, similar_score in similar_dict.items():
                            similar_df = self.opp_df[self.opp_df['opp_id'] == similar_opp_id]

                            for index_s, row_s in similar_df.iterrows():
                                date_str = str(row_s['last_app_date'])
                                if date_str != 'nan':
                                    similar_opp_date = parse(date_str)
                                    # print(type(similar_candidate_date))
                                    diff = math.fabs((target_opp_date - similar_opp_date).total_seconds()) / 86400.0
                                    # print('Time diff: %.2f' %diff)

                                    if diff < temporalGap_dict[similar_opp_id]:
                                        temporalGap_dict[similar_opp_id] = diff
                        # print(temporalGap_dict)
                        temporalGap_dict = {k: v for k, v in temporalGap_dict.items() if v < temporalWindow}
                        nearest_similar_opp_list = sorted(temporalGap_dict, key=temporalGap_dict.get)

                        log.debug('Number of nearest similar opportunity: %d' % len(nearest_similar_opp_list))

                        edges = [('o_' + str(target_opp_id), 'o_' + str(c_id), {'weight': similar_dict[c_id]})
                                 for c_id in nearest_similar_opp_list[:200]]

                        if len(edges) == 0:
                            log.debug('Opening up temporal window to %d. ' % int(temporalWindow * 1.5))
                            addSimilarEdge(row, temporalWindow=int(temporalWindow * 1.5))
                        graph.add_edges_from(edges)

        self.opp_df.apply(addSimilarEdge, axis=1)
        self.__printGraphInformation(graph)

        return graph


    def __printGraphInformation(self, graph):
        log.info('Number of nodes: %d' % graph.number_of_nodes())
        log.info('Number of edges: %d' % graph.number_of_edges())


# class GraphBasedCandidateRecommender():
#     '''
#     Recommend candidates to opportunity (i.e. employer)
#     '''
#     def __init__(self, train_app_df, test_app_df, opp_df):
#         self.train_app_df = train_app_df
#         self.test_app_df = test_app_df
#         self.opp_df = opp_df
#
#         # Drop unused column to reduce memory
#         self.train_app_df = self.train_app_df.drop(['app_doc2vec', 'first_name', 'last_name',
#                                                     'mid_name', 'email', 'phone', 'app_source_json'], axis=1)
#         self.test_app_df = self.test_app_df.drop(['app_doc2vec', 'first_name', 'last_name',
#                                                     'mid_name', 'email', 'phone'], axis=1)
#         self.opp_df = self.opp_df.drop(['opp_doc2vec', 'opp_source_json'], axis=1)
#
#
#
#     def getRecommendationResult(self, topNPercent=0.01):
#         '''
#         Get personalised recommendation all candidates in test_app_df
#
#         :param topNPercent: Number of recommendation per candidate
#         :return:
#         '''
#         log.info('## Computing Graph-based Candidate Recommendation ##')
#
#         # Prepare the graph
#         graph = self.__createGraphType1()
#
#         # Number of topN
#         topN = int(topNPercent * len(self.train_app_df['candidate_id'].unique()))
#         # topN=1000
#         log.debug('TopN: %d' %topN)
#
#         # List of open opp that recommendation is required
#         open_opp_id_list = self.opp_df[self.opp_df['is_open'] == 1].opp_id.values
#
#         opp_chunks = list(Utilities.chunks(open_opp_id_list, 10))
#         log.info('  Computing Personalised Page Rank')
#         # log.debug(len(opp_chunks))
#         # log.debug(len(opp_chunks[0]))
#         # self.computeOppPPR(opp_chunks[0], graph, topN)
#
#         self.train_app_df = ''
#         self.test_app_df = ''
#         self.opp_df = ''
#
#         # num_cores = multiprocessing.cpu_count()
#         results = Parallel(n_jobs=30)(delayed(self.computeOppPPR)(chunk, graph, topN)
#                                              for chunk in opp_chunks)
#
#         # Variable to store the result
#         recommendationDict = {k: v for item in results for k, v in item.items()}
#
#         # log.debug('Recommendation Result: %s' %recommendationDict)
#         return recommendationDict
#
#     def computeOppPPR(self, chunk_opp_id, graph, topN):
#         result_dict = {}
#         for opp_id in chunk_opp_id:
#             log.debug('Computing Pagerank for opportunity: %d' % opp_id)
#
#             # Create Personalisation dict required for personalised page rank
#             personalised_dict = {n: 0 for n in graph.nodes()}
#
#             # Personalise to this candidate
#             personalised_dict['o_' + str(opp_id)] = 1
#             ppr = nx.pagerank(graph, alpha=0.75, personalization=personalised_dict)
#
#             # Sort the pagerank from highest to lowest
#             sorted_keys = sorted(ppr, key=ppr.get, reverse=True)
#             # result = [float(i[2:]) for i in sorted_keys if i.startswith('c_')]
#             result = [float(i[2:]) for i in sorted_keys if i.startswith('c_')]
#
#             result_dict[opp_id] = result[:topN]
#
#         return result_dict
#
#     def __createBaseGraph(self):
#         '''
#         Create a basic graph
#         :return:
#         '''
#         log.info('Creating Base Graph')
#
#         graph = nx.Graph()
#
#         # Step 1: Add all candidates (i.e. from train and test) as nodes into graph
#         candidate_id_list = self.train_app_df['candidate_id'].unique()
#         candidate_id_list = np.concatenate([candidate_id_list, self.test_app_df['candidate_id'].unique()])
#         candidate_nodes = [('c_' + str(candidate), {'type': 'c'}) for candidate in candidate_id_list]
#         graph.add_nodes_from(candidate_nodes);
#
#         # Step 2: Add all opportunities as nodes into graph
#         opp_id_list = self.opp_df['opp_id'].unique()
#         opp_nodes = [('o_' + str(opp), {'type': 'o'}) for opp in opp_id_list]
#         graph.add_nodes_from(opp_nodes)
#         log.debug('Number of unique Opportunity in opp_df: %d' % len(self.opp_df['opp_id'].unique()))
#
#         # Step 3: Create edges between candidates and opportunities if they are interview or offer (State == 2 or 3)
#         edges = [('o_' + str(row.opp_id), 'c_' + str(row.candidate_id), {'weight': row.app_state})
#                  for index, row in self.train_app_df.iterrows() if row['app_state'] > 1]
#         graph.add_edges_from(edges)
#         # log.debug(edges[:10])
#         # log.debug(self.train_app_df.shape[0])
#         log.debug('Basic graph size. ')
#         self.__printGraphInformation(graph)
#
#         return graph
#
#     def __createGraphType1(self):
#         '''
#         Create basic bi-directional graph weighted by state
#         Only edges for interview and offer are used.
#         Edges for apply are not included
#         No similar user or opportunity edges are added
#         No opportunity attribute are added
#         :return:
#         '''
#         log.info('Creating Graph Type 1 using base graph')
#         graph = self.__createBaseGraph()
#
#         # List of candidates that recommendation is required
#         target_candidate_id_list = self.test_app_df['candidate_id'].unique()
#
#         # Find the list of open opportunity
#         open_opp_id_list = self.opp_df[self.opp_df['is_open'] == 1].opp_id.values
#
#         # Step 4: Add edges for target candidate
#         edges = []
#         for target_candidate in target_candidate_id_list:
#             num_edges = len(graph.edges('c_' + str(target_candidate)))
#             # log.debug('Candidate %d has %d edges' %(target_candidate, num_edges))
#
#             if num_edges == 0:
#                 # Add similar to edges
#                 similar = self.test_app_df[self.test_app_df.candidate_id == target_candidate].app_similar_to_doc2vec.values
#                 similar = ast.literal_eval(similar[0])
#                 # log.debug('similar: %s' % similar[:4])
#
#                 edgeLimit = 5
#                 for s_id, s_weight in similar:
#                     if s_id != target_candidate: # Prevent self-loop
#                         if edgeLimit != 0:
#                             edgeLimit -= 1
#                             edges.append(('c_' + str(s_id), 'c_' + str(target_candidate), {'weight': s_weight}))
#                         else:
#                             break
#
#         # Step 5: Add edges for open opportunities
#         for open_opp in open_opp_id_list:
#             num_edges = len(graph.edges('o_' + str(open_opp)))
#             # log.debug('Opp %d has %d edges' % (open_opp, num_edges))
#
#             if num_edges <= 10:
#                 # Add similar to edges
#                 similar = self.opp_df[self.opp_df.opp_id == open_opp].opp_similar_to_doc2vec.values
#                 similar = ast.literal_eval(similar[0])
#                 # log.debug('similar: %s' % similar[:4])
#
#                 edgeLimit = 20
#                 for s_id, s_weight in similar:
#                     if s_id != open_opp: # Prevent self-loop
#                         if edgeLimit != 0:
#                             edgeLimit -= 1
#                             edges.append(('o_' + str(s_id), 'o_' + str(open_opp), {'weight': s_weight}))
#                         else:
#                             break
#
#         graph.add_edges_from(edges)
#         log.debug('After adding similar edges.')
#         self.__printGraphInformation(graph)
#
#         return graph
#
#     def __printGraphInformation(self, graph):
#         log.info('Number of nodes: %d' % graph.number_of_nodes())
#         log.info('Number of edges: %d' % graph.number_of_edges())

# class GraphBasedBiDirectionalRecommender_old():
#     '''
#     Recommend opportunity to candidates based on bidirectional preference
#     '''
#     def __init__(self, train_app_df, test_app_df, opp_df):
#         self.train_app_df = train_app_df
#         self.test_app_df = test_app_df
#         self.opp_df = opp_df
#
#     def getRecommendationResult(self, topN=30):
#         '''
#         Get personalised recommendation all candidates in test_app_df
#
#         :param topN: Number of recommendation per candidate
#         :return:
#         '''
#         candidate_recommender = GraphBasedCandidateRecommender(self.train_app_df, self.test_app_df, self.opp_df)
#         can_result = candidate_recommender.getRecommendationResult(topNPercent=0.05)
#
#         opp_recommender = GraphBasedOpportunityRecommender(self.train_app_df, self.test_app_df, self.opp_df)
#         opp_result = opp_recommender.getRecommendationResult(topN=500)
#
#         final_result = {}
#         for can_id, opp_list in opp_result.items():
#             final_result[can_id] = [opp for opp in opp_list if can_id in can_result[opp]]
#
#         # Trim result to topN
#         final_result = {can_id:opp_list[:topN] for can_id, opp_list in final_result.items()}
#
#         return final_result

class ContentBasedReRanker:
    def __init__(self, train_app_df, test_app_df, opp_df, cv_df, graph_result):
        '''
        Init the object
        :param train_app_df: Training dataframe
        :param test_app_df: Test dataframe
        :param opp_df: Opportunity dataframe
        :param cv_df: CV dataframe
        :param graph_result: PPR result in dict
        '''
        log.info('## Init Employer Preference Ranking Function ##')
        self.train_app_df = train_app_df
        self.test_app_df = test_app_df
        self.opp_df = opp_df
        self.cv_df = cv_df
        self.graph_result = graph_result

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
        self.opp_df['content'] = opp_df['title'].map(str) + " " + \
                            opp_df['role_specifics'].map(str) + " " + \
                            opp_df['skills'].map(str)
        self.opp_df['content'] = Parallel(n_jobs=55)(
            delayed(Utilities.processOpp)(row.content) for _, row in opp_df.iterrows())

        # Open Opp DF
        self.open_opp_df = self.opp_df[self.opp_df.is_open == 1].copy()

        self.featureHelper = Feature_Engineering.FeatureEngineering(self.opp_df)

    def getRecommendationResult(self, topN=10):
        '''
        Get personalised recommendation all candidates in test_app_df

        :param topN: Number of recommendation per candidate
        :return:
        '''
        log.info('## Computing rank based on Employer\'s Preference ##')

        # Create model train data
        x_train, y_train = self.prepareTrainData(isBalance=False)

        # List of candidates that recommendation is required
        candidate_id_list = self.test_app_df['candidate_id'].unique()
        cv_list = self.test_app_df['cv name'].unique()

        self.cv_df = self.cv_df[self.cv_df['cv name'].isin(cv_list)]
        self.train_app_df = ''
        self.opp_df = ''
        log.debug('Length of cv_df: %d' % self.cv_df.shape[0])

        candidate_df_list = self.prepareTestData(self.graph_result, candidate_id_list)
        log.debug('No of unique candidate: %d' % len(candidate_id_list))
        log.debug('No of candidate_df_list: %d' % len(candidate_df_list))

        log.info('Training model now. ')
        model = self.trainModel(x_train, y_train, modelType='keras')
        log.info('Model created. Computing recommendation')

        recommendationDict = collections.defaultdict(list)
        for df in candidate_df_list:
            can_id = float(df.head(1).candidate_id.values[0])
            # log.debug('can_id: %d' %can_id)
            # log.debug(type(can_id))

            # Pick the relevant column from the DF
            nparr = self.getModelDataFormat(df, False)

            pred_score = model.predict(nparr)

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

        return recommendationDict

    def getModelDataFormat(self, df, isTrain=True):
        '''
        Pick the relevant column from the DF.

        :param df: dataframe with all the feature
        :param isTrain: boolean flag to pick the gold label for training.
        Gold label is not in test, wrongly selected will cause error.
        :return: numarray of feature
        '''
        if isTrain:
            tmp = [np.concatenate([[row.doc2vec_score], [row.tfidf_score],
                                   [row.wmd_score], [row.wmd_score_sq], [row.wmd_score_sqrt],
                                   [row.bm25_score], [row.bm25_sqrt_score], [row.bm25_sq_score],
                                   [row.cv_len], [row.cv_len_sqrt], [row.cv_len_sq],
                                   [row.pmi_score], [row.pmi_score_sqrt], [row.pmi_score_sq],
                                   [row.jobdes_len], [row.jobdes_len_sqrt], [row.jobdes_len_sq],
                                   [row.app_state]])
                   for index, row in df.iterrows()]
        else:
            tmp = [np.concatenate([[row.doc2vec_score], [row.tfidf_score],
                                   [row.wmd_score], [row.wmd_score_sq], [row.wmd_score_sqrt],
                                   [row.bm25_score], [row.bm25_sqrt_score], [row.bm25_sq_score],
                                   [row.cv_len], [row.cv_len_sqrt], [row.cv_len_sq],
                                   [row.pmi_score], [row.pmi_score_sqrt], [row.pmi_score_sq],
                                   [row.jobdes_len], [row.jobdes_len_sqrt], [row.jobdes_len_sq]])
                   for index, row in df.iterrows()]

        return np.array(tmp)

    def prepareTrainData(self, isBalance):
        '''
        Compute the features for training

        :param isBalance: Some client has limited interaction data for interview and hire.
        Balancing the data improves the training
        :return: x_train = numpy array of feature, y_train = gold label
        '''
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

    def duplicate_rows(self, df, rec_list):

        original_row_count = df.shape[0]
        for _, row in df.iterrows():
            for i in range(len(rec_list) - 1):
                # Append this row at the end of the DataFrame
                df = df.append(row)

        # optional: sort it by index
        df.sort_index(inplace=True)

        tmp_list = []
        for i in range(original_row_count):
            tmp_list.extend(rec_list)

        df['opp_id'] = tmp_list

        return df

    def computeCandidateModelData(self, test_app_df, cv_df, graph_result, featureHelper, candidate_id):
        log.debug('Computing Candidate: %d' % candidate_id)
        df = test_app_df[test_app_df.candidate_id == candidate_id].copy()
        df = pd.merge(df, cv_df, how='left', on='cv name')

        df = self.duplicate_rows(df, graph_result[candidate_id])

        df = pd.merge(df, self.open_opp_df, how='left', on='opp_id')

        df = featureHelper.processFeatures(df, nWorker=1)

        return df

    def prepareTestData(self, graph_result, candidate_id_list):
        '''
        Compute the features for test candidates

        :param graph_result: List of opportunites to create feature against
        :param candidate_id_list: List of candidate_id to make recommmendation
        :return: x_train = numpy array of feature, y_train = gold label
        '''
        log.info('Creating test feature.')
        candidate_df_list = Parallel(n_jobs=20)(delayed(self.computeCandidateModelData)
                                                (self.test_app_df, self.cv_df, graph_result,
                                                 self.featureHelper, candidate_id)
                                                for candidate_id in candidate_id_list)
        return candidate_df_list

    def trainModel(self, x_train, y_train, modelType='keras'):
        '''
        Train model

        :param x_train: numpy training feature
        :param y_train: numpy gold label
        :param modelType: 2 model is available, keras and xgboost
        :return: trained model
        '''
        if modelType == 'keras':
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
            model = XGBRegressor(learning_rate=0.1, silent=True, objective='binary:logistic', nthread=-1,
                                 gamma=0.9,
                                 min_child_weight=2, max_delta_step=0, subsample=0.9, colsample_bytree=0.7,
                                 colsample_bylevel=1, reg_alpha=0.0009, reg_lambda=1, scale_pos_weight=1,
                                 base_score=0.5, seed=0, missing=None, max_depth=5, n_estimators=100)

            model.fit(x_train, y_train)

        return model

class GraphBasedBiDirectionalRecommender():
    '''
    Recommend opportunity to candidates based on bidirectional preference
    '''
    def __init__(self, train_app_df, test_app_df, opp_df, cv_filename):
        '''
        Init the object
        :param train_app_df: Training dataframe
        :param test_app_df: Test dataframe
        :param opp_df: Opportunity dataframe
        :param cv_filename: CV filename
        '''
        log.info('## Init Reciprocal Recommender ##')
        self.train_app_df = train_app_df
        self.test_app_df = test_app_df
        self.opp_df = opp_df
        self.cv_filename = cv_filename

    def getRecommendationResult_harmonic(self, topN=30):
        '''
        Get personalised recommendation all candidates in test_app_df

        :param topN: Number of recommendation per candidate
        :return:
        '''
        opp_recommender = GraphBasedOpportunityRecommender(self.train_app_df, self.test_app_df, self.opp_df)
        opp_result = opp_recommender.getRecommendationResult(topN=topN)

        cv_df = pd.read_csv(self.cv_filename)
        candidate_recommender = ContentBasedReRanker(self.train_app_df, self.test_app_df, self.opp_df, cv_df, opp_result)
        can_result = candidate_recommender.getRecommendationResult(topN=topN)

        candidate_id_list = list(opp_result.keys())

        final_result = {}
        for candidate_id in candidate_id_list:
            graph = opp_result[candidate_id]
            keras = can_result[candidate_id]

            rescore = {}
            for opp in graph:
                idx1 = graph.index(opp) + 1
                idx2 = keras.index(opp) + 1

                score = 1 / ((1 / idx1) + (1 / idx2))

                rescore[opp] = score
            sorted_opp = sorted(rescore, key=rescore.get, reverse=False)
            final_result[candidate_id] = sorted_opp

        return final_result

    def getRecommendationResult_schulze(self, topN=30):
        '''
        Get personalised recommendation all candidates in test_app_df

        :param topN: Number of recommendation per candidate
        :return:
        '''

        opp_recommender = GraphBasedOpportunityRecommender(self.train_app_df, self.test_app_df, self.opp_df)
        opp_result = opp_recommender.getRecommendationResult(topN=topN)

        cv_df = pd.read_csv(self.cv_filename)
        candidate_recommender = ContentBasedReRanker(self.train_app_df, self.test_app_df, self.opp_df, cv_df, opp_result)
        can_result = candidate_recommender.getRecommendationResult(topN=topN)

        candidate_id_list = list(opp_result.keys())

        final_result = {}
        for candidate_id in candidate_id_list:
            graph = opp_result[candidate_id]
            keras = can_result[candidate_id]

            graph = [[i] for i in graph]
            keras = [[i] for i in keras]

            ballots = []

            ballots.append({"count": 1, "ballot": graph})
            ballots.append({"count": 1, "ballot": keras})

            result_dict = SchulzeNPR(ballots, ballot_notation=CondorcetHelper.BALLOT_NOTATION_GROUPING).as_dict()

            final_result[candidate_id] = result_dict['order']

        return final_result

    def getRecommendationResult_lexicographic(self, topN=30):
        '''
        Get personalised recommendation all candidates in test_app_df

        :param topN: Number of recommendation per candidate
        :return:
        '''
        opp_recommender = GraphBasedOpportunityRecommender(self.train_app_df, self.test_app_df, self.opp_df)
        opp_result = opp_recommender.getRecommendationResult(topN=topN)

        cv_df = pd.read_csv(self.cv_filename)
        candidate_recommender = ContentBasedReRanker(self.train_app_df, self.test_app_df, self.opp_df, cv_df, opp_result)
        can_result = candidate_recommender.getRecommendationResult(topN=topN)

        candidate_id_list = list(opp_result.keys())

        final_result = {}

        for candidate_id in candidate_id_list:
            graph = opp_result[candidate_id]
            keras = can_result[candidate_id]

            lex_list = [] # list of [candidateid, rank1, rank2]
            for opp in graph:
                idx1 = graph.index(opp) + 1
                idx2 = keras.index(opp) + 1

                if idx1 > idx2:
                    lex_list.append([opp, idx2, idx1])
                else:
                    lex_list.append([opp, idx1, idx2])

            list_sorted = sorted(lex_list, key=operator.itemgetter(1, 2))
            list_sorted = [i[0] for i in list_sorted]

            final_result[candidate_id] = list_sorted

        return final_result