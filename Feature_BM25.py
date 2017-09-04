from gensim.corpora import Dictionary, MmCorpus
from gensim.summarization.bm25 import BM25
import numpy as np
import pandas as pd
from collections import defaultdict
import logging as log
from joblib import Parallel, delayed
import Utilities


class Feature_BM25():
    def __init__(self, df):
        log.info('Init BM25')
        # Convert documents into corpus
        log.debug("Converting documents into corpus")
        df['content_str'] = df['content'].map(lambda  x: self.__word_joiner(x))
        self.__convertToCorpus(df['content_str'])
        df.drop('content_str', axis=1, inplace=True)

        # Initialise the BM25 computation
        log.debug("Computing BM25 baselines")
        self.bm25 = BM25(self.corpus)
        self.avgIDF = sum(map(lambda k: float(self.bm25.idf[k]), self.bm25.idf.keys())) / len(self.bm25.idf.keys())

        # Mapping of opp_id to indices of BM25 documents
        log.debug("Mapping opp_id to corpus index")
        self.productDict = defaultdict(int)
        counter = 0
        for oppid in df['opp_id']:
            self.productDict[oppid] = counter
            counter = counter + 1

    def getBM25_Score(self, df, targetColumnName):
        # Apply scoring function across each row of dataframe
        return df.apply(self.getBM25, args=(targetColumnName,), axis=1)

    def getBM25(self, row, targetColumnName):
        """
        Meant to be use in panda apply.
        See below
        :param df:
        :return:
        """
        oppid = row['opp_id']
        cv_data = row[targetColumnName]
        return self.score(cv_data, oppid)

    def score(self, cv_data, oppid):
        """
        BM25 will return BM25(query1, doc1).
        if documnent doesn't exists, it will be ignored.
        :param query: Expecting a list of words. ['One','Two','Three']
        :param document: A  'product_id'
        :return:
        """
        #Convert query into vector
        queryVector = self.__getVectorForDocument(cv_data)[0]


        # Find out the index of the document in the document corpus
        documentindex = self.productDict[oppid]

        score=self.bm25.get_score(queryVector, documentindex, self.avgIDF)
        return score

    def __word_joiner(self, s):
        return " ".join([word for word in s])

    def __convertToCorpus(self, documents):
        """
        Steps to make the documents compatible to gensim
        :param documents:
        :return:
        """
        # Preprocessing the text
        text = self.__getBagOfWords(documentDF=documents, return_type='document_tokens')

        # Create a Gensim text corpus based on documents
        log.debug("Creating a text dictionary")
        self.dictionary = Dictionary(line.split() for line in documents)

        # Create a Gensim document corpus based on text corpus and each document
        self.corpus = [self.dictionary.doc2bow(line) for line in text]
        log.info("Saving corpus to file")
        MmCorpus.serialize('model/opp_corpus.mm', self.corpus)
        self.corpus = MmCorpus('model/opp_corpus.mm')

    def __getBagOfWords(self, documentDF=None, return_type='document_tokens'):
        """
        To retrieve a bag of words from the documents (No pre-processing except tokenisation)
        :param documentDF: Strictly a Nx1 Dataframe. Row representing document, Col representing a string of words
        :param return_type: 'document_tokens' returns a NxM array (Sparse), each row representing a document, each col representing a token of the document.
                            'array_tokens' returns a Nx1 array, each row representing a token
        :return:
        """

        texts=[]
        if(return_type=='document_tokens'):
            log.debug("Document level tokenisation")
            texts=[words for words in (document.split() for document in documentDF)]
        elif(return_type=='array_tokens'):
            log.debug("Highest level tokenisation")
            for words in (document.split() for document in documentDF):
                for word in words:
                    texts.append(word)
        return texts

    def __getVectorForDocument(self, document):
        """
        Convert the document into list of vectors .
        :param document: document, can also be a query. Expecting a list of words.
        :return:
        """
        document_vector = [self.dictionary.doc2bow(text) for text in [document]]
        return document_vector
