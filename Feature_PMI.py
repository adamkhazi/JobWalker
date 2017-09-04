import nltk
from nltk.util import ngrams
from collections import Counter
from math import log2
import pandas as pd
from joblib import Parallel, delayed
import logging as log
import Utilities

class Feature_PMI():
    bigrams=None
    unigrams=None
    corpussize=None

    def __init__(self, df):
        """
        Features for PMI.
        Will init corpus.
        :param text: The corpus
        """
        # Creating content
        df['content_str'] = df['content'].map(lambda x: self.__word_joiner(x))
        text = df['content_str'].str.cat(sep=' ')
        df.drop('content_str', axis=1, inplace=True)

        self._generateBigrams(text)
        self._generateUnigrams(text)
        self.corpussize=len(Utilities.CVTokeniser(text))
        print("Feature_PMI: Corpus size:",self.corpussize)

    def __word_joiner(self, s):
        return " ".join([word for word in s])

    def _generateNgrams(self,text,n=2):
        """
        Compute an ngram, given the test and N.
        :param text: The corpus
        :param n: The number, default bigram.
        :return:
        """
        token = Utilities.CVTokeniser(text)
        # token = nltk.word_tokenize(text)
        computedNgrams=ngrams(token,n)
        return Counter(computedNgrams)

    def _generateBigrams(self,text):
        """
        Generate and store the bigrams
        :param text: corpus
        :return:
        """
        self.bigrams=self._generateNgrams(text,2)

    def _generateUnigrams(self,text):
        """
        Generate and store the unigrams
        :param text: corpus
        :return:
        """
        self.unigrams=self._generateNgrams(text,1)

    def _getCountForBigram(self, word1, word2):
        """
        Return the count of occurances for bigram
        :param word1:
        :param word2:
        :return:
        """
        return self.bigrams[(word1,word2)]

    def _getCountForUnigram(self,word1):
        """
        Return the count of occurances for bigram
        :param word1:
        :return:
        """
        count=self.unigrams[(word1)]
        if count==0:
            count=0.001
        return count


    def computePMI(self, word1, word2):
        """
        Compute the PMI value of a bigram
        :param word1:
        :param word2:
        :return:
        """
        pmi=0
        if(word1 is not None and word2 is not None):
            #PMI = P(word1|word2)/P(word1)P(word2)
            #    = P(word2|word1)/P(word1)P(word2)
            P_w1w2 = self._getCountForBigram(word1,word2)/self.corpussize
            p_w1 = self._getCountForUnigram(word1)/self.corpussize
            p_w2 = self._getCountForUnigram(word2)/self.corpussize
            try:
                pmi=log2(P_w1w2/(p_w1*p_w2))
            except ValueError:
                pmi=99999
                # print("P_w1w2:",P_w1w2," p_w1:",p_w1," p_w2:",p_w2)
        return pmi

    def _getPMI(self, df, targetColname):
        """
        Internal method for dataframe apply
        :param df:
        :return:
        """
        pmi = 0
        search_term = df[targetColname]
        noofterms = len(search_term)
        startindex = 0
        pmiAccumulate = 0
        if(noofterms>1):
            for i in range(0,noofterms-1):
                pmi = self.computePMI(search_term[i],search_term[i+1])
                pmiAccumulate = pmiAccumulate+pmi
            pmiAccumulate = pmiAccumulate/noofterms
            pmi = pmiAccumulate
        return pmi

    def computePMIColumn(self, trainset, targetColname='cv_data'):
        """
        Compute the PMT for entire recordset.
        :param trainset:
        :param destColName:
        :param searchTermColname:
        :return:
        """
        if(~isinstance(trainset, pd.DataFrame)):
            if(trainset.shape[1]<2):
                if (targetColname not in list(trainset)):
                    log.error("Invalid Dataframe for PMI compute","Expecting A Pandas.Dataframe with columns:", targetColname)

        #Apply scoring function across each row of dataframe
        return trainset.apply(self._getPMI, args=(targetColname,), axis=1)
