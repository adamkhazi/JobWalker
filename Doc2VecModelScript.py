from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import os
import Utilities
import time
import collections
import numpy as np
import logging as log
import pandas as pd
from joblib import Parallel, delayed

log.basicConfig(level=log.INFO,
                # filemode='w',
                # filename='01_Client200Doc2vecTraining20170718.log',
                format='%(asctime)s; %(module)s; %(funcName)s; %(levelname)s; %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')

def createDoc2Vec_AllCV_Opp_Model_Parallel(opp_filename, CVFolderName, doc2vec_model_name):
    log.info("Creating Doc2Vec Model using CV and Opp documents. Takes ~ 1 hour.")

    opp_df = pd.read_csv(opp_filename)
    log.info('Number of unique opp id in opp.csv: %d' % len(opp_df['opp_id'].unique()))

    cv_data = {}
    log.info("Reading in CV data")

    # Scan the entire dir for .txt file
    for root, dirs, files in os.walk(os.path.expanduser(CVFolderName)):
        for f in files:
            fname = os.path.join(root, f)

            data = ''
            if not fname.endswith(".txt"):
                continue
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

            if len(data) > 500:
                if Utilities.isEnglish(Utilities.CVTokeniser(data)):
                    cv_data[f] = data
            else:
                log.debug('Empty File: %s' % fname)

    # print(len(cv_data))
    log.info("Processing CV data into Doc2Vec format")
    app_trainingDocuments = Parallel(n_jobs=55)(delayed(getCVTaggedDoc)(key, data) for key, data in cv_data.items())

    log.info('Number of CV added in Doc2Vec for training: %d' % len(app_trainingDocuments))


    ##################################################################
    log.info("Preparing Opp data")
    opp_df['content'] = opp_df['title'].map(str) + " " + \
                        opp_df['role_specifics'].map(str) + " " + \
                        opp_df['skills'].map(str)
    opp_data = {}
    # Prepare the documents in gensim format
    for index, row in opp_df.iterrows():
        opp_data[row.opp_id] = row.content

    opp_trainingDocuments = Parallel(n_jobs=55)(delayed(getOppTaggedDoc)(key, data) for key, data in opp_data.items())
    log.info('Number of Opp added in Doc2Vec for training: %d' % len(opp_trainingDocuments))

    trainingDocuments = app_trainingDocuments + opp_trainingDocuments
    log.info('Total doc in Doc2Vec for training: %d' % len(trainingDocuments))

    __trainDoc2VecModel(trainingDocuments=trainingDocuments,
                        Epoch=30,
                        doc2vec_model_name=doc2vec_model_name)

def createDoc2Vec_CV_Opp_Model_Parallel(app_filename, opp_filename, doc2vec_model_name):
    log.info("Creating Doc2Vec Model using CV and Opp documents. Takes ~ 1 hour.")

    app_df = pd.read_csv(app_filename)
    opp_df = pd.read_csv(opp_filename)
    log.info('Number of unique CV in app.csv: %d' %len(app_df['cv name'].unique()))
    log.info('Number of unique opp id in opp.csv: %d' % len(opp_df['opp_id'].unique()))

    cv_data = {}
    log.info("Reading in CV data")
    for index, row in app_df.iterrows():
        fname = Utilities.convertCVName2FolderStructure(row['cv name'])

        try:
            with open(fname, encoding='utf8') as cv:
                data = cv.read()
        except UnicodeDecodeError:
            try:
                with open(fname, encoding='ISO-8859-1') as cv:
                    data = cv.read()
            except Exception:
                log.error(Exception)

        if len(data) > 500:
            if Utilities.isEnglish(Utilities.CVTokeniser(data)):
                cv_data[row['cv name']] = data
        # else:
        #     log.error('Should not have such error. Filename: %s' % fname)

    # print(len(cv_data))
    log.info("Processing CV data into Doc2Vec format")
    app_trainingDocuments = Parallel(n_jobs=55)(delayed(getCVTaggedDoc)(key, data) for key, data in cv_data.items())

    log.info('Number of CV added in Doc2Vec for training: %d' % len(app_trainingDocuments))


    ##################################################################
    log.info("Preparing Opp data")
    opp_df['content'] = opp_df['title'].map(str) + " " + \
                        opp_df['role_specifics'].map(str) + " " + \
                        opp_df['skills'].map(str)
    opp_data = {}
    # Prepare the documents in gensim format
    for index, row in opp_df.iterrows():
        opp_data[row.opp_id] = row.content

    opp_trainingDocuments = Parallel(n_jobs=55)(delayed(getOppTaggedDoc)(key, data) for key, data in opp_data.items())
    log.info('Number of Opp added in Doc2Vec for training: %d' % len(opp_trainingDocuments))

    trainingDocuments = app_trainingDocuments + opp_trainingDocuments
    log.info('Total doc in Doc2Vec for training: %d' % len(trainingDocuments))

    __trainDoc2VecModel(trainingDocuments=trainingDocuments,
                        Epoch=30,
                        doc2vec_model_name=doc2vec_model_name)

def createDoc2Vec_CV_Model_Parallel(app_Filename='data/app_200.csv', doc2vec_model_name='model/doc2vec_trainedvocab_200_CV.d2v'):
    log.info("Creating Doc2Vec Model using CV documents. Takes ~ 1 hour.")

    app_df = pd.read_csv(app_Filename)
    log.info('Number of unique CV in app.csv: %d' %len(app_df['cv name'].unique()))

    cv_data = {}
    log.info("Reading in CV data")
    for index, row in app_df.iterrows():
        fname = Utilities.convertCVName2FolderStructure(row['cv name'])

        try:
            with open(fname, encoding='utf8') as cv:
                data = cv.read()
        except UnicodeDecodeError:
            try:
                with open(fname, encoding='ISO-8859-1') as cv:
                    data = cv.read()
            except Exception:
                log.error(Exception)

        if len(data) > 500:
            if Utilities.isEnglish(Utilities.CVTokeniser(data)):
                cv_data[row['cv name']] = data
        else:
            log.error('Should not be such error. Filename: %s' % fname)

    # print(len(cv_data))
    log.info("Processing CV data into Doc2Vec format")
    app_trainingDocuments = Parallel(n_jobs=55)(delayed(getCVTaggedDoc)(key, data) for key, data in cv_data.items())

    log.info('Number of CV added in Doc2Vec for training: %d' % len(app_trainingDocuments))


    __trainDoc2VecModel(trainingDocuments=app_trainingDocuments,
                        Epoch=30,
                        doc2vec_model_name=doc2vec_model_name)

def createDoc2Vec_AllCV_Model_Parallel(CVFolderName, doc2vec_model_name):
    log.info("Creating Doc2Vec Model using All CV documents. Takes ~ 1 hour.")

    cv_data = {}
    log.info("Reading in CV data")

    # Scan the entire dir for .txt file
    for root, dirs, files in os.walk(os.path.expanduser(CVFolderName)):
        for f in files:
            fname = os.path.join(root, f)

            data = ''
            if not fname.endswith(".txt"):
                continue
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

            if len(data) > 500:
                if Utilities.isEnglish(Utilities.CVTokeniser(data)):
                    cv_data[f] = data
            else:
                log.debug('Empty File: %s' % fname)

    # print(len(cv_data))
    log.info("Processing CV data into Doc2Vec format")
    app_trainingDocuments = Parallel(n_jobs=55)(delayed(getCVTaggedDoc)(key, data) for key, data in cv_data.items())

    log.info('Number of CV added in Doc2Vec for training: %d' % len(app_trainingDocuments))

    __trainDoc2VecModel(trainingDocuments=app_trainingDocuments,
                        Epoch=30,
                        doc2vec_model_name=doc2vec_model_name)

def getOppTaggedDoc(opp_key, opp_data):
    return TaggedDocument(Utilities.processCV(opp_data), ['oppid_' + str(opp_key)])


def getCVTaggedDoc(cv_key, cv_data):
    return TaggedDocument(Utilities.processCV(cv_data), [cv_key])

def __trainDoc2VecModel(trainingDocuments, Epoch, doc2vec_model_name):
    log.info('######### Creating Doc2Vec Model ##############')
    log.info('## Number of document for train: %d' %len(trainingDocuments))
    start_time = time.time()
    model = Doc2Vec(size=300, min_count=5, window=5, iter=Epoch, workers=16, alpha=0.1, min_alpha=0.0001,
                    dm_concat=1, dm=0, negative=5)

    # Build the vocab using gensim format
    model.build_vocab(trainingDocuments)

    model.train(trainingDocuments, total_examples=model.corpus_count, epochs=model.iter)

    # Start training with random shuffle in every epoch
    # for epoch in range(Epoch):
    #     # shuffle(trainingDocuments)
    #     # model.train(trainingDocuments)
    #     model.train(trainingDocuments, total_examples=model.corpus_count)

    # Save the model to disk
    model.save(doc2vec_model_name)
    log.info("## trainDoc2VecModel took: %s minutes" % round(((time.time() - start_time) / 60), 2))

    return model

def main():
    # createDoc2Vec_CV_Opp_Model_Parallel(app_filename='data/app_048.csv',
    #                                     opp_filename='data/opp_048.csv',
    #                                     doc2vec_model_name='model/doc2vec_trainedvocab_048_CV_Opp_1_nostem.d2v')

    # createDoc2Vec_CV_Model_Parallel(app_Filename='data/app_048.csv',
    #                                 doc2vec_model_name='model/doc2vec_trainedvocab_048_CV_1_nostem.d2v')
    #
    createDoc2Vec_AllCV_Opp_Model_Parallel(opp_filename='data/opp_331.csv',
                                           CVFolderName = '/u01/bigdata/vX/CV/CV/331',
                                           doc2vec_model_name='model/doc2vec_trainedvocab_331_AllCV_Opp_1_nostem.d2v')
    #
    # createDoc2Vec_AllCV_Model_Parallel(CVFolderName = '/u01/bigdata/vX/CV/CV/048',
    #                                    doc2vec_model_name='model/doc2vec_trainedvocab_048_AllCV_1_nostem.d2v')


if __name__ == "__main__":
    main()