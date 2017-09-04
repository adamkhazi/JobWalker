import re
import Stemmer
from nltk.corpus import stopwords
import nltk
import pandas as pd
# nltk.download()

english_vocab = set(w.lower() for w in nltk.corpus.words.words())
def isEnglish(tokens):
    '''
    Check if at least 30% of tokens are english
    :param tokens: list of words
    :return: boolean true or false. True if english else false
    '''
    token_count = [1 if token.lower() in english_vocab else 0 for token in tokens]

    total_tokens = len(tokens)
    english_token = sum(token_count)
    if total_tokens == 0:
        ratio = 0
    else:
        ratio = english_token/total_tokens

    if ratio < 0.3:
        return False
    else:
        return True

tokeniser = re.compile("(?:[A-Za-z]{1,2}\.)+|[\w\']+|\?\!")
def CVTokeniser(string):
    '''
    Default tokeniser for consistent tokenisation
    :param string:
    :return:
    '''
    return tokeniser.findall(string)

cleanr = re.compile('\[.+?\]')
def removeOpportunityTag(str):
    '''
    Used to remove tags found in opportunity description
    :param str:
    :return:
    '''
    cleantext = re.sub(cleanr, ' ', str)
    return cleantext

def remove_nonascii_char(s):
    '''
    Remove all non ascii char
    :param s:
    :return:
    '''
    return "".join(letter for letter in s if ord(letter) < 128)

stemmer = Stemmer.Stemmer('english')
def performStemming(data):
    '''
    Perform word stemming

    :param data:
    :return:
    '''
    return " ".join([stemmer.stemWord(word.lower()) for word in CVTokeniser(data)])

def removeStopword(data):
    '''
    Remove all stop words
    :param data:
    :return:
    '''
    return " ".join([word for word in CVTokeniser(data) if not word.lower() in stopwords.words('english')])

def processOpp(opp_data):
    '''
    Wrapper function to process opportunity description
    :param opp_data: Opportunity description
    :return: Cleaned up opportunity description
    '''
    opp_data = removeOpportunityTag(opp_data)
    opp_data = remove_nonascii_char(opp_data)
    opp_data = removeStopword(opp_data)
    return CVTokeniser(opp_data)

def processCV(cv_data):
    '''
    Wrapper function to process CV
    :param cv_data: CV
    :return: Cleaned up CV
    '''
    cv_data = remove_nonascii_char(cv_data)
    cv_data = removeStopword(cv_data)
    return CVTokeniser(cv_data)

def getColdStartCandidate(train_df, gold_df):
    can_id_train_unique = train_df['candidate_id'].unique()
    can_id_gold_unique = gold_df['candidate_id'].unique()
    # print('Number of unique Candidate in Gold: %d' % len(can_id_gold_unique))
    # print('Number of cold user: %d' % len(set(can_id_gold_unique) - set(can_id_train_unique)))
    cold_candidates = set(set(can_id_gold_unique) - set(can_id_train_unique))
    warm_candidates = set(set(can_id_gold_unique) - cold_candidates)
    return [cold_candidates, warm_candidates]

CV_base_folder = '/u01/bigdata/vX/CV/CV/'
def convertCVName2FolderStructure(cvName, includeBaseFolder=True):
    '''
    Helper function
    Convert file_private_200_838a111dd7e2bc831f4f28c84e5a96b9ef9bb612.pdf to
    /u01/bigdata/vX/CV/CV/200/8/83/838/838a/838a1/838a111dd7e2bc831f4f28c84e5a96b9ef9bb612.txt
    :param cvName:
    :return:
    '''
    tokens = cvName.split(sep='_')
    clientid = str(tokens[2])

    if len(clientid)==2:
        clientid_3 = '0' + clientid
    else:
        clientid_3 = clientid

    h = tokens[3].split(sep='.')
    hash = h[0]
    extension = h[1]

    # print(cvName)
    if includeBaseFolder:
        folder = CV_base_folder + clientid_3 + '/' + hash[0] \
               + '/' + hash[0:2] \
               + '/' + hash[0:3] \
               + '/' + hash[0:4] \
               + '/' + hash[0:5] \
               + '/file_private_' + clientid + '_' + hash
    else:
        folder = 'file_private_' + clientid + '_' + hash

    if extension == 'docx' or extension == 'pdf':
        folder = folder + '.txt'
    elif extension == 'doc':
        folder = folder + '.doc.txt'

    return folder

# Create a function called "chunks" with two arguments, l and n:
def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

def df_crossjoin(df1, df2, **kwargs):
    df1['_tmpkey'] = 1
    df2['_tmpkey'] = 1

    res = pd.merge(df1, df2, on='_tmpkey', **kwargs).drop('_tmpkey', axis=1)
    res.index = pd.MultiIndex.from_product((df1.index, df2.index))

    df1.drop('_tmpkey', axis=1, inplace=True)
    df2.drop('_tmpkey', axis=1, inplace=True)

    return res


if __name__ == "__main__":
    print(convertCVName2FolderStructure('file_private_200_838a111dd7e2bc831f4f28c84e5a96b9ef9bb612.pdf'))