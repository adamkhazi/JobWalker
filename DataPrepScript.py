import pandas as pd
import json
import os
import numpy as np
import re
import sys
from datetime import datetime
from collections import defaultdict
import Enumeration
import Utilities
import logging as log
from joblib import Parallel, delayed
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from dateutil.parser import parse
import Doc2VecModelScript

log.basicConfig(level=log.DEBUG,
                # filemode='w',
                # filename='01_422DataPrep.log',
                format='%(asctime)s; %(module)s; %(funcName)s; %(levelname)s; %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')

def readApplicationJSON2CSV(data_source_dir, dest_csv):
    # Read the following fields from Application JSON
    # 1. opp_id
    # 2. state
    # 3. app_id
    # 4. first name
    # 5. middle name
    # 6. last name
    # 7. email
    # 8. phone
    # 9. json source file name
    # into a csv

    log.info('### Processing raw application JSON ###')
    log.info('Source Directory: %s' %data_source_dir)
    # Create a Pandas DF to store the data
    columnsName = ['app_id', 'opp_id', 'app_state', 'first_name', 'mid_name',
                               'last_name', 'email', 'phone', 'app_source_json']
    df = pd.DataFrame(columns=columnsName)
    # df = pd.DataFrame()
    count = 0
    emailFound = 0
    firstnameFound = 0
    lastnameFound = 0
    midnameFound = 0
    phoneFound = 0
    keyDict = {}
    tmpList = []

    # Scan the entire dir for .json file
    for root, dirs, files in os.walk(os.path.expanduser(data_source_dir)):
        for f in files:
            fname = os.path.join(root, f)

            if not fname.endswith(".json"):
                continue
            with open(fname, encoding='utf8') as js:
                result = {}
                data = js.read()
                j_data = json.loads(data)

                result['app_state'] = j_data['state']
                result['app_id'] = j_data['id']
                result['opp_id'] = j_data['opp_id']
                result['app_source_json'] = fname
                result['first_name'] = ''
                result['mid_name'] = ''
                result['last_name'] = ''
                result['email'] = ''
                result['phone'] = []

                # print(fname)

                buf = __id_generator(j_data)
                # print('result================================')
                # print(buf)

                for k, v in buf.items():
                    keyDict[k] = 1

                    if k == 'Cell / Mobile Number' or k == 'Telephone Number (include area code)' or k =='Mobile' \
                            or k=='Primary phone number' or k=='Phone Number with Area Code (mobile preferred)' \
                            or k=='Telephone/Mobile Number (with country prefix eg. 0048)' or k=='Mobile/Cell' or \
                            k == 'Phone number':
                        result['phone'].append(v)
                        phoneFound += 1
                    elif k == 'Email' or k=='E-mail' or k=='Non University Email address' \
                            or k=='Primary Email' or k=='E-mail Address' or k=='Email' or k=='E-mail address'\
                            or k=='Contact email':
                        result['email'] = v
                        emailFound += 1
                    elif k == 'First / Legal Name' or k == 'First Name(s)' or k=='First name' or k=='First Name':
                        result['first_name'] = v
                        firstnameFound += 1
                    elif k == 'Last / Family Name' or k == 'Surname' or k=='Last name' or k=='Last/Family Name' \
                            or k=='Last Name (Family Name)':
                        result['last_name'] = v
                        lastnameFound += 1
                    elif k == 'Middle Name / Initial' or k=='Middle initial' or k=='Middle Name/Initial':
                        result['mid_name'] = v
                        midnameFound += 1

                start_time = datetime.now()
                # df.loc[df.shape[0]] = result # Inserting one at a time is bloody slow
                tmpList.append(result)
                # df.loc[count] = result
                end_time = datetime.now()


                if count % 5000 == 0:
                    log.debug('%d JSON Done' %count)
                    # print('Duration: {}'.format(end_time - start_time))

                    tmpDF = pd.DataFrame(tmpList, columns=columnsName)

                    df = df.append(tmpDF, True)
                    # print('head: ====== %s' % df.head(1))
                    tmpList = []
                    sys.stdout.flush()
                count += 1

    # Flush the last batch into dataframe
    log.info('Number of processed JSON: %d' % count)
    # print('Duration: {}'.format(end_time - start_time))
    tmpDF = pd.DataFrame(tmpList, columns=columnsName)
    df = df.append(tmpDF)
    tmpList = []
    sys.stdout.flush()


    log.info('Writing CSV to: %s' %dest_csv)
    df.to_csv(dest_csv, index=False)
    log.info('Number of application in CSV: %d' %df.shape[0])
    log.info('Number of unique opportunity id in CSV: %d' % len(df['opp_id'].unique()))

    log.debug('Email Found: %d' %emailFound)
    log.debug('first name found: %d' %firstnameFound)
    log.debug('last name found: %d' %lastnameFound)
    log.debug('mid name found: %d' %midnameFound)
    log.debug('phone found: %d' %phoneFound)
    sys.stdout.flush()

    # for k, v in keyDict.items():
    #     print(k)


def __id_generator(dic):
    # print(dic)
    result = {}
    for k, v in dic.items():
        # print(k)
        if k == 'question':
            # print(v)
            if 'name' in str(v).lower() or 'initial' in str(v).lower():
                result[v] = dic['answer']
            #     if 'last' in str(v).lower():
            #         result[v] = dic['answer']
            #         # if dic['answer'] != '':
            #         #     result['last_name'] = dic['answer']
            #     elif 'first' in str(v).lower():
            #         result[v] = dic['answer']
            #         # if dic['answer'] != '':
            #         #     result['first_name'] = dic['answer']
            #     elif 'middle' in str(v).lower():
            #         result[v] = dic['answer']
            #         # if dic['answer'] != '':
            #         #     result['mid_name'] = dic['answer']
            # elif 'title' in str(v):
            #     result[v] = dic['answer']
            elif 'email' in str(v).lower() or 'e-mail' in str(v).lower() :
                result[v] = dic['answer']
                # result['email'] = dic['answer']
            elif re.match(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)", dic['answer']):
                result['E-mail'] = dic['answer']
            elif 'phone' in str(v).lower() or 'mobile' in str(v).lower() or 'contact number' in str(v).lower():
                result[v] = dic['answer']
                # if 'phone' not in result:
                #     result['phone'] = [] # kenna overwritten
                # result['phone'].append(dic['answer'])
            # yield v
        elif isinstance(v, dict):
            # print(k)
            # print(v)
            result.update(__id_generator(v))
        elif isinstance(v, list):
            # print(v)
            for i in v:
                if isinstance(i, dict):
                    result.update(__id_generator(i))
            # for id_val in __id_generator(v):
            #     print(id_val)

    return result

def joinAppWithCVandReport(app_csv_filename, id_csv_filename, cv_csv_filename, dest_csv):
    log.info('### Joining App.csv with CV filename and report')
    log.info('Source app csv: %s' %app_csv_filename)
    log.info('CV filename: %s' %cv_csv_filename)
    log.info('Report filename: %s' %id_csv_filename)
    id_df = pd.read_csv(id_csv_filename)
    app_df = pd.read_csv(app_csv_filename)
    cv_df = pd.read_csv(cv_csv_filename)
    # print(id_df.head(1))

    result_df = pd.merge(app_df, id_df, how='left',
                         left_on=['app_id', 'opp_id'],
                         right_on=['app_id', 'opp_id'])
    log.debug(result_df.info())
    result_df = pd.merge(result_df, cv_df, how='left',
                         left_on=['app_id', 'opp_id', 'candidate_id'],
                         right_on=['app_id', 'opp_id', 'candidate_id'])



    log.debug('Number of NAN candidate id: %d' % result_df['candidate_id'].isnull().sum())
    log.debug('Number of NAN app id: %d' % result_df['app_id'].isnull().sum())
    log.debug('Number of NAN opp id: %d' % result_df['opp_id'].isnull().sum())
    result_df.drop(result_df[result_df['candidate_id'].isnull()].index, inplace=True)
    result_df.drop(result_df[result_df['app_id'].isnull()].index, inplace=True)
    result_df.drop(result_df[result_df['opp_id'].isnull()].index, inplace=True)
    log.debug('Number of NAN candidate id: %d' % result_df['candidate_id'].isnull().sum())
    log.debug('Number of NAN app id: %d' % result_df['app_id'].isnull().sum())
    log.debug('Number of NAN opp id: %d' % result_df['opp_id'].isnull().sum())

    result_df.to_csv(dest_csv, index=False)
    log.debug(result_df.info())

    log.info('Rows in %s: %d ' % (app_csv_filename, app_df.shape[0]))
    log.info('Rows in %s: %d ' % (id_csv_filename, id_df.shape[0]))
    log.info('Rows in %s: %d ' % (cv_csv_filename, cv_df.shape[0]))
    log.info('Rows in %s: %d ' % (dest_csv, result_df.shape[0]))

def fileCounter(data_source_dir):
    dirs = [d for d in os.listdir(data_source_dir) if os.path.isdir(os.path.join(data_source_dir, d))]
    dirs.sort()
    print(dirs)

    for client_dir in dirs:
        counter = 0
        # Scan the entire dir
        # cpt = sum([len(files) for r, d, files in os.walk(data_source_dir+'/'+client_dir)])
        # print('%s : %d' % (client_dir, cpt))
        for root, dirs, files in os.walk(os.path.expanduser(data_source_dir+'/'+client_dir)):
            counter += len(files)
            # print(counter)
            # for f in files:
            #     fname = os.path.join(root, f)
        print('%s : %d' %(client_dir, counter))

def readOppJSON2CSV_048(destFilename):

    data_source_dir = '/backup/vX/apps_opps/opp/048'
    dest_csv = destFilename
    log.info('Reading Client 048 Opp JSON from: %s' %data_source_dir)
    # Read the following fields from Application JSON
    # 1. opp_id
    # 2. state
    # 3. title
    # 4. country
    # 5. location
    # 6. role_type
    # 7. role_specifics
    # 8. skills
    # 9. others
    # 10. json source file name
    # into a csv

    # Create a Pandas DF to store the data
    columnsName = ['opp_id', 'opp_state', 'title', 'country', 'location', 'salary', 'hours', 'department', 'contract_type',
                   'role_type', 'role_specifics', 'skills', 'others', 'opp_source_json']
    df = pd.DataFrame(columns=columnsName)

    counterDict = defaultdict(int)
    keysDict = defaultdict(int)
    tmpList = []
    # Scan the entire dir for .json file
    for root, dirs, files in os.walk(os.path.expanduser(data_source_dir)):
        for f in files:
            fname = os.path.join(root, f)

            if not fname.endswith(".json"):
                continue
            with open(fname, encoding='utf8') as js:
                result = {}
                data = js.read()
                j_data = json.loads(data)

                result['opp_state'] = j_data['state']
                result['title'] = j_data['title']
                result['opp_id'] = j_data['id']
                result['opp_source_json'] = fname
                result['country'] = ''
                result['location'] = ''
                result['role_type'] = ''
                result['role_specifics'] = ''
                result['skills'] = ''
                result['others'] = ''
                result['salary'] = -1
                result['hours'] = -1
                result['department'] = ''
                result['contract_type'] = ''

                # print(fname)
                counterDict['json'] += 1

                buf = __QnA_generator(j_data)

                for k, v in buf.items():
                    keysDict[k] += 1
                    if k == '[b]Job Description[/b]' or k == '[b]Description[/b]' :
                        result['role_specifics'] = v
                        counterDict['role_specifics'] += 1
                    elif k == '[b]Location[/b]':
                        result['location'] = v
                        counterDict['location'] += 1
                    elif k == '[b]Business Area[/b]':
                        result['department'] = v
                        counterDict['department'] += 1
                    elif k == '[b]Position[/b]':
                        result['role_type'] = v
                        counterDict['role_type'] += 1



                start_time = datetime.now()
                tmpList.append(result)
                end_time = datetime.now()

                if counterDict['json'] % 5000 == 0:
                    log.debug('%d JSON Done' % counterDict['json'])
                    # print('Duration: {}'.format(end_time - start_time))

                    tmpDF = pd.DataFrame(tmpList, columns=columnsName)

                    df = df.append(tmpDF, True)
                    # print('head: ====== %s' % df.head(1))
                    tmpList = []
                    sys.stdout.flush()

    # Flush the last batch into dataframe
    log.info('Number of Opp JSON processed: %d' % counterDict['json'])
    # print('Duration: {}'.format(end_time - start_time))
    tmpDF = pd.DataFrame(tmpList, columns=columnsName)
    df = df.append(tmpDF)
    tmpList = []
    sys.stdout.flush()

    df = df.drop(df[df.role_specifics.str.len() < 100].index)

    log.info('Writing Opp CSV to: %s' %dest_csv)
    df.to_csv(dest_csv, index=False)
    log.info('Number of Opp: %d' %df.shape[0])
    log.info('Number of Unique Opp: %d' % len(df['opp_id'].unique()))

    print(counterDict)
    # for k, v in keysDict.items():
    #     print('%s: %d' %(k, v))
    # sys.stdout.flush()

def readOppJSON2CSV_422(destFilename):

    data_source_dir = '/backup/vX/apps_opps/opp/422'
    dest_csv = destFilename
    log.info('Reading Client 422 Opp JSON from: %s' %data_source_dir)
    # Read the following fields from Application JSON
    # 1. opp_id
    # 2. state
    # 3. title
    # 4. country
    # 5. location
    # 6. role_type
    # 7. role_specifics
    # 8. skills
    # 9. others
    # 10. json source file name
    # into a csv

    # Create a Pandas DF to store the data
    columnsName = ['opp_id', 'opp_state', 'title', 'country', 'location', 'salary', 'hours', 'department', 'contract_type',
                   'role_type', 'role_specifics', 'skills', 'others', 'opp_source_json']
    df = pd.DataFrame(columns=columnsName)

    counterDict = defaultdict(int)
    keysDict = defaultdict(int)
    tmpList = []
    # Scan the entire dir for .json file
    for root, dirs, files in os.walk(os.path.expanduser(data_source_dir)):
        for f in files:
            fname = os.path.join(root, f)

            if not fname.endswith(".json"):
                continue
            with open(fname, encoding='utf8') as js:
                result = {}
                data = js.read()
                j_data = json.loads(data)

                result['opp_state'] = j_data['state']
                result['title'] = j_data['title']
                result['opp_id'] = j_data['id']
                result['opp_source_json'] = fname
                result['country'] = ''
                result['location'] = ''
                result['role_type'] = ''
                result['role_specifics'] = ''
                result['skills'] = ''
                result['others'] = ''
                result['salary'] = -1
                result['hours'] = -1
                result['department'] = ''
                result['contract_type'] = ''

                # print(fname)
                counterDict['json'] += 1

                buf = __QnA_generator(j_data)

                for k, v in buf.items():
                    keysDict[k] += 1
                    if k == 'Job description' :
                        result['role_specifics'] = v
                        counterDict['role_specifics'] += 1
                    elif k == 'Address / location of vacancy':
                        result['location'] = v
                        counterDict['location'] += 1
                    # elif k == '[b]Business Area[/b]':
                    #     result['department'] = v
                    #     counterDict['department'] += 1
                    elif k == 'Role Type':
                        result['role_type'] = v
                        counterDict['role_type'] += 1
                    elif k == 'Hours':
                        result['hours'] = v
                        counterDict['hours'] += 1


                start_time = datetime.now()
                tmpList.append(result)
                end_time = datetime.now()

                if counterDict['json'] % 5000 == 0:
                    log.debug('%d JSON Done' % counterDict['json'])
                    # print('Duration: {}'.format(end_time - start_time))

                    tmpDF = pd.DataFrame(tmpList, columns=columnsName)

                    df = df.append(tmpDF, True)
                    # print('head: ====== %s' % df.head(1))
                    tmpList = []
                    sys.stdout.flush()

    # Flush the last batch into dataframe
    log.info('Number of Opp JSON processed: %d' % counterDict['json'])
    # print('Duration: {}'.format(end_time - start_time))
    tmpDF = pd.DataFrame(tmpList, columns=columnsName)
    df = df.append(tmpDF)
    tmpList = []
    sys.stdout.flush()

    df = df.drop(df[df.role_specifics.str.len() < 100].index)

    log.info('Writing Opp CSV to: %s' %dest_csv)
    df.to_csv(dest_csv, index=False)
    log.info('Number of Opp: %d' %df.shape[0])
    log.info('Number of Unique Opp: %d' % len(df['opp_id'].unique()))

    # print(counterDict)
    # for k, v in keysDict.items():
    #     print('%s: %d' %(k, v))
    # sys.stdout.flush()

def readOppJSON2CSV_331(destFilename):

    data_source_dir = '/backup/vX/apps_opps/opp/331'
    dest_csv = destFilename
    log.info('Reading Client 331 Opp JSON from: %s' %data_source_dir)
    # Read the following fields from Application JSON
    # 1. opp_id
    # 2. state
    # 3. title
    # 4. country
    # 5. location
    # 6. role_type
    # 7. role_specifics
    # 8. skills
    # 9. others
    # 10. json source file name
    # into a csv

    # Create a Pandas DF to store the data
    columnsName = ['opp_id', 'opp_state', 'title', 'country', 'location', 'salary', 'hours', 'department', 'contract_type',
                   'role_type', 'role_specifics', 'skills', 'others', 'opp_source_json']
    df = pd.DataFrame(columns=columnsName)

    counterDict = defaultdict(int)
    keysDict = defaultdict(int)
    tmpList = []
    # Scan the entire dir for .json file
    for root, dirs, files in os.walk(os.path.expanduser(data_source_dir)):
        for f in files:
            fname = os.path.join(root, f)

            if not fname.endswith(".json"):
                continue
            with open(fname, encoding='utf8') as js:
                result = {}
                data = js.read()
                j_data = json.loads(data)

                result['opp_state'] = j_data['state']
                result['title'] = j_data['title']
                result['opp_id'] = j_data['id']
                result['opp_source_json'] = fname
                result['country'] = ''
                result['location'] = ''
                result['role_type'] = ''
                result['role_specifics'] = ''
                result['skills'] = ''
                result['others'] = ''
                result['salary'] = -1
                result['hours'] = -1
                result['department'] = ''
                result['contract_type'] = ''

                # print(fname)
                counterDict['json'] += 1

                buf = __QnA_generator(j_data)

                for k, v in buf.items():
                    keysDict[k] += 1
                    if k == 'Job Description' :
                        result['role_specifics'] = v
                        counterDict['role_specifics'] += 1
                    elif k == 'Location':
                        result['location'] = v
                        counterDict['location'] += 1
                    elif k == 'Business Unit':
                        result['department'] = v
                        counterDict['department'] += 1
                    elif k == 'Job Family':
                        result['skills'] = v
                        counterDict['skills'] += 1
                    # elif k == 'Hours':
                    #     result['hours'] = v
                    #     counterDict['hours'] += 1


                start_time = datetime.now()
                tmpList.append(result)
                end_time = datetime.now()

                if counterDict['json'] % 5000 == 0:
                    log.debug('%d JSON Done' % counterDict['json'])
                    # print('Duration: {}'.format(end_time - start_time))

                    tmpDF = pd.DataFrame(tmpList, columns=columnsName)

                    df = df.append(tmpDF, True)
                    # print('head: ====== %s' % df.head(1))
                    tmpList = []
                    sys.stdout.flush()

    # Flush the last batch into dataframe
    log.info('Number of Opp JSON processed: %d' % counterDict['json'])
    # print('Duration: {}'.format(end_time - start_time))
    tmpDF = pd.DataFrame(tmpList, columns=columnsName)
    df = df.append(tmpDF)
    tmpList = []
    sys.stdout.flush()

    df = df.drop(df[df.role_specifics.str.len() < 100].index)

    log.info('Writing Opp CSV to: %s' %dest_csv)
    df.to_csv(dest_csv, index=False)
    log.info('Number of Opp: %d' %df.shape[0])
    log.info('Number of Unique Opp: %d' % len(df['opp_id'].unique()))

    print(counterDict)
    for k, v in keysDict.items():
        print('%s: %d' %(k, v))
    sys.stdout.flush()

def readOppJSON2CSV_200(destFilename):

    data_source_dir = '/backup/vX/apps_opps/opp/200'
    dest_csv = destFilename
    log.info('Reading Client 200 Opp JSON from: %s' %data_source_dir)
    # Read the following fields from Application JSON
    # 1. opp_id
    # 2. state
    # 3. title
    # 4. country
    # 5. location
    # 6. role_type
    # 7. role_specifics
    # 8. skills
    # 9. others
    # 10. json source file name
    # into a csv

    # Create a Pandas DF to store the data
    columnsName = ['opp_id', 'opp_state', 'title', 'country', 'location', 'salary', 'hours', 'department', 'contract_type',
                   'role_type', 'role_specifics', 'skills', 'others', 'opp_source_json']
    df = pd.DataFrame(columns=columnsName)

    counterDict = defaultdict(int)
    keysDict = {}
    tmpList = []
    # Scan the entire dir for .json file
    for root, dirs, files in os.walk(os.path.expanduser(data_source_dir)):
        for f in files:
            fname = os.path.join(root, f)

            if not fname.endswith(".json"):
                continue
            with open(fname, encoding='utf8') as js:
                result = {}
                data = js.read()
                j_data = json.loads(data)

                result['opp_state'] = j_data['state']
                result['title'] = j_data['title']
                result['opp_id'] = j_data['id']
                result['opp_source_json'] = fname
                result['country'] = ''
                result['location'] = ''
                result['role_type'] = ''
                result['role_specifics'] = ''
                result['skills'] = ''
                result['others'] = ''
                result['salary'] = -1
                result['hours'] = -1
                result['department'] = ''
                result['contract_type'] = ''

                # print(fname)
                counterDict['json'] += 1

                buf = __QnA_generator(j_data)

                for k, v in buf.items():
                    keysDict[k] = 1
                    if k == 'Country':
                        result['country'] = v
                        counterDict['country'] += 1
                    elif k == 'Location':
                        result['location'] = v
                        counterDict['location'] += 1
                    elif k == 'Role type':
                        result['role_type'] = v
                        counterDict['role_type'] += 1
                    elif k == 'Role Specifics':
                        result['role_specifics'] = v
                        counterDict['role_specifics'] += 1
                    elif k == 'Essential skills / behaviours' or k == 'Essential skills/behaviours':
                        result['skills'] = v
                        counterDict['skills'] += 1
                    elif k == 'Other Information':
                        result['others'] = v
                        counterDict['others'] += 1
                    elif k == 'Hours':
                        result['hours'] = v
                        counterDict['hours'] += 1
                    elif k == 'Department':
                        result['department'] = v
                        counterDict['department'] += 1
                    elif k == 'Salary':
                        result['salary'] = v
                        counterDict['salary'] += 1
                    elif k == 'Contract type':
                        result['contract_type'] = v
                        counterDict['contract_type'] += 1


                start_time = datetime.now()
                tmpList.append(result)
                end_time = datetime.now()

                if counterDict['json'] % 5000 == 0:
                    log.debug('%d JSON Done' % counterDict['json'])
                    # print('Duration: {}'.format(end_time - start_time))

                    tmpDF = pd.DataFrame(tmpList, columns=columnsName)

                    df = df.append(tmpDF, True)
                    # print('head: ====== %s' % df.head(1))
                    tmpList = []
                    sys.stdout.flush()

    # Flush the last batch into dataframe
    log.info('Number of Opp JSON processed: %d' % counterDict['json'])
    # print('Duration: {}'.format(end_time - start_time))
    tmpDF = pd.DataFrame(tmpList, columns=columnsName)
    df = df.append(tmpDF)
    tmpList = []
    sys.stdout.flush()

    log.info('Writing Opp CSV to: %s' %dest_csv)
    df.to_csv(dest_csv, index=False)
    log.info('Number of Opp: %d' %df.shape[0])
    log.info('Number of Unique Opp: %d' % len(df['opp_id'].unique()))

    # print(counterDict)
    # for k, v in keysDict.items():
    #     print(k)
    # sys.stdout.flush()

def fuseOppType2OppCSV(oppTypeFilename, opp_sourceFilename, opp_destFilename):
    log.info('### Fusing Opportunity Type to opp CSV ###')
    oppType_df = pd.read_csv(oppTypeFilename)
    opp_df = pd.read_csv(opp_sourceFilename)

    result_df = pd.merge(opp_df, oppType_df, 'left', left_on=['opp_id'], right_on=['opp_id'])
    result_df.to_csv(opp_destFilename, index=False)

def dropNonVacanciesOpp(opp_sourceFilename, opp_destFilename):
    df = pd.read_csv(opp_sourceFilename)

    not_vacancy = df[df['opp_type']!='Vacancy'].index
    log.info('Number of opp with opp_type != vacancy: %d' %len(not_vacancy))
    log.info('Before dropping non vacancy opp: %d' %df.shape[0])
    df.drop(not_vacancy, inplace=True)
    log.info('After dropping non vacancy opp: %d' %df.shape[0])
    df.to_csv(opp_destFilename, index=False)

    log.debug('Number of NAN role_specifics: %d' % df['role_specifics'].isnull().sum())

def __QnA_generator(dic):
    result = {}
    for k, v in dic.items():
        if k == 'question':
            result[v] = dic['answer']
        elif isinstance(v, dict):
            result.update(__QnA_generator(v))
        elif isinstance(v, list):
            for i in v:
                if isinstance(i, dict):
                    result.update(__QnA_generator(i))
    return result

cleanr = re.compile('\[.+?\]')
def __tagRemoval(str):
    # print(str)
    cleantext = re.sub(cleanr, '', str)
    # print(cleantext)
    return cleantext

def regenerateCandidateID(input_app_filename, output_app_filename):
    '''
    Regenerate candidate id based on first, last name and email
    Effect: job seeker with multiple candidate id will be combined using one of their old id
    :param input_app_filename:
    :param output_app_filename:
    :return:
    '''
    log.info('==== Regenerate candidate id based on first, last name and email.')
    log.info('Effect: job seeker with multiple candidate id will be combined using a new id')

    df = pd.read_csv(input_app_filename)
    log.info('Before Number of application in CSV: %d' % df.shape[0])
    log.info('Before Number of unique candidate id in CSV: %d' % len(df['candidate_id'].unique()))
    log.info('Before Number of NAN candidate id in CSV: %d' % df['candidate_id'].isnull().sum())
    df['candidate_id_original'] = df['candidate_id']
    groupby_name_email = df.groupby([df['first_name'].str.lower(),
                                df['last_name'].str.lower(),
                                df['email'].str.lower()])['candidate_id'].apply(lambda x: len(x.unique()))

    duplicate_index = groupby_name_email[groupby_name_email >= 2].index
    log.info('Found %d job seeker with multiple account.' %len(duplicate_index))

    for i in range(len(duplicate_index)):
        tmp = df[(df['first_name'].str.lower() == duplicate_index[i][0]) &
                       (df['last_name'].str.lower() == duplicate_index[i][1]) &
                       (df['email'].str.lower() == duplicate_index[i][2])]

        # Use one of the original candidate id
        c_id = tmp.iloc[0]['candidate_id']
        index2 = tmp.index
        for j in index2:
            df.set_value(j, 'candidate_id', c_id)

    log.info('After Number of application in CSV: %d' %df.shape[0])
    log.info('After Number of unique candidate id in CSV: %d' % len(df['candidate_id'].unique()))
    log.info('After Number of NAN candidate id in CSV: %d' % df['candidate_id'].isnull().sum())

    log.info('Writing result to: %s'%output_app_filename)
    df.to_csv(output_app_filename, index=False)

def translate_App_State(input_app_filename, translation_csv_filename, output_app_filename):
    '''
    Application JSON's state translation
    Translate State string into applied, interviewed and offered using Jack's translation CSV
    :param input_app_filename:
    :param translation_csv_filename:
    :param output_app_filename:
    :return:
    '''
    log.info("===== Application JSON's state translation")
    log.info("Translate State string into applied, interviewed and offered using Jack's translation CSV")

    df = pd.read_csv(input_app_filename)
    translation_df = pd.read_csv(translation_csv_filename)
    # print(translation_df.info())

    unique_app_state = df['app_state'].unique()
    # translate = translation_df['candidate_label'].unique()
    translate = translation_df['recruiter_label'].unique()


    translation_dict = {}
    for index, row in translation_df.iterrows():
        if str(row['Event Pocess state - so not applicable to first round or offer']).lower() == 'y':
            translation_dict[row.recruiter_label] = Enumeration.App_State.Event
            # print(row.recruiter_label)
        elif str(row['Offer ']).lower() == 'y':
            translation_dict[row.recruiter_label] = Enumeration.App_State.Offered
        elif str(row['First round invite']).lower() == 'y':
            translation_dict[row.recruiter_label] = Enumeration.App_State.Interviewed
        elif str(row['First round invite']).lower() == 'n':
            translation_dict[row.recruiter_label] = Enumeration.App_State.Applied
        else:
            translation_dict[row.recruiter_label] = Enumeration.App_State.Unknown


    # print(translation_dict)
    counter_dict = defaultdict(int)
    converted_result = []
    for index, row in df.iterrows():
        if row.app_state in translation_dict:
            converted_result.append(translation_dict[row.app_state])
            if row.app_state == 'Application withdrawn':
                counter_dict['Of -1 there are x Application withdrawn'] += 1
            counter_dict[translation_dict[row.app_state]] += 1
        else:
            log.error('Let WCN know that "%s" state is missing. :D' %row.app_state)

    df.rename(columns={'app_state':'app_state_original'}, inplace=True)
    df['app_state'] = converted_result
    log.info(counter_dict)
    # print(df[['app_state', 'app_state_original']].head(50))
    df.to_csv(output_app_filename, index=False)

def dropEventAndUnknown(source_filename, dest_filename):
    '''
    Drop Event and Unknown
    :param source_filename:
    :param dest_filename:
    :return:
    '''
    log.info('=== Drop all Event and unknown rows. ')
    df = pd.read_csv(source_filename)
    # print(df.head(1))
    # Drop Unknown
    unknown = df[df['app_state']==Enumeration.App_State.Unknown].index
    log.info('Number of app with state==unknown: %d' %len(unknown))
    log.info('Before dropping unknown: %d' %df.shape[0])
    df.drop(unknown, inplace=True)
    log.info('After dropping unknown: %d' %df.shape[0])

    # Drop Event
    event = df[df['app_state'] == Enumeration.App_State.Event].index
    if len(event) > 0:
        df.drop(event, inplace=True)
        # assert(False)

    # Drop app with invalid CV. Invalid == file not found or smaller than 100 bytes or no CV
    # Drop empty CV first
    log.info('Number of app without CV: %d' %df['cv name'].isnull().sum())
    log.info('Before dropping without CV: %d' % df.shape[0])
    df = df.dropna(subset=['cv name'])
    log.info('After dropping without CV: %d' % df.shape[0])

    count = 0
    dropIndex = []
    for index, row in df.iterrows():
        cvname = str(row['cv name'])

        fname = Utilities.convertCVName2FolderStructure(cvname)

        try:
            with open(fname, encoding='utf8') as cv:
                data = cv.read()
        except UnicodeDecodeError:
            try:
                with open(fname, encoding='ISO-8859-1') as cv:
                    data = cv.read()
            except Exception:
                print(Exception)
        except FileNotFoundError:
            # log.debug('  CV %s not found error: %s' %(cvname, fname))
            dropIndex.append(index)
            count += 1
            continue

        if len(data) < 500:
            # log.debug('  CV length is too short. Assumed invalid: %s' %fname)
            dropIndex.append(index)
            count += 1
        else:
            if Utilities.isEnglish(Utilities.CVTokeniser(data)):
                pass
            else:
                dropIndex.append(index)
                count += 1

    log.info('Number of missing CV in dir and invalid size CV: %d' %count)
    log.info('Before dropping missing or non english CV: %d' % df.shape[0])
    df.drop(dropIndex, inplace=True)
    log.info('After dropping missing or non english CV: %d' % df.shape[0])

    df.to_csv(dest_filename, index=False)

def dropMultipleApplication(source_filename, dest_filename):
    '''
    Some candidates apply multiple times to the same opp.
    Keep the one the got the candidate the furthest or latest and drop the rest
    :param source_filename:
    :param dest_filename:
    :return: None
    '''
    log.info('=== Drop duplicate application to the same Opp by the same candidate.')
    df = pd.read_csv(source_filename)
    len_original = df.shape[0]
    df['submitted_date'] = pd.to_datetime(df.submitted_date)
    groupby_df = df.groupby(['candidate_id', 'opp_id'])['opp_id'].count()

    # print(groupby_df[groupby_df >= 2])

    for index, value in groupby_df.iteritems():

        if value > 1:
            # print(index)
            sub_df = df[(df['candidate_id']==index[0]) & (df['opp_id']==index[1])]
            sub_df = sub_df.sort_values(by='submitted_date', ascending=False)
            # print(sub_df[['opp_id', 'candidate_id', 'app_state', 'app_state_original', 'submitted_date']])
            # print('Length of sub_df: %d' %len(sub_df))
            # print('Max of state: %d' %sub_df.app_state.max())
            # print('Max of state IDX: %d' % sub_df.app_state.idxmax())
            # print('All Index: %s' %sub_df.index.values)
            drop_index = np.setdiff1d(sub_df.index.values, sub_df.app_state.idxmax())
            # print('Drop index: %s' %drop_index)
            df.drop(drop_index, inplace=True)

    len_after = df.shape[0]

    log.info('Original DF len: %d' % len_original)
    log.info('After drop DF len: %d' % len_after)
    log.info('Dropped: %d' % (len_original-len_after))

    df.to_csv(dest_filename, index=False)

def removeApplicationWithInvalidOpp_id(source_app_filename, source_opp_filename, dest_filename):
    log.info('Removing Application with missing Opportunity information.')

    app_df = pd.read_csv(source_app_filename)
    opp_df = pd.read_csv(source_opp_filename)

    log.info('Number of application in App CSV: %d' % app_df.shape[0])
    log.info('Number of opportunity in Opp CSV: %d' % opp_df.shape[0])

    # Get the list of unique opp_in in opp_df
    unique_opp = opp_df['opp_id'].unique()
    # print(unique_opp[:50])

    all_indexes = app_df.index
    valid_indexes = app_df[app_df['opp_id'].isin(unique_opp)].index
    invalid_indexes = np.setdiff1d(all_indexes, valid_indexes)
    log.info('Number of application with Opp_id in opp.csv: %d' % len(valid_indexes))
    log.info('Number of application with Opp_id in opp.csv: %d' % len(invalid_indexes))

    # Drop the application with opp_id not found in opp.csv
    log.info('Original app_DF len: %d' % app_df.shape[0])
    app_df.drop(invalid_indexes, inplace=True)
    log.info('After drop invalid_indexes len: %d' % app_df.shape[0])

    log.info('Writing app_df to csv: %s' % dest_filename)
    app_df.to_csv(dest_filename, index=False)

def createOppStartEndDateColumn(source_app_filename, source_opp_filename, dest_filename):
    log.info('Estimating start and end date for an opportunity.')

    app_df = pd.read_csv(source_app_filename)
    opp_df = pd.read_csv(source_opp_filename)

    result_firstDate = []
    result_lastDate = []
    empty_opp = [] # Opp_id that no application was ever received
    for index, row in opp_df.iterrows():
        # Get all the application submission date for that opp_id
        date_list = list(app_df[app_df['opp_id']==row['opp_id']].submitted_date.values)

        if len(date_list) > 0:
            # Parse it to a datetime object
            for date in date_list:
                date = parse(date)

            # Sort it in ascending order
            date_list = sorted(date_list)

            result_firstDate.append(date_list[0])
            result_lastDate.append(date_list[len(date_list)-1])
        else:
            result_firstDate.append('')
            result_lastDate.append('')
            empty_opp.append(index)

    opp_df['first_app_date'] = result_firstDate
    opp_df['last_app_date'] = result_lastDate
    log.info('Length of result_firstDate: %d' % len(result_firstDate))

    # log.info('Found number of opportunity with no application: %d' % len(empty_opp))
    # log.info('Before dropping opportunity with no application: %d' % opp_df.shape[0])
    # opp_df = opp_df.drop(empty_opp)
    # log.info('After dropping opportunity with no application: %d' % opp_df.shape[0])

    log.info('Writing opp_df to csv: %s' % dest_filename)
    opp_df.to_csv(dest_filename, index=False)


def create_app_similarTo_doc2vec(source_app_filename='data/app_200.csv',
                             doc2vecFilename='model/doc2vec_trainedvocab_200.d2v',
                             dest_filename='data/app_200.csv'):
    '''
    Add column similar_to_doc2vec to app cv
    Top 100 similar candidate are stored
    :param source_app_filename:
    :param doc2vecFilename:
    :param dest_filename:
    :return:
    '''

    app_df = pd.read_csv(source_app_filename)

    # Load the doc2vec model
    doc2vecmodel = Doc2Vec.load(doc2vecFilename)

    cv_candidateID_dict = {row['cv name']: row['candidate_id'] for index, row in app_df.iterrows()}
    log.info('Number of unique CV: %d' % len(cv_candidateID_dict))

    cv_l = list(cv_candidateID_dict.keys())
    cv_chunks = list(Utilities.chunks(cv_l, 3000))

    # print(cv_l)
    # print(len(cv_chunks))
    # print(len(cv_chunks[0]))

    log.info('Computing doc2vec vectors for all candidate CV. Takes ~5min.')
    # getDoc2VecVector(doc2vecmodel, cv_l)
    doc_vec_result = Parallel(n_jobs=55)(delayed(getDoc2VecVector)(doc2vecmodel, chunk)
                                  for chunk in cv_chunks)

    log.info('Length of results should be same as unique CV: %d' % len(doc_vec_result))
    # Variable to store the result
    cv_vec_dict = {k: v for item in doc_vec_result for k, v in item.items()}
    log.info('Length of cv_vec_dict should be same as unique CV: %d' % len(cv_vec_dict))

    vector_list = list(cv_vec_dict.values())
    cv_list = list(cv_vec_dict.keys())

    log.info('Computing cosine similarity.')

    # Algo 1
    # result = cosine_similarity(vector_list, vector_list)
    # log.debug('Length of cosine result: %d' % len(result))
    # log.debug('Size of cosine result: %d' % result.nbytes)
    #
    # log.debug('Sorting similar candidate based on their doc2vec score')
    # count = 0
    # similarity_dict = {}
    # for arr in result:
    #     sort2 = np.argsort(arr)[::-1]
    #
    #     similarity_dict[cv_list[count]] = [(cv_candidateID_dict[cv_list[i]], arr[i]) for
    #                                        i in sort2[1:51]]
    #     count += 1

    # # Algo 2
    # BLOCK_SIZE = 5000
    # vector_chunks = list(Utilities.chunks(vector_list, BLOCK_SIZE))
    # similarity_dict = {}
    # count = 0
    # for chuck_index in range(len(vector_chunks)):
    #     log.debug('Processing chunk %d of %d' %(chuck_index, len(vector_chunks)))
    #
    #     cosine_result = cosine_similarity(vector_chunks[chuck_index], vector_list)
    #
    #     sort = np.argsort(-cosine_result, axis=1)
    #     for arr in sort:
    #         similarity_dict[cv_list[count]] = [
    #             (cv_candidateID_dict[cv_list[i]], cosine_result[count%BLOCK_SIZE][i]) for i in arr[1:100]]
    #         count += 1

    # Algo 3
    BLOCK_SIZE = 500
    topN = 100
    vector_chunks = list(Utilities.chunks(vector_list, BLOCK_SIZE))



    sorted_result = Parallel(n_jobs=15)(delayed(sortTopNSimilar)
                                        (BLOCK_SIZE * chuck_index,
                                         vector_chunks[chuck_index],
                                         vector_list, BLOCK_SIZE,
                                         cv_list,
                                         cv_candidateID_dict)
                                        for chuck_index in range(len(vector_chunks)))

    # similarity_dict = {} #cv name: array of [(candidate_id, similarity score)]
    similarity_dict = {k: v for item in sorted_result for k, v in item.items()}


    log.info('sort complete ')
    #################################################

    similarColumnValue = []
    doc2vecColumnValue = []
    for index, row in app_df.iterrows():
        if row['cv name'] in similarity_dict:
            similarColumnValue.append(similarity_dict[row['cv name']])
            # print(similarity_dict[row['cv name']])
        else:
            similarColumnValue.append([])

        if row['cv name'] in cv_vec_dict:
            # doc2vecColumnValue.append(cv_vec_dict[row['cv name']])
            vec = list(cv_vec_dict[row['cv name']])
            doc2vecColumnValue.append(vec)
        else:
            doc2vecColumnValue.append([])

    app_df['app_similar_to_doc2vec'] = similarColumnValue
    app_df['app_doc2vec'] = doc2vecColumnValue
    log.info('Length of newColumnValue: %d ' % len(similarColumnValue))
    log.info('Length of app_df: %d' % app_df.shape[0])

    print(app_df[['candidate_id', 'app_similar_to_doc2vec']].head(2))

    app_df.to_csv(dest_filename, index=False)

def sortTopNSimilar(count_pos, v_chuck, vector_list, BLOCK_SIZE, cv_list, cv_candidateID_dict):
    topN = 100
    log.debug('Processing chunk %d' % (count_pos/BLOCK_SIZE))
    log.debug('count_pos: %d' % count_pos)
    cosine_result = cosine_similarity(v_chuck, vector_list)

    # Sort the array and return the index in descending order
    sort = np.argsort(-cosine_result, axis=1)

    tmp_similarity_dict = {}
    for arr in sort:
        tmp_similarity_dict[cv_list[count_pos]] = \
            [(cv_candidateID_dict[cv_list[i]], cosine_result[count_pos % BLOCK_SIZE][i])
             for i in arr[1:topN]]
        count_pos += 1
    return tmp_similarity_dict

def getDoc2VecVector(doc2vecmodel, chunk):
    log.debug('Inferring vector for number of CV: %d' % len(chunk))
    cv_vec_dict2 = {}
    for key in chunk:
        fname = Utilities.convertCVName2FolderStructure(key, includeBaseFolder=True)

        try:
            with open(fname, encoding='utf8') as cv:
                data = cv.read()

        except UnicodeDecodeError:
            try:
                with open(fname, encoding='ISO-8859-1') as cv:
                    data = cv.read()
            except Exception:
                log.error(Exception)
        except FileNotFoundError:
            log.error('  CV not found error: %s' % fname)
            return

        if len(data) > 499:
            vector = doc2vecmodel.infer_vector(Utilities.processCV(data),
                                               alpha=0.025, min_alpha=0.001, steps=15)
            cv_vec_dict2[key] = vector
        else:
            log.error('CV shorter than 499bytes: %s' % key)

    return cv_vec_dict2


def create_opp_similarTo_doc2vec(source_opp_filename, doc2vecFilename, dest_filename):
    '''
    Add column opp_similar_to_doc2vec to opp
    Top 100 similar opp are stored
    :param source_opp_filename:
    :param doc2vecFilename:
    :param dest_filename:
    :return:
    '''

    opp_df = pd.read_csv(source_opp_filename)

    # Load the doc2vec model
    doc2vecmodel = Doc2Vec.load(doc2vecFilename)

    opp_df['content'] = opp_df['title'].map(str) + " " + \
                        opp_df['role_specifics'].map(str) + " " +\
                        opp_df['skills'].map(str)

    log.info('Inferring vector for each opportunity. ')
    opp_vec_dict = {}
    for index, row in opp_df.iterrows():
        opp_vec_dict[row.opp_id] = doc2vecmodel.infer_vector(Utilities.processOpp(row['content']),
                                                             alpha=0.025, min_alpha=0.001, steps=15)
    # Remove the content column
    opp_df = opp_df.drop('content', axis=1)

    vec_list = list(opp_vec_dict.values())
    opp_list = list(opp_vec_dict.keys())

    result = cosine_similarity(vec_list, vec_list)
    log.debug('Length of cosine result: %d' % len(result))
    log.debug('Size of cosine result: %d' % result.nbytes)

    log.info('Sorting similar opp based on their doc2vec score')
    count = 0
    similarity_dict = {} # {opp_id: [(opp_id, similar_score), ...]}
    for arr in result:
        sort2 = np.argsort(arr)[::-1]

        similarity_dict[opp_list[count]] = [(opp_list[i], arr[i]) for i in sort2[1:401]]
        count += 1
    # print(similarity_dict)

    log.info('sort complete ')

    #################################################

    similarColumnValue = []
    doc2VecColumn = []
    for index, row in opp_df.iterrows():
        if row['opp_id'] in similarity_dict:
            similarColumnValue.append(similarity_dict[row['opp_id']])
        else:
            similarColumnValue.append([])

        if row['opp_id'] in opp_vec_dict:
            # doc2VecColumn.append(np.array2string(opp_vec_dict[row['opp_id']]).replace('\n', ''))
            vec = list(opp_vec_dict[row['opp_id']])
            doc2VecColumn.append(vec)
        else:
            doc2VecColumn.append([])

    opp_df['opp_similar_to_doc2vec'] = similarColumnValue
    opp_df['opp_doc2vec'] = doc2VecColumn
    log.info('Length of newColumnValue: %d ' % len(similarColumnValue))
    log.info('Length of opp_df: %d' % opp_df.shape[0])

    # print(opp_df[['opp_id', 'opp_similar_to_doc2vec']].head(2))

    opp_df.to_csv(dest_filename, index=False)

def createCVwordcountColumn(source_app_filename, dest_filename):
    app_df = pd.read_csv(source_app_filename)

    cv_length = []
    for index, row in app_df.iterrows():
        cvname = str(row['cv name'])

        fname = Utilities.convertCVName2FolderStructure(cvname)

        try:
            with open(fname, encoding='utf8') as cv:
                data = cv.read()
        except UnicodeDecodeError:
            try:
                with open(fname, encoding='ISO-8859-1') as cv:
                    data = cv.read()
            except Exception:
                print(Exception)
        except FileNotFoundError:
            log.debug('  CV %s not found error: %s' %(cvname, fname))
            cv_length.append(0)
            continue

        cv_length.append(len(data))
        # if len(data) < 500:
        #     # log.debug('  CV length is too short. Assumed invalid: %s' %fname)
        #     dropIndex.append(index)
        #     count += 1
        # else:
        #     if Utilities.isEnglish(Utilities.CVTokeniser(data)):
        #         pass
        #     else:
        #         dropIndex.append(index)
        #         count += 1

    if len(cv_length) == app_df.shape[0]:
        app_df['cv_len'] = cv_length
        log.info('Writing to file: %s' %dest_filename)
        app_df.to_csv(dest_filename, index=False)

def createCV_CSV(source_app_filename, dest_filename):
    log.info('Creating %s ' %dest_filename)
    app_df = pd.read_csv(source_app_filename)

    # cv_data_list = []
    # cv_name_list = []
    cv_dict = {}
    log.debug('Reading in all CV data')
    for index, row in app_df.iterrows():
        # cv_name_list.append(row['cv name'])
        cvname = str(row['cv name'])

        fname = Utilities.convertCVName2FolderStructure(cvname)

        try:
            with open(fname, encoding='utf8') as cv:
                data = cv.read()
        except UnicodeDecodeError:
            try:
                with open(fname, encoding='ISO-8859-1') as cv:
                    data = cv.read()
            except Exception:
                print(Exception)
        except FileNotFoundError:
            log.debug('  CV %s not found error: %s' % (cvname, fname))
            # cv_data_list.append('')
            continue

        # cv_data_list.append(Utilities.processCV(data))
        cv_dict[row['cv name']] = data

    log.info("Processing CV data")
    cv_Documents = Parallel(n_jobs=50)(delayed(processCV)(key, data) for key, data in cv_dict.items())
    cv_Documents = {k: v for item in cv_Documents for k, v in item.items()}

    log.debug('Length of cv data list: %d' %len(cv_Documents))
    log.debug('Length of app_df: %d' % app_df.shape[0])
    df = pd.DataFrame()
    if len(cv_Documents) == len(app_df['cv name'].unique()):
        df['cv name'] = list(cv_Documents.keys())
        df['cv_data'] = list(cv_Documents.values())
        log.info('Writing to file: %s' %dest_filename)
        df.to_csv(dest_filename, index=False)

def processCV(key, cv_data):
    return {key:Utilities.processCV(cv_data)}

def main():
    # WCN files
    report_filename = '/backup/vX/apps_opps/rep/report_200_live.csv'
    cv_filename = '/backup/vX/CV/200/files_200.csv'

    # Output filename
    app_filename = 'data/app_200_demo.csv'
    opp_filename = 'data/opp_200_demo.csv'

    # # Read Opportunity JSON fields into CSV
    # This is customised specially for Client 200
    readOppJSON2CSV_200(opp_filename)
    # This is customised specially for Client 048
    # readOppJSON2CSV_048(opp_filename)
    # This is customised specially for Client 422
    # readOppJSON2CSV_422(opp_filename)
    # This is customised specially for Client 331
    # readOppJSON2CSV_331(opp_filename)
    #
    # # Fuse Opp type to the CSV so that we know if it is a event, vacancy, talentbank,...
    fuseOppType2OppCSV(oppTypeFilename='/backup/vX/opp_types/opp_type_200_live.csv',
                       opp_sourceFilename=opp_filename,
                       opp_destFilename=opp_filename)

    dropNonVacanciesOpp(opp_sourceFilename=opp_filename, opp_destFilename=opp_filename)


    ############################################################################
    ############################################################################


    # # Read application JSON into CSV
    # # Tested for Client 200, 32, 35, 48, 51, 422
    # readApplicationJSON2CSV('/backup/vX/apps_opps/app/048', 'data/app_048_raw.csv')
    readApplicationJSON2CSV('/backup/vX/apps_opps/app/200', 'data/app_200_raw_demo.csv')
    # readApplicationJSON2CSV('/backup/vX/apps_opps/app/422', 'data/app_422_raw.csv')
    # readApplicationJSON2CSV('/backup/vX/apps_opps/app/331', 'data/app_331_raw.csv')
    #
    # Join application csv with Richard's csv that he got from the system
    # This function should be generic for all clients
    joinAppWithCVandReport('data/app_200_raw_demo.csv',
                           report_filename,
                           cv_filename,
                           app_filename)

    # Application JSON's state translation
    # Translate State string into applied, interviewed and offered using Jack's translation CSV
    translate_App_State(app_filename, 'data/AllStatuses.csv', app_filename)

    # Drop all event. Drop application with state==unknown. Drop application with invalid and non english CV
    dropEventAndUnknown(app_filename, app_filename)

    # Regenerate candidate id based on first, last name and email
    # Effect job seeker with multiple candidate id will be combined using a new id
    regenerateCandidateID(app_filename, app_filename)

    # Check for candidate that applies multiple times to the same opp
    dropMultipleApplication(app_filename, app_filename)

    # Remove application in app.csv which opp_id not in Opp.csv
    removeApplicationWithInvalidOpp_id(source_app_filename=app_filename,
                                       source_opp_filename=opp_filename,
                                       dest_filename=app_filename)

    # Estimate the start and end date for an opportunity using application information
    createOppStartEndDateColumn(source_app_filename=app_filename,
                                source_opp_filename=opp_filename,
                                dest_filename=opp_filename)

    # # Adds CV word count into app.csv
    # createCVwordcountColumn(source_app_filename='data/app_331_doc2vec_AllCV_Opp_1_nostem.csv',
    #                         dest_filename='data/app_331_doc2vec_AllCV_Opp_1_nostem_1.csv')

    # Train the Doc2Vec model for the client
    Doc2VecModelScript.createDoc2Vec_AllCV_Opp_Model_Parallel(opp_filename=opp_filename,
                                           CVFolderName='/u01/bigdata/vX/CV/CV/200',
                                           doc2vec_model_name='model/doc2vec_200_demo.d2v')

    # Create similar_to column in app.csv based on doc2vec
    create_app_similarTo_doc2vec(source_app_filename=app_filename,
                                 doc2vecFilename='model/doc2vec_200_demo.d2v',
                                 dest_filename='data/app_200_doc2vec_demo.csv')

    # Create opp_similar_to column in opp.csv based on doc2vec
    create_opp_similarTo_doc2vec(source_opp_filename=opp_filename,
                                 doc2vecFilename='model/doc2vec_200_demo.d2v',
                                 dest_filename='data/opp_200_doc2vec_demo.csv')

    # Create a cv.csv for each client to preprocess all cv into a single CSV instead of reading of the directory
    createCV_CSV(source_app_filename='data/app_200_doc2vec_demo.csv', dest_filename='data/cv_200_demo.csv')

if __name__ == "__main__":
    main()
