import json
import logging
import re
import subprocess
from os import listdir
from os.path import isfile, join, getsize

import numpy as np
import pandas as pd
import QueryLogReader
from GLIMPSE_personalized_KGsummarization.src.base import KnowledgeGraph
from virtuoso_connector import makeQueryLogsUserList

logging.basicConfig(format='[%(asctime)s] - %(message)s',
                    level=logging.DEBUG)


def keyword_analysis(df:pd.DataFrame):
    """
    Do a keyword analysis of the keywords "SELECT", "ASK", "DESCRIBE", "CONSTRUCT"
    :param df: The dataframe where there is a 'query' column
    :return: A dict of the keyword statistics
    """
    # Do keyword analysis
    keywords = ["SELECT", "ASK", "DESCRIBE", "CONSTRUCT", "OPTIONAL", "FILTER"]

    keywords_stats = []

    for keyword in keywords:
        l = len(df[df['query'].str.contains(keyword, case=False)])
        keywords_stats.append({"Keyword": keyword, "Absolute": l, "Relative": str(round(l / len(df) * 100, 2)) + "%"})
    return keywords_stats
    #pd.DataFrame(keywords_stats).to_csv("keywords_stats_" + str(sys.argv[1:]) + ".csv")

def uid_analysis(df: pd.DataFrame):
    """
    Analyse how many unique uids are in the query logs and how many queries each have executed
    :param df: The dataframe with a 'uid' column
    :return: A dataframe with columns 'uid, count'
    """
    # Do log size analysis
    df_unique_id = df['uid'].value_counts().to_frame().reset_index()
    df_unique_id.columns = ["uid","count"]
    return df_unique_id

def getFileContaining(iri:str, path):
    """
    Use grep to search all files in a directory for an iri
    :param iri:
    :param path:
    :return:
    """
    files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".nt")]
    for f in files:
        subprocess.call(["grep","-m","1", iri , path + "/" +f])
    print("no file contains iri" + str(iri))

def analyseIRIUse(path):
    """
    Analyse the use of iris in query logs. Usage is saved in the "query_log_iri_count.csv" file.
    :param path: path to query logs
    :return:
    """
    ### PARSE LOGS ###
    parsed_logs = QueryLogReader.parseDirectoryOfLogs(path)
    user_list = {}

    ### Divide the log into personal logs in dict format {uid: list<string>} ###
    for d in parsed_logs:
        try:
            user_list[d['uid']].append(d['query'])
        except KeyError:
            user_list[d['uid']] = [d['query']]

    ### Get uid analysis ###
    df = pd.DataFrame(parsed_logs)
    df_unique = uid_analysis(df)
    # filter on count
    df_unique = df_unique[df_unique['count'] < 1000]
    df_unique = df_unique[df_unique['count'] > 200]

    uids = [df_unique['uid'].iloc[i] for i in range(len(df_unique))]

    iri_count = {}

    for uid in uids:
        queries = user_list[uid]
        for q in queries:
            if "where" in q.lower():
                split = q.lower().split("where")[1]
                iris = re.findall("<[^\s]*>",split)
                for iri in iris:
                    try:
                        iri_count[iri] +=1
                    except KeyError:
                        iri_count[iri] =1
    rows = []
    for k in iri_count.keys():
        rows.append({'iri': k, 'count': iri_count[k]})
    df = pd.DataFrame(rows)
    df.to_csv("query_log_iri_count.csv")

def analyseAnswersFull(KG: KnowledgeGraph):
    """
    Analyse the unique entities and unique relationships in the query logs
    :param KG:
    :return:
    """
    logging.info("Number of relationships: "+ str(KG.number_of_relationships()))

    path = "user_query_log_answers" + str(2) + "/"
    user_log_answer_files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".csv")]
    number_of_users = len(user_log_answer_files)

    user_answers = []
    user_ids = []

    for file in user_log_answer_files:
        df = pd.read_csv(path + str(file))
        user_ids.append(file.split(".csv")[0])
        # list of lists of answers as iris
        user_answers.append([["<" + iri + ">" for iri in f.split(" ")] for f in df['answers']])

    unique_entity = []
    unique_relation = []

    for idx_u in range(number_of_users):
        user_unique_entity = []
        user_unique_relation = []

        for answers in user_answers[idx_u]:
            for answer in answers:
                if KG.has_entity(answer):
                    user_unique_entity.append(answer)
                elif KG.has_relationship(answer):
                    user_unique_relation.append(answer)
                else:
                    logging.info("NO entity or relation: " + answer)
        unique_entity.append(set(user_unique_entity))
        unique_relation.append(set(user_unique_relation))

    unique_entity_len = [len(l) for l in unique_entity]
    unique_relation_len = [len(l) for l in unique_relation]
    logging.info("unique entities: " + str(unique_entity_len))
    logging.info("unique relations: " + str(unique_relation_len))

    logging.info("unique entities avg: " + str(np.mean(np.array(unique_entity_len))))
    logging.info("unique relations avg: " + str(np.mean(np.array(unique_relation_len))))

def analyseLanguageInExperimentData():
    """
    Analyse the use of the 'lang' keyword and the refered language.
    :return:
    """
    userlist = makeQueryLogsUserList()
    langs_count = {}
    for i in userlist:
        queries = userlist[i]
        for q in queries:
            if 'lang(' in q:
                for lang_string in re.findall('lang\(*.*\)\s*=\s*".*"', q):
                    idx = lang_string.index('"')
                    lang = lang_string[idx + 1: idx + 3]
                    if langs_count.get(lang):
                        langs_count[lang] += 1
                    else:
                        langs_count[lang] = 1
    print(json.dumps([{'lang':k,'count':langs_count[k]} for k in langs_count.keys()]))

def analyseAnswersStats(path):
    """
    Analyse the a query log folder for number of users and the corresponding query count
    :param path:
    :return:
    """
    user_log_answer_files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".csv")]
    number_of_users = len(user_log_answer_files)

    answers_count = []
    for file in user_log_answer_files:
        df = pd.read_csv(path + str(file))
        answers_count.append(len(df['answers']))
    answers_count = np.array(answers_count)
    print("users count: "+ str(number_of_users))
    print("min:" + str(np.min(answers_count)))
    print("max:" + str(np.max(answers_count)))
    print("mean:" + str(np.mean(answers_count)))
    print("median:" +str(np.median(answers_count)))
