from os import listdir
from os.path import isfile, join
import subprocess

import pandas as pd

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

def queryDBPediaResourceAnalysis(df:pd.DataFrame):
    marker = "<http://dbpedia.org/"
    counts = {}
    for q in df['query']:
        split = q.split(marker)
        if len(split) > 1:
            sub_key = split[1]
            for sub_key in split:
                rest = sub_key.split(">")[0]
                #copy = (rest + '.')[:-1] # copy
                #for r in rest[::-1]: #reverse
                #    if r == "/":
                #        break
                #    else:
                #        copy = copy[len(copy)-1]

                key = marker + rest + ">"

                if counts.get(key) is None:
                    counts[key] = 1
                else:
                    counts[key] = counts[key] + 1
    return counts



#df = df_unique_id[df_unique_id['count'] < 500]
#df = df[df['count'] > 200]

#print("Users with log size < 500 and > 200: " + str(len(df)))


def getFileContainingSlow(iri:str):
    path = "./DBPedia3.9Full"
    files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".nt")]
    for f in files:
        with open(path + "/" +f) as file:
            if iri in file.read():
                print(f)
                break
    print("no file contains iri" + str(iri))

def getFileContaining(iri:str):
    path = "./DBPedia3.9Full"
    files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".nt")]
    for f in files:
        print(f)
        subprocess.call(["grep -m 1",iri , path + "/" +f])
    print("no file contains iri" + str(iri))


