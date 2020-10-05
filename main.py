import subprocess
import sys
import urllib.request
from os import listdir
from os.path import isfile, join

import pandas as pd
import requests
from bs4 import BeautifulSoup

# mypath = "dbpedia3.9"
# parsed_logs = QueryLogReader.parseDirectoryOfLogs(mypath)
# df = pd.DataFrame(parsed_logs)
# d = queryDBPediaResourceAnalysis(df)
from virtuoso_connector import makeQueryLogsUserList, VirtuosoConnector

def listFD(url, ext=''):
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]

def downloadSelectedFiles(path_to_dest):
    url = 'http://downloads.dbpedia.org/3.9/en/'
    ext = 'nt.bz2'
    print('Beginning file download with urllib2...')
    selected = ["long_abstracts_en.nt.bz2"]
    not_selected = ["article_categories_en.nt.bz2", "category_labels_en.nt.bz2",
                    "geo_coordinates_en.nt.bz2", "instance_types_en.nt.bz2"
                    , "mappingbased_properties_en.nt.bz2",
                    "persondata_en.nt.bz2", "topical_concepts_en.nt.bz2" ]
    for file in listFD(url, ext):
        name = file.split("http://downloads.dbpedia.org/3.9/en//")[1]
        if name in selected:
            filepath = path_to_dest + "/" + name
            urllib.request.urlretrieve(file, filepath)
            print("Downloaded:" + str(name) + " at " + str(filepath))


def downLoadDBPedia39():
    url = 'http://downloads.dbpedia.org/3.9/en/'
    ext = 'nt.bz2'
    print('Beginning file download with urllib2...')
    download = False
    for file in listFD(url, ext):
        name = str(file)
        print(name)
        if name == "http://downloads.dbpedia.org/3.9/en//old_interlanguage_links_see_also_chapters_en.nt.bz2":
            print("Resume download")
            download = True
        if download:
            name = file.split("http://downloads.dbpedia.org/3.9/en//")[1]

            urllib.request.urlretrieve(file, './DBPedia3.9Full/' + name)
            print("Downloaded")

def makeUniqueIriFiles():
    path = "./DBPedia3.9Full"
    outPath ="./DBPedia3.9Unique"
    files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".nt")]
    print(files)
    for f in files:
        subprocess.run("grep -Eo '<h.*?>' " + path + "/" + f + " | sort -u >> " + outPath + "/" + f, shell=True)
        print("done " + f)


def collapseIriNamespace():
    df = pd.read_csv('query_log_iri_count.csv')
    iris = df['iri']
    iri_namespace = {}
    for iri in iris:
        i = iri.replace(">", "")
        i_split = i.split("/")
        word = i_split[0]
        for w in i_split[1:len(i_split)-1]:
            word = word + "/" + w
        try:
            iri_namespace[word] += 1
        except KeyError:
            iri_namespace[word] = 1
    rows = []
    for k in iri_namespace.keys():
        rows.append({'iri_collapsed': k, 'count': iri_namespace[k]})
    df = pd.DataFrame(rows)
    df.to_csv("query_log_iri_count_collapsed.csv")

def main():
    """
    Extract the user query stats from the local virtuoso endpoint
    :return:
    """
    v = VirtuosoConnector()

    logs = makeQueryLogsUserList()
    blacklisted = ['2ca6b561c0a7e3be4c32b1f2204c8615', '86f4a8d9c625e093028dbf54d901aefa',
                   "b2314d110688b9244c22d960337c9063"  # sameAs
        , "8b71b56bf98141fb230fde6109a07fd4"  # lang=de
        , "bda7fb9f06bf208401a026552c455b82"  # only thumbnails
        , "f1fb84c74d59777730f97824f79ecbb7"  # too long execution
        , "a51b23ef9030c78767fafff62b613a28"  # too few non error queries
        , "da46b5512d83ebc25b0822603e7641a7"  # too few non error queries
        , "a2e9e8a829b9361d4e1dd77c4823e610"  # too many sameAs queries
        , "7d5ad4664ea38873983588d3f16a770b"  # obvious auto agent
        , "21ab73a224ae312b7973478914d69b21"  # to few non error queries (33)
        , "baf9f0e0700f08ac66d6f9c845903b0c"  # obvious auto agent
        , "66b94515e6dc5ae4e2f3f2784a76e227"  # too few non error queries
                   ]

    rows = []
    for k in logs.keys():
        if k in blacklisted:
            continue
        print("uid " + str(k))
        answers, errors, nr_q, no_ans = v.extractAnswersToUsersQueryLog(logs[k])
        row = {"uid": str(k), "answers": len(answers), "errors": errors, "queries": nr_q, "no answer": no_ans}
        print(row)
        rows.append(row)
    pd.DataFrame(rows).to_csv("user_stats2.csv")

def main2():
    user_list = makeQueryLogsUserList()
    for k in user_list.keys():
        rows = []
        for q in user_list[k]:
            rows.append({'query': q})
        df = pd.DataFrame(rows)
        df.to_csv("user_query_logs/" + k + ".csv")

def main3(key):
    v = VirtuosoConnector()
    v2 = VirtuosoConnector("https://dbpedia.org/sparql", "http://dbpedia.org")
    logs = makeQueryLogsUserList()
    for k in [key]:
        for q in logs[k]:
            try:
                print("\n New:")
                results = v.extractIRIsFromJsonResults(v.query(q))
                print("local: " + str(len(results)))
                results2 = v2.extractIRIsFromJsonResults(v2.query(q))
                print("server: " + str(len(results2)))
                if len(results) == 0 and len(results2) != 0:
                    print(q)
            except:
                print("error")

def extractAnswersToQuery():
    user_list = makeQueryLogsUserList()
    df = pd.read_csv("user_stats2.csv")
    df = df[df['answers'] >= 40]
    v = VirtuosoConnector()
    for uid in df['uid']:
        rows = []
        i = 0
        for q in user_list[uid]:
            try:
                results = v.query(q)
                results = v.extractIRIsFromJsonResults(results)
                results = [r for r in results if "http" in r]
                if len(results)>0:
                    # Filter out non iri answers
                    rows.append({'id': i, 'answers': " ".join(results)})
                i += 1
            except Exception as e:
                print(e)

        pd.DataFrame(rows).to_csv("user_query_log_answers/" + uid + ".csv")




#analyseIRIUse(df)
#getFileContaining("<http://www.w3.org/2004/02/skos/core")
#makeUniqueIriFiles()
#analyseIRIUse()
#analyseFilesForIriUse()
#collapseIriNamespace()
#extractAnswersToQuery()

path_to_dest = sys.argv[1]
downloadSelectedFiles(path_to_dest)