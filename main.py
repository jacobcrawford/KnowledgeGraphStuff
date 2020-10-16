import subprocess
import math
import time
from rdflib import URIRef, Literal
import urllib.request
from os import listdir
from os.path import isfile, join
import json
import re

import numpy as np
import pandas as pd
import requests
from SPARQLWrapper import SPARQLWrapper, RDF, N3, RDFXML
from bs4 import BeautifulSoup

from GLIMPSE_personalized_KGsummarization.src.algorithms import query_vector, random_walk_with_restart
from main2 import loadDBPedia
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
    df = df[df['answers'] >= 15]
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

        pd.DataFrame(rows).to_csv("user_query_log_answers2/" + uid + ".csv")

def extractAnswersToQueryInRDF():
    def removeOptional(query):
        o_idx = query.lower().find("optional")
        rem_o_end = query[o_idx:len(query)].find("}")
        result = query[0: o_idx] + query[o_idx + rem_o_end: len(query)]
        print(result)
        if "optional" in result.lower():
            print("MORE")
            return removeOptional(result)
        return result

    def makeConstructQuery(q):

        select_idx = q.lower().find("select")
        # remove select
        first = q[:select_idx]
        # extract where
        brac_start_idx = q.find('{')
        brac_end_idx = len(q)  - q[::-1].find('}') - 1
        where  = q[brac_start_idx:brac_end_idx + 1]
        if "filter" in where.lower():
            f = where.lower().find("filter")
            where_c = where[0:f] + "}"
        else:
            where_c = where
        if "optional" in where.lower():
            where_c = removeOptional(where_c)
        return first + "CONSTRUCT " + where_c + "WHERE" + where + q[brac_end_idx + 2: len(q)+1]

    def extractTriples(results_rdf_lib):
        results_triple = []
        for triple in results_rdf_lib:
            e1 = triple[0]
            r = triple[1]
            e2 = triple[2]
            # Skip triples with property values
            if type(e1) == Literal or type(e2) == Literal:
                break
            result = str(e1) + " " + str(r) + " " + str(e2)
            results_triple.append(result)
        return results_triple

    a = "SELECT ?name ?description_en ?description_de ?musician WHERE { \
         ?musician <http://purl.org/dc/terms/subject> <http://dbpedia.org/resource/Category:German_musicians> .\
         ?musician foaf:name ?name .\
         OPTIONAL {\
             ?musician rdfs:comment ?description_en .\
             FILTER (LANG(?description_en) = 'en') .\
         }\
         OPTIONAL {\
             ?musician rdfs:comment ?description_de .\
             FILTER (LANG(?description_de) = 'de') .\
         }\
       }\
        "

    user_list = makeQueryLogsUserList()
    df = pd.read_csv("user_stats2.csv")
    df = df[df['answers'] >= 20]
    v = VirtuosoConnector(format=RDFXML)
    v1 = VirtuosoConnector()
    for uid in df['uid']:
        rows = []
        i = 0
        success = 0
        for q in user_list[uid]:
            try:
                results = v1.query(q)
                if len(results) == 0:
                    continue
                else:
                    success += 1
            except:
                continue
            try:
                qc = makeConstructQuery(q)
                results = v.query(qc)

                if len(results) == 0:
                    continue
                results = extractTriples(results)
                if len(results) != 0:
                    rows.append({'id': i, 'answers': " ".join(results)})
                    i += 1
            except Exception as e:
                print("ERROR")
                print(e)
                print("\n####################")
                print(q)
                print("&&&&&&&&&&&&&&&&&&&")
                print(qc)
                print("####################\n")
        print("Number of success queries:" + str(success))
        print("Queries with results:" + str(len(rows)))
        pd.DataFrame(rows).to_csv("user_query_log_answersRDF/" + uid + ".csv")


def pageRankExperiment(path):
    KG = loadDBPedia(path)
    version = "2"
    path = "user_query_log_answers" + version + "/"
    user_log_answer_files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".csv")]
    number_of_users = len(user_log_answer_files)

    user_log_train = []
    user_log_test = []
    user_answers = []

    for i in range(number_of_users):

        for file in user_log_answer_files:
            df = pd.read_csv(path + str(file))
            # list of lists of answers as iris
            user_answers.append([["<" + iri + ">" for iri in f.split(" ")] for f in df['answers']])

        # Split log in 70%
        split_index_train = int(len(user_answers[i]) * 0.7)

        # collapse to one list of entities
        user_log_train.append([f for c in user_answers[i][:split_index_train] for f in c if KG.is_entity(f)])
        user_log_test.append([f for c in user_answers[i][split_index_train:] for f in c if KG.is_entity(f)])

    K = [10*(10**-i)*KG.number_of_triples() for i in range(2, 7)]

    for ppr in [2,5]:
        for k in K :
            rows = []
            for i in range(number_of_users):
                t1 = time.time()
                qv = query_vector(KG, user_log_train[i])
                M = KG.transition_matrix()
                ppr_v = random_walk_with_restart(M,qv,0.15,ppr)

                t2 = time.time()

                # Extract k indexes
                indexes = np.argpartition(ppr_v,-k)[-k:]
                summary = [KG.id_entity(i) for i in indexes]

                count = 0
                total = len(user_log_test[i])
                for iri in user_log_test[i]:
                    if iri in summary:
                        count += 1
                rows.append({'match': count, 'total': total, '%': count / total, 'runtime': t2 - t1})
            pd.DataFrame(rows).to_csv("experiments_results_pagerank/v" +version+ "T#" + str(KG.number_of_triples()) + "_E#" + str(KG.number_of_entities()) + "_K#" + str(k) +"_PPR#" + str(ppr)+ ".csv")

def printResults():
    path1 = "experiments_results"
    path2 = "experiments_results_pagerank"

    ppr2 = [f for f in listdir(path2) if isfile(join(path2, f)) and f.endswith(".csv") and "PPR#2" in f]
    ppr5 = [f for f in listdir(path2) if isfile(join(path2, f)) and f.endswith(".csv") and "PPR#5" in f]
    glimpse1 = [f for f in listdir(path1) if isfile(join(path1, f)) and f.endswith(".csv") and "e#0.01" in f]
    glimpse2 = [f for f in listdir(path1) if isfile(join(path1, f)) and f.endswith(".csv") and "e#0.001" in f]

    rows = []
    div = 10000
    tms = 1000000
    print("PPR2")
    def pctf(x):
        f = math.ceil(int(x)/47408000*tms)/div
        print("F:" + str(f), str(type(f)))
        if f > 1:
            print(f)
            return str(math.floor(f))
        f = str(f)
        return f[0:f.find("1")+1]


    for p in ppr2:
        df = pd.read_csv(path2+ "/"+p)
        k = str(p.split("K#")[1].split("_PPR")[0])
        print("k = " +k )
        pct = pctf(k)
        acc = df['%'].sum()/len(df['%'])
        print(acc)
        rows.append({'Accuracy':str(acc), 'Algorithm':"ppr2", 'K in % of |T|': pct})

    print("\nPPR5")

    for p in ppr5:
        df = pd.read_csv(path2+ "/"+p)
        k = str(p.split("K#")[1].split("_PPR")[0])
        pct = pctf(k)
        print("k = " + k)
        acc = df['%'].sum() / len(df['%'])
        print(acc)
        rows.append({'Accuracy': str(acc), 'Algorithm': "ppr5", 'K in % of |T|': pct})

    print("\nGLIMPSE e=0.01")
    for p in glimpse1:
        df = pd.read_csv(path1 + "/" + p)
        k = str(p.split("K#")[1].split("e#")[0])
        pct = pctf(k)
        print("k = " + k)
        acc = df['%'].sum() / len(df['%'])
        print(acc)
        rows.append({'Accuracy': str(acc), 'Algorithm': "glimpse-2", 'K in % of |T|': pct})

    print("\nGLIMPSE e=0.001")
    for p in glimpse2:
        df = pd.read_csv(path1 + "/" + p)
        k = str(p.split("K#")[1].split("e#")[0])
        pct = pctf(k)
        print("k = " + k)
        acc = df['%'].sum() / len(df['%'])
        print(acc)
        rows.append({'Accuracy': str(acc), 'Algorithm': "glimpse-3", 'K in % of |T|': pct})


    print(json.dumps(rows))

def f1skew(fn):
    return (2/(1 + fn))/(1+(1/(1+fn)))

path = sys.argv[1]
pageRankExperiment(path)
#runGLIMPSEExperiment()
#printResults()
#extractAnswersToQuery()
#print(f1skew(1))