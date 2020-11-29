import json
import math
import sys
import time
from os import listdir
from os.path import isfile, join
import logging
import argparse

import numpy as np
import pandas as pd

from GLIMPSE_personalized_KGsummarization.src.algorithms import query_vector, random_walk_with_restart, query_vector_rdf
from GLIMPSE_personalized_KGsummarization.src.base import DBPedia, KnowledgeGraph
from GLIMPSE_personalized_KGsummarization.src.glimpse import GLIMPSE, Summary

logging.basicConfig(format='[%(asctime)s] - %(message)s',
                    level=logging.DEBUG)

def float_in_zero_one(value):
    """Check if a float value is in [0, 1]"""
    value = float(value)
    if value < 0 or value > 1:
        raise argparse.ArgumentTypeError('Value must be a float between 0 and 1')
    return value

def summaryAccuracy(summary, user_log):
    total_count = 0
    total_entities = 0

    accuracies = []

    for answers_to_query in user_log:
        count = 0
        total_answers = len(answers_to_query)
        if total_answers == 0:
            continue
        else:
            total_entities += total_answers
            for iri in answers_to_query:
                if summary.has_entity(iri):
                    count += 1
                    total_count += 1
            accuracies.append(count / total_answers)
    return np.mean(np.array(accuracies))

def loadDBPedia(path, include_properties=False):
    print("loading from: " + path)
    KG = DBPedia(rdf_gz=path,include_properties=include_properties)
    # Load the KG into memory
    logging.info('Loading {}'.format(KG.name()))
    KG.load()
    logging.info('Loaded {}'.format(KG.name()))
    return KG

def printResults(version, use_etf=False, key='K in % of |T|', include_properties=False):
    path1 = "experiments_results"
    path2 = "experiments_results_pagerank"

    etf = {"474":1.89094736842, "4740":1.66558391338, "47408":1.3411616641, "474080":0.703619142439, "4740806":0.479221758377}
    version_string = version if version is not None else ""

    ppr2 = [f for f in listdir(path2) if isfile(join(path2, f)) and f.endswith(".csv") and "PPR#2" in f and version_string in f]
    ppr5 = [f for f in listdir(path2) if isfile(join(path2, f)) and f.endswith(".csv") and "PPR#5" in f and version_string in f]
    glimpse1 = [f for f in listdir(path1) if isfile(join(path1, f)) and f.endswith(".csv") and "e#1e-05" in f and version_string in f]
    glimpse2 = [f for f in listdir(path1) if isfile(join(path1, f)) and f.endswith(".csv") and "e#0.001" in f and version_string in f]

    rows = []
    div = 10000
    tms = 1000000
    kg_triples = 69213000 if include_properties else 47408000

    def pctf(x):
        f = math.ceil(int(x)/kg_triples*tms)/div
        if f > 1:
            print(f)
            return str(math.floor(f))
        f = str(f)
        return f[0:f.find("1")+1]

    for p in ppr2:
        df = pd.read_csv(path2+ "/"+p)
        k = str(p.split("K#")[1].split("_PPR")[0])
        value = pctf(k) if not use_etf else int(k)
        acc = df['%'].sum()/len(df['%'])
        rows.append({'Accuracy':str(acc), 'Algorithm':"ppr2", key: value})

    print("\nPPR5")

    for p in ppr5:
        if etf: break
        df = pd.read_csv(path2+ "/"+p)
        k = str(p.split("K#")[1].split("_PPR")[0])
        value = pctf(k)
        acc = df['%'].sum() / len(df['%'])
        rows.append({'Accuracy': str(acc), 'Algorithm': "ppr5", key: value})

    print("\nGLIMPSE e=0.01")
    for p in glimpse1:
        df = pd.read_csv(path1 + "/" + p)
        k = str(p.split("K#")[1].split("e#")[0])
        value = pctf(k) if not use_etf else int(etf[k]*int(k))
        acc = df['%'].sum() / len(df['%'])
        rows.append({'Accuracy': str(acc), 'Algorithm': "glimpse-5 prob", key: value})

    print("\nGLIMPSE e=0.001")
    for p in glimpse2:
        if etf: break
        df = pd.read_csv(path1 + "/" + p)
        k = str(p.split("K#")[1].split("e#")[0])
        value = pctf(k)
        #df_exclude = df.index.isin(exclude)
        #df = df[~df_exclude]
        acc = df['%'].sum() / len(df['%'])
        rows.append({'Accuracy': str(acc), 'Algorithm': "glimpse-3", key: value})
    print(json.dumps(rows))


def pageRankExperiment(path):
    KG = loadDBPedia(path)
    version = "2"
    answers_version = "2"
    path = "user_query_log_answers" + answers_version + "/"
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
            pd.DataFrame(rows).to_csv("experiments_results_pagerank/v" +version+ "T#" + str(KG.number_of_triples()) + "_E#" + str(KG.number_of_entities()) + "_K#" + str(int(k)) +"_PPR#" + str(ppr)+ ".csv")


def runGLIMPSEExperiment():
    version = "3"
    answers_version = "2"
    path = "user_query_log_answers"+answers_version+"/"
    user_log_answer_files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".csv")]
    number_of_users = len(user_log_answer_files)

    user_log_train = []
    user_log_test = []
    user_answers = []
    user_ids = []

    for file in user_log_answer_files:

        df = pd.read_csv(path+str(file))
        user_ids.append(file.split(".csv")[0])
        # list of lists of answers as iris
        user_answers.append([ ["<" +iri+">" for iri in f.split(" ")] for f in df['answers']])

    path = sys.argv[1]
    KG = loadDBPedia(path)
    K = [10*(10**-i)*KG.number_of_triples() for i in range(2, 7)]
    E = [1e-2]

    logging.info("KG entities: " +str(len(KG.entity_id_)))
    logging.info("KG triples: " +str(KG.number_of_triples_))

    for i in range(number_of_users):
        #print("User log queries: " + str(len(user_answers[i])))
        # Split log in 70%
        split_index_train = int(len(user_answers[i]) * 0.7)

        # TODO make a list of lists where each list is the answers to one query
        user_log_train.append([[entity for entity in answers_to_query if KG.is_entity(entity)]
                               for answers_to_query in user_answers[i][:split_index_train] ])
        user_log_test.append([[entity for entity in answers_to_query if KG.is_entity(entity)]
                               for answers_to_query in user_answers[i][split_index_train:] ])

        logging.info("user answers:" + str(len(user_log_train[i])+len(user_log_test[i])))

    for k in K:
        logging.info("Running for K=" + str(k))
        entities_triple_factor = []
        for e in E:
            logging.info("  Running for e=" + str(e))
            rows = []

            for idx_u in range(number_of_users):
                KG.reset()
                # model user pref
                #logging.info("      Running GLIMPSE on user: " + user_ids[idx_u])
                t1 = time.time()
                summary = GLIMPSE(KG, k, user_log_train[idx_u], e)
                logging.info("Ent/tri: " + str(summary.number_of_entities()) + "/" + str(summary.number_of_triples()))
                entities_triple_factor.append(summary.number_of_entities()/summary.number_of_triples())
                #logging.info("      Done")
                t2 = time.time()
                total_count = 0
                total_entities = 0

                accuracies = []

                for answers_to_query in user_log_test[idx_u]:
                    count = 0
                    total_answers = len(answers_to_query)
                    if total_answers == 0:
                        continue
                    else:
                        total_entities += total_answers
                        for iri in answers_to_query:
                            if summary.has_entity(iri):
                                count += 1
                                total_count +=1
                        accuracies.append(count/total_answers)

                mean_accuracy = np.mean(np.array(accuracies))
                #logging.info("      Summary  accuracy " + str(mean_accuracy) + "%")
                rows.append({'match': total_count, 'total': total_entities, '%':mean_accuracy , 'runtime': t2-t1 })
            logging.info("Finish for k: " + str(k))
            logging.info("Mean entities: " + str(np.mean(np.array(entities_triple_factor))))
            #pd.DataFrame(rows).to_csv("experiments_results/v"+version+ "T#" +str(KG.number_of_triples())+"_E#"+str(KG.number_of_entities()) +"K#"+str(int(k))+"e#"+str(e)+ ".csv")


def makeTrainingAndTestData(number_of_users, user_answers, KG):
    user_log_train = []
    user_log_test = []
    for i in range(number_of_users):
        # Split log in 70%
        split_index_train = int(len(user_answers[i]) * 0.7)

        user_log_train.append([[entity for entity in answers_to_query if KG.is_entity(entity)]
                               for answers_to_query in user_answers[i][:split_index_train]])
        user_log_test.append([[entity for entity in answers_to_query if KG.is_entity(entity)]
                              for answers_to_query in user_answers[i][split_index_train:]])
    return user_log_train, user_log_test


def calculateAccuracyAndTotals(user_log_test_u, summary):
    accuracies = []
    total_count = 0
    total_entities = 0
    for answers_to_query in user_log_test_u:
        count = 0
        total_answers = len(answers_to_query)
        if total_answers == 0:
            continue
        else:
            total_entities += total_answers
            for iri in answers_to_query:
                if summary.has_entity(iri):
                    count += 1
                    total_count += 1
            accuracies.append(count / total_answers)

    return np.mean(np.array(accuracies)), total_entities, total_count


def runGLIMPSEExperimentOnce(k_pct, e,version, answers_version, kg_path):
    """
    Run one instance of the
    :param k: k in pct of T
    :param e: sampling parameter
    :param version: Version to use in output file
    :param answers_version: Version of answers to use. Most recent is 2
    :param kg_path: Path to the folder of the DBPedia3.9 files
    :return:
    """
    path = "user_query_log_answers" + answers_version + "/"
    user_log_answer_files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".csv")]
    number_of_users = len(user_log_answer_files)

    KG = loadDBPedia(kg_path)

    user_answers = []
    user_ids = []

    for file in user_log_answer_files:
        df = pd.read_csv(path + str(file))
        user_ids.append(file.split(".csv")[0])
        # list of lists of answers as iris
        user_answers.append([["<" + iri + ">" for iri in f.split(" ")] for f in df['answers']])

    user_log_train, user_log_test = makeTrainingAndTestData(number_of_users,user_answers, KG)

    k = k_pct*KG.number_of_triples()

    logging.info("KG entities: " + str(KG.number_of_entities()))
    logging.info("KG triples: " + str(KG.number_of_triples_))

    logging.info("Running for K=" + str(k) + ", e=" + str(e))
    rows = []
    for idx_u in range(number_of_users):
        KG.reset()
        # model user pref
        logging.info("  Running GLIMPSE on user: " + user_ids[idx_u])
        t1 = time.time()
        summary = GLIMPSE(KG, k, user_log_train[idx_u], e)
        logging.info("  Done")
        t2 = time.time()

        mean_accuracy, total_entities,total_count = calculateAccuracyAndTotals(user_log_test[idx_u], summary)

        logging.info("      Summary  accuracy " + str(mean_accuracy) + "%")
        rows.append({'match': total_count, 'total': total_entities, '%': mean_accuracy, 'runtime': t2 - t1})

    pd.DataFrame(rows).to_csv("experiments_results/v" + version + "T#" + str(KG.number_of_triples()) + "_E#" + str(
        KG.number_of_entities()) + "K#" + str(int(k)) + "e#" + str(e) + ".csv")

def runGLIMPSEDynamicExperiment(k_pct, e,version, answers_version, kg_path, split, retrain=False, only_last=False):
    """
    Run dynamic experiments of glimpse
    :param k_pct: k in pct of T
    :param e: Sampling parameter
    :param version: Version of experiment. Output file depends on the version
    :param answers_version: Version of answer to use. Last i 2
    :param kg_path: Path to the folder of the DBPedia3.9 files
    :param split: number in [0,1]. Decides the size of one split.
    :param retrain: Weather or not to retrain
    :param only_last: Weather or not to retrain only on last interval.
    :return: Output file to "experiments_results"
    """
    path = "user_query_log_answers" + answers_version + "/"
    user_log_answer_files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".csv")]


    KG = loadDBPedia(kg_path)
    k = k_pct * KG.number_of_triples()

    logging.info("KG entities: " + str(KG.number_of_entities()))
    logging.info("KG triples: " + str(KG.number_of_triples_))

    user_answers = []
    user_ids = []

    for file in user_log_answer_files:
        df = pd.read_csv(path + str(file))
        if len(df) > 40:
            user_ids.append(file.split(".csv")[0])
            # list of lists of answers as iris
            user_answers.append([["<" + iri + ">" for iri in f.split(" ")] for f in df['answers']])

    number_of_users = len(user_ids)

    # Split data
    user_data_split = []
    for i in range(int(1/split)): #[0,1,2,3,4]
        user_data_split.append([])
        for j in range(number_of_users): #[0,...,14]
            split_index_start = int(len(user_answers[j]) * (split*i))
            split_index_end = int(len(user_answers[j]) * (split*(i+1)))
            user_data_split[i].append([[entity for entity in answers_to_query if KG.is_entity(entity)]
                               for answers_to_query in user_answers[j][split_index_start:split_index_end]])

    rows = []
    for idx_u in range(number_of_users):
        KG.reset()
        summary = GLIMPSE(KG, k, user_data_split[0][idx_u], e)
        if not retrain:
            rows.append({str(split*i): summaryAccuracy(summary,user_data_split[i][idx_u]) for i in range(1, int(1/split))})
        else:

                res = {}
                for i in range(0,len(user_data_split)-1):
                    train = []
                    if not only_last:
                        for j in range(0,i+1):
                            train = train + user_data_split[j][idx_u]
                    else:
                        train = user_data_split[i][idx_u]
                        logging.info("training only on last interval")
                    test = user_data_split[i+1][idx_u]
                    summary = GLIMPSE(KG, k, train, e)
                    res[str(split * i)] = summaryAccuracy(summary, test)
                rows.append(res)

        logging.info("Finished for user: " + user_ids[idx_u])
    pd.DataFrame(rows).to_csv("experiments_results/v"+str(version)+ "T#" +str(KG.number_of_triples())+"_E#"+str(KG.number_of_entities()) +"K#"+str(int(k))+"e#"+str(e)+"S"+str(split)+ ".csv")

def makeRDFData(user_log_answer_files,path, KG: KnowledgeGraph):
    # filter out logs of size < 10
    answers = []
    uids = []
    for i, file in enumerate(user_log_answer_files):
        user_answers = []

        df = pd.read_csv(path + str(file))
        if len(df) > 10:
            for answer in df['answers']:
                triples = []
                j = 0

                iris = answer.split(" ")
                while j < len(iris):
                    e1, r, e2 = iris[j], iris[j + 1], iris[j + 2]
                    if "http" in e1:
                        e1 = "<" + e1 + ">"
                    if "http" in e2:
                        e2 = "<" + e2 + ">"
                    if "http" in r:
                        r = "<" + r + ">"
                    triple = (e1, r, e2)
                    if KG.has_triple(triple):
                        triples.append(triple)
                    j = j + 3
                user_answers.append(triples)

            # Append answers
            answers.append(user_answers)
            # Append uid
            uids.append(file.split(".")[0])
    return answers,uids


def makeSplitRDF(split_pct,log):
    user_log_train = []
    user_log_test = []
    for idx in range(len(log)):
        split = math.floor(len(log[idx])*split_pct)
        user_log_train.append(log[idx][0:split])
        user_log_test.append(log[idx][split:len(log[idx])])
    return user_log_train, user_log_test


def calculateMeanAccuracyRDF(test_log, summary):
    accuracies = []
    for answer in test_log:
        total_triples = len(answer)
        triples_in_summary = len([triple for triple in answer if summary.has_triple(triple)])

        accuracies.append(triples_in_summary / total_triples)

    return np.mean(np.array(accuracies))

def runGLIMPSEExperimentOnceRDF(k_in_pct, e,version, answers_version, kg_path=None, include_relationship_prob=False, include_properties=False,KG_in=None):
    """

    :param k_in_pct: k in pct of T
    :param e: sampling parameter
    :param version: Version to use in output file
    :param answers_version: Version of answers to use. Most recent is 2
    :param kg_path: Path to the folder of the DBPedia3.9 files
    :param include_relationship_prob:  Include probability of a relationship or not
    :param include_properties: Include properties or not
    :param KG_in: Input KG object to use. If none is given the KG will be loaded.
    :return:
    """
    path = "user_query_log_answers" + answers_version + "/"
    user_log_answer_files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".csv")]
    KG = loadDBPedia(kg_path, include_properties=include_properties) if KG_in is None else KG_in
    answers, uids = makeRDFData(user_log_answer_files,path, KG)

    # Make train and test sets
    user_log_train, user_log_test = makeSplitRDF(0.7, answers)

    k = k_in_pct*KG.number_of_triples()

    logging.info("Running for K=" + str(k) + ", e=" + str(e))
    rows = []
    for idx_u in range(len(uids)):
        KG.reset()
        # model user pref
        logging.info("  Running GLIMPSE on user: " + uids[idx_u])
        t1 = time.time()
        summary = GLIMPSE(KG, k, user_log_train[idx_u], e, 1, True, include_relation_prob=include_relationship_prob)
        logging.info("  Done")
        t2 = time.time()

        mean_accuracy = calculateMeanAccuracyRDF(user_log_test[idx_u], summary)

        logging.info("      Summary  accuracy " + str(mean_accuracy) + "%")
        rows.append({'%': mean_accuracy, 'runtime': t2 - t1,'entities':str(summary.number_of_entities()),'relationships': str(summary.number_of_relationships())})

    pd.DataFrame(rows).to_csv("experiments_results/v" + version + "T#" + str(KG.number_of_triples()) + "_E#" + str(
        KG.number_of_entities()) + "K#" + str(int(k)) + "e#" + str(e) + ".csv")

def runPagerankExperimentOnceRDF(k_in_pct,ppr,version,answers_version, kg_path=None, KG_in=None):
    """

    :param k_in_pct: k in pct of T
    :param ppr: random walk length
    :param version: Version to use in output file
    :param answers_version: Version of answers to use. Most recent is 2
    :param kg_path: Path to the folder of the DBPedia3.9 files
    :param KG_in: Input KG object to use. If none is given the KG will be loaded.
    :return:
    """
    logging.info("Starting ppr" + str(ppr) + " for k=" + str(k_in_pct))
    KG = loadDBPedia(kg_path, include_properties=True) if KG_in is None else KG_in
    path = "user_query_log_answers" + answers_version + "/"
    user_log_answer_files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".csv")]

    k = k_in_pct * KG.number_of_triples()

    answers, uids = makeRDFData(user_log_answer_files, path, KG)

    # Make train and test sets
    user_log_train, user_log_test = makeSplitRDF(0.7, answers)

    rows = []
    for idx_u in range(len(uids)):
        t1 = time.time()

        qv,y = query_vector_rdf(KG, user_log_train[idx_u])
        M = KG.transition_matrix()
        ppr_v = random_walk_with_restart(M, qv, 0.15, ppr)

        # Extract k triples
        summary = Summary(KG)

        argsort = np.flip(np.argsort(ppr_v))

        for entity_id in argsort:
            if summary.number_of_triples() > k:
                break
            e1 = KG.id_entity(entity_id)
            # Entity might not have outgoing edges
            try:
                KG.__getitem__(e1)
            except:
                continue
            for r in KG[e1]:
                for e2 in KG[e1][r]:
                    summary.add_triple((e1, r, e2))
                    if summary.number_of_triples() > k:
                        break
                if summary.number_of_triples() > k:
                    break

        logging.info("number of triples in summary:" + str(summary.number_of_triples()))
        logging.info("number of entities in summary:" + str(summary.number_of_entities()))
        logging.info("number of relations in summary:" + str(summary.number_of_relationships()))
        t2 = time.time()

        mean_accuracy = calculateMeanAccuracyRDF(user_log_test[idx_u], summary)
        logging.info("      Summary  accuracy " + str(mean_accuracy) + "%")
        rows.append({'%': mean_accuracy, 'runtime': t2 - t1,'entities':str(summary.number_of_entities()),'relationships': str(summary.number_of_relationships())})

    pd.DataFrame(rows).to_csv(
        "experiments_results_pagerank/v" + version + "T#" + str(KG.number_of_triples()) + "_E#" + str(
            KG.number_of_entities()) + "_K#" + str(int(k)) + "_PPR#" + str(ppr) + ".csv")
    logging.info("Done")

def pageRankExperimentOnce(k_in_pct,ppr,version,answers_version, kg_path):
    """

    :param k_in_pct: k in pct of T
    :param ppr: random walk length
    :param version: Version to use in output file
    :param answers_version: Version of answers to use. Most recent is 2
    :param kg_path: Path to the folder of the DBPedia3.9 files
    :return:
    """
    logging.info("Starting ppr"+str(ppr)+" for k="+ str(k))
    KG = loadDBPedia(kg_path)
    path = "user_query_log_answers" + answers_version + "/"
    user_log_answer_files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".csv")]
    number_of_users = len(user_log_answer_files)

    k = k_in_pct*KG.number_of_triples()

    user_answers = []
    user_ids = []

    for file in user_log_answer_files:
        df = pd.read_csv(path + str(file))
        user_ids.append(file.split(".csv")[0])
        # list of lists of answers as iris
        user_answers.append([["<" + iri + ">" for iri in f.split(" ")] for f in df['answers']])

    user_log_train, user_log_test = makeTrainingAndTestData(number_of_users,user_answers, KG)

    rows = []
    for idx_u in range(number_of_users):
        t1 = time.time()
        qv = query_vector(KG, user_log_train[idx_u])
        M = KG.transition_matrix()
        ppr_v = random_walk_with_restart(M, qv, 0.15, ppr)

        t2 = time.time()

        # Extract k indexes
        indexes = np.argpartition(ppr_v, -k)[-k:]
        summary = Summary(KG)
        summary.entities_ = set([KG.id_entity(i) for i in indexes])

        mean_accuracy, total_entities,total_count = calculateAccuracyAndTotals(user_log_test[idx_u], summary)
        logging.info("      Summary  accuracy " + str(mean_accuracy) + "%")
        rows.append({'match': total_count, 'total': total_entities, '%': mean_accuracy, 'runtime': t2 - t1})
    pd.DataFrame(rows).to_csv(
        "experiments_results_pagerank/v" + version + "T#" + str(KG.number_of_triples()) + "_E#" + str(
            KG.number_of_entities()) + "_K#" + str(int(k)) + "_PPR#" + str(ppr) + ".csv")
    logging.info("Done")

METHODS = {
    'glimpse',
    'ppr',
}

VERSIONS = {
    '2': 'More users',
    '3': 'Normalizing query vector',
    '4': 'Construct query results',
    '5': 'Dynamic setup',
    '6': 'Construct query with relationship probabilities',
    '7': 'Dynamic retrain', #TODO
    '8': 'Dynamic retrain on last interval',#TODO
    '9': 'prints and logs'
}


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--KG-path', default='../dbpedia3.9/', help='Path to the KG files')
    parser.add_argument('--percent-triples', type=float_in_zero_one, default=0.001,
            help='Ratio of number of triples of KG to use as K '
                 '(summary constraint). Default is 0.001.')
    parser.add_argument('--walk', help="length of ppr walk")
    parser.add_argument('--version', help='version of experiments', choices=list(VERSIONS.keys()))
    parser.add_argument('--version-answers', help='version of extracted user answers')
    parser.add_argument('--epsilon',
            help='Set this flag to the epsilon parameter.Defines the sampling size.', default=1e-2)
    parser.add_argument('--method', default=['glimpse'],
            choices=list(METHODS),
            help='Experiments to run')
    return parser.parse_args()

def main():
    args = parse_args()
    k_in_pct = float(args.percent_triples)
    version = args.version
    answer_version = args.version_answers
    kg_path = args.KG_path

    if args.method == 'glimpse':
        e = float(args.epsilon)
        if 'RDF' in answer_version:
            if version == '6':
                runGLIMPSEExperimentOnceRDF(k_in_pct, e, version, answer_version, kg_path, include_relationship_prob=True)
            else:
                runGLIMPSEExperimentOnceRDF(k_in_pct, e, version, answer_version, kg_path )
        else:
            runGLIMPSEExperimentOnce(k_in_pct,e,version, answer_version, kg_path)

    elif args.method == 'ppr':
        ppr = int(args.walk)
        if 'RDF' in answer_version:
            runPagerankExperimentOnceRDF(k_in_pct, ppr, version, answer_version, kg_path)
        else:
            pageRankExperimentOnce(k_in_pct, ppr, version, answer_version, kg_path)
    else:
        logging.info("running nothing. method parameter not set")

if __name__ == '__main__':
    main()



