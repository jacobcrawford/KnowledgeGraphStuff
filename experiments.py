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

from GLIMPSE_personalized_KGsummarization.src.algorithms import query_vector, random_walk_with_restart
from GLIMPSE_personalized_KGsummarization.src.base import DBPedia
from GLIMPSE_personalized_KGsummarization.src.glimpse import GLIMPSE

logging.basicConfig(format='[%(asctime)s] - %(message)s',
                    level=logging.DEBUG)

def float_in_zero_one(value):
    """Check if a float value is in [0, 1]"""
    value = float(value)
    if value < 0 or value > 1:
        raise argparse.ArgumentTypeError('Value must be a float between 0 and 1')
    return value

def loadDBPedia(path):
    print("loading from: " + path)
    KG = DBPedia(rdf_gz=path)
    # Load the KG into memory
    logging.info('Loading {}'.format(KG.name()))
    KG.load()
    logging.info('Loaded {}'.format(KG.name()))
    return KG

def printResults(version):
    path1 = "experiments_results"
    path2 = "experiments_results_pagerank"

    version_string = version if version is not None else ""

    ppr2 = [f for f in listdir(path2) if isfile(join(path2, f)) and f.endswith(".csv") and "PPR#2" in f and version_string in f]
    ppr5 = [f for f in listdir(path2) if isfile(join(path2, f)) and f.endswith(".csv") and "PPR#5" in f and version_string in f]
    glimpse1 = [f for f in listdir(path1) if isfile(join(path1, f)) and f.endswith(".csv") and "e#0.01" in f and version_string in f]
    glimpse2 = [f for f in listdir(path1) if isfile(join(path1, f)) and f.endswith(".csv") and "e#0.001" in f and version_string in f]

    rows = []
    div = 10000
    tms = 1000000
    print("PPR2")
    def pctf(x):
        f = math.ceil(int(x)/47408000*tms)/div
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
    K = [10*(10**-i) for i in range(2, 7)]
    E = [1e-2]

    logging.info("KG entities: " +str(len(KG.entity_id_)))
    logging.info("KG triples: " +str(KG.number_of_triples_))

    for i in range(number_of_users):
        print("User log queries: " + str(len(user_answers[i])))
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
        for e in E:
            logging.info("  Running for e=" + str(e))
            rows = []
            for idx_u in range(number_of_users):
                KG.reset()
                # model user pref
                logging.info("      Running GLIMPSE on user: " + user_ids[idx_u])
                t1 = time.time()
                summary = GLIMPSE(KG, k, user_log_train[idx_u], e)
                logging.info("      Done")
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
                logging.info("      Summary  accuracy " + str(mean_accuracy) + "%")
                rows.append({'match': total_count, 'total': total_entities, '%':mean_accuracy , 'runtime': t2-t1 })

            pd.DataFrame(rows).to_csv("experiments_results/v"+version+ "T#" +str(KG.number_of_triples())+"_E#"+str(KG.number_of_entities()) +"K#"+str(int(k))+"e#"+str(e)+ ".csv")

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



def runGLIMPSEExperimentOnce(k, e,version, answers_version, kg_path):
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

    k = k*KG.number_of_triples()

    logging.info("KG entities: " + str(len(KG.number_of_entities())))
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

def pageRankExperimentOnce(k,ppr,version,answers_version, kg_path):
    logging.info("Starting ppr"+str(ppr)+" for k="+ str(k))
    KG = loadDBPedia(kg_path)
    path = "user_query_log_answers" + answers_version + "/"
    user_log_answer_files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".csv")]
    number_of_users = len(user_log_answer_files)

    k = k*KG.number_of_triples()

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
        summary = [KG.id_entity(i) for i in indexes]

        mean_accuracy, total_entities,total_count = calculateAccuracyAndTotals(user_log_test[idx_u], summary)
        logging.info("      Summary  accuracy " + str(mean_accuracy) + "%")
        rows.append({'match': total_count, 'total': total_entities, '%': mean_accuracy, 'runtime': t2 - t1})
    pd.DataFrame(rows).to_csv(
        "experiments_results_pagerank/v" + version + "T#" + str(KG.number_of_triples()) + "_E#" + str(
            KG.number_of_entities()) + "_K#" + str(int(k)) + "_PPR#" + str(ppr) + ".csv")
    logging.info("Done")


def f1skew(fn):
    return (2/(1 + fn))/(1+(1/(1+fn)))

METHODS = {
    'glimpse',
    'ppr'
}

VERSIONS = {
    '2': 'More users',
    '3': 'Normalizing query vector'
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
    k = args.percent_triples
    version = args.version
    answer_version = args.version_answers
    kg_path = args.KG_path

    if args.method == 'glimpse':
        e = args.epsilon
        runGLIMPSEExperimentOnce(k,e,version, answer_version, kg_path)
    elif args.method == 'ppr':
        ppr = int(args.walk)
        pageRankExperimentOnce(k,ppr,version,answer_version, kg_path )
    else:
        logging.info("running nothing. method parameter not set")


if __name__ == '__main__':
    main()
