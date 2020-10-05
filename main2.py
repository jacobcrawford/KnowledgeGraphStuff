import logging
import sys
from os import listdir
from os.path import isfile, join

import pandas as pd
from GLIMPSE_personalized_KGsummarization.src.base import DBPedia
from GLIMPSE_personalized_KGsummarization.src.glimpse import GLIMPSE



logging.basicConfig(format='[%(asctime)s] - %(message)s',
                    level=logging.DEBUG)





def loadDBPedia(path):

    print("loading from: " + path)
    KG = DBPedia(rdf_gz=path)
    # Load the KG into memory
    logging.info('Loading {}'.format(KG.name()))
    KG.load()
    logging.info('Loaded {}'.format(KG.name()))
    return KG

path = "user_query_log_answers/"
user_log_answer_files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".csv")]
number_of_users = len(user_log_answer_files)


user_log_train = []
user_log_test = []
user_answers = []

for file in user_log_answer_files:

    df = pd.read_csv(path+str(file))
    # list of lists of answers as iris
    user_answers.append([ ["<" +iri+">" for iri in f.split(" ")] for f in df['answers']])




path = sys.argv[1]
KG = loadDBPedia(path)

logging.info("KG entities: " +str(len(KG.entity_id_)))
logging.info("KG triples: " +str(KG.number_of_triples_))

for i in range(number_of_users):
    logging.info("user answers:" + str(len(user_answers[i])))
    # Split log in 70%
    split_index_train = int(len(user_answers[i])*0.7)

    # collapse to one list of entities
    user_log_train.append([f for c in user_answers[i][:split_index_train] for f in c if KG.is_entity(f)])
    user_log_test.append([f for c in user_answers[i][split_index_train:] for f in c if KG.is_entity(f)])





# model user pref
logging.info("Running GLIMPSE")
summary = GLIMPSE(KG,10000,user_log_train, 1e-2)
logging.info("done")

entities_test = len(user_log_test)
count = 0
for iri in user_log_test:
    if summary.has_entity(iri):
        count +=1

logging.info("Summary contained" + str(count) + "/" + str(entities_test) + " :" + str(count/entities_test) + "%")

