import logging
import sys
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


df = pd.read_csv("user_query_log_answers/6d418da8de1b4e19787dc71797f22003.csv")
# list of lists of answers as iris
user_answers = [ ["<" +iri+">" for iri in f.split(" ")] for f in df['answers']]

path = sys.argv[1]
KG = loadDBPedia(path)

logging.info("KG entities: " +str(len(KG.entity_id_)))
logging.info("KG triples: " +str(KG.number_of_triples_))

# Split log in 70%
split_index_train = int(len(user_answers)*0.7)

# collapse to one list of entities
user_log_train = [f for f in user_answers[:split_index_train] if KG.is_entity(f)]
user_log_test = [f for f in user_answers[split_index_train:] if KG.is_entity(f)]

# model user pref
logging.info("Running GLIMPSE")
summary = GLIMPSE(KG,10000,user_log_train)
logging.info("done")

entities_test = len(user_log_test)
count = 0
for iri in user_log_test:
    if summary.has_entity(iri):
        count +=1

logging.info("Summary contained" + str(count) + "/" + str(entities_test) + " :" + str(count/entities_test) + "%")

