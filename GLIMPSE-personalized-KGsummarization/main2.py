import logging
import sys

import pandas as pd

logging.basicConfig(format='[%(asctime)s] - %(message)s',
                    level=logging.DEBUG)


from .src.base import DBPedia


def loadDBPedia(path):

    print("loading from: " + path)
    KG = DBPedia(rdf_gz=path)
    # Load the KG into memory
    logging.info('Loading {}'.format(KG.name()))
    KG.load()
    logging.info('Loaded {}'.format(KG.name()))
    return KG


df = pd.read_csv("../user_query_log_answers.csv")
# list of lists of answers as iris
answers = [f.split(" ") for f in df['answers']]

path = sys.argv[1]
KG = loadDBPedia(path)

print("checking " + "http://dbpedia.org/resource/Artur_%C5%BBmijewski_(actor)")
print(KG.is_entity("http://dbpedia.org/resource/Artur_%C5%BBmijewski_(actor)"))

for i in answers[0][:10]:
    print("checking iri" + i)
    print(KG.is_entity(i))

