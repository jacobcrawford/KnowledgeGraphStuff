import logging
import sys
import pandas as pd
from GLIMPSE_personalized_KGsummarization.src.base import DBPedia



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
answers = [f.split(" ") for f in df['answers']]

path = sys.argv[1]
KG = loadDBPedia(path)

print("checking " + "http://dbpedia.org/resource/Artur_%C5%BBmijewski_(actor)")
print(KG.is_entity("http://dbpedia.org/resource/Artur_%C5%BBmijewski_(actor)"))


print("KG entities: " +str(len(KG.entity_id_)))
print("KG triples: " +str(KG.number_of_triples_))
print("entities head")

for i in range(10):
    print(KG.id_entity(i))
print("\n")


for i in answers[0][:10]:
    print("checking iri" + i)
    print(KG.is_entity(i))

