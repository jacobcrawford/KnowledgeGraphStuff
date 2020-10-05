import logging
import sys

logging.basicConfig(format='[%(asctime)s] - %(message)s',
                    level=logging.DEBUG)

from src.base import DBPedia





def loadDBPedia():
    path = sys.argv[1]
    print("loading from: " + path)
    KG = DBPedia(rdf_gz=path)
    # Load the KG into memory
    logging.info('Loading {}'.format(KG.name()))
    KG.load()
    logging.info('Loaded {}'.format(KG.name()))
    return KG

loadDBPedia()