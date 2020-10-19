K = [10*(10**-i) for i in range(2, 7)]
E = [1e-2]
import os
import logging

logging.basicConfig(format='[%(asctime)s] - %(message)s',
                    level=logging.DEBUG)

answer_version = "2"
version = "3"

if (False):
    for k in K:
        for e in E:
            os.system('nohup python experiments.py'
                      + ' --method glimpse'
                      + ' --percent-triples ' + str(k)
                      + ' --version ' + version
                      + ' --version-answers ' + answer_version
                      + ' --epsilon ' + str(e)
                      + ' > ' + 'GLIMPSE' + version+'a' + answer_version + ' #K' + str(k) + '#E' + str(e) + '.out'
                      + ' &')

for k in K:
    for ppr in [2,5]:
        call = 'nohup python experiments.py'\
               + ' --method ppr'\
               + ' --walk '+str(ppr)\
               + ' --percent-triples ' + str(k)\
               + ' --version ' + version\
               + ' --version-answers ' + answer_version\
               + ' > ' + 'PPR'+str(ppr) + version+'a' + answer_version + '#K' + str(k) + '.out'\
               + ' &'
        logging.info(call)
        os.system(call)

