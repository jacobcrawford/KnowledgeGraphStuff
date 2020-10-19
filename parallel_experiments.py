K = [10*(10**-i) for i in range(2, 7)]
E = [1e-2]
import os
import logging

logging.basicConfig(format='[%(asctime)s] - %(message)s',
                    level=logging.DEBUG)

answer_version = "v2"
version = "v3"

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
        logging.info("Starting ppr"+str(ppr)+" for k="+ str(k))
        os.system('nohup python experiments.py'
                  + ' --method ppr'
                  + ' --walk '+str(ppr)
                  + ' --percent-triples ' + str(k)
                  + ' --version ' + version
                  + ' --version-answers ' + answer_version
                  + ' > ' + 'PPR'+str(ppr) + version+'a' + answer_version + '#K' + str(k) + '.out'
                  + ' &')
        exit(1)
