import json
import math
from os import listdir
from os.path import join, isfile

import numpy as np
import pandas as pd

from experiments import runGLIMPSEExperiment, runGLIMPSEDynamicExperiment, printResults, runGLIMPSEExperimentOnceRDF, \
    loadDBPedia, runPagerankExperimentOnceRDF


def f1skew(acc):
    return (2*acc)/(1 + acc)

def merge_accuracy_for_old_and_normalization(split_low_high=False):
    path1 = "experiments_results"
    path2 = "experiments_results_pagerank"

    div = 10000
    tms = 1000000

    print("PPR2")
    def pctf(x):
        f = math.ceil(int(x)/47408000*tms)/div
        if f > 1:
            return str(math.floor(f))
        f = str(f)
        return f[0:f.find("1")+1]


    ppr2 = [f for f in listdir(path2) if isfile(join(path2, f)) and f.endswith(".csv") and "PPR#2" in f and "v3" in f]
    glimpse1 = [f for f in listdir(path1) if isfile(join(path1, f)) and f.endswith(".csv") and "e#0.01" in f and "v3" in f]

    ppr2_old = [f for f in listdir(path2) if
            isfile(join(path2, f)) and f.endswith(".csv") and "PPR#2" in f and "v2" in f ]
    glimpse1_old  = [f for f in listdir(path1) if
                isfile(join(path1, f)) and f.endswith(".csv") and "e#0.01" in f and "v2" in f]

    unique_entities = np.array(
        [143386, 6109, 468, 50405, 50220, 5114, 51025, 183291, 104142, 398, 52659, 81628, 920, 55066,
         49251])
    split_low = [i[0] for i in np.argwhere(unique_entities < np.median(unique_entities))]
    split_high =[i[0] for i in np.argwhere(unique_entities >= np.median(unique_entities))]
    all_indexes = [i for i in range(len(unique_entities))]

    def addPPR2(files,output,algo="",indexes=None):
        if indexes == None:
            indexes = [i for i in range(len(files))]

        for p in files:
            df = pd.read_csv(path2+ "/"+p)
            k = str(p.split("K#")[1].split("_PPR")[0])
            pct = pctf(k)
            acc2 = []
            for idx,(i,j) in enumerate(zip(df['match'],df['total'])):
                if idx in indexes:
                    acc2.append(i/j)
            acc2=np.array(acc2)
            acc2 = np.mean(acc2)
            output.append({'Accuracy': str(acc2), 'Algorithm': "ppr2 " + algo, 'K in % of |T|': pct})

    def addGLIMPSE(files,output, algo="", indexes=None):
        if indexes == None:
            indexes = [i for i in range(len(files))]
        ### calc acc
        for p in files:
            df = pd.read_csv(path1 + "/" + p)
            k = str(p.split("K#")[1].split("e#")[0])
            pct = pctf(k)
            acc2 = []
            for idx, (i, j) in enumerate(zip(df['match'], df['total'])):
                if idx in indexes:
                    acc2.append(i / j)
            acc2=np.array(acc2)
            acc2 = np.mean(acc2)
            output.append({'Accuracy': str(acc2), 'Algorithm': "glimpse-2 "+algo, 'K in % of |T|': pct})


    if split_low_high:
        rows_low = []
        rows_high = []
        addGLIMPSE(glimpse1_old, rows_low,'-old',split_low )
        addGLIMPSE(glimpse1,rows_low,'-norm',split_low)
        addPPR2(ppr2_old,rows_low,'-old',split_low)
        addPPR2(ppr2, rows_low, '-norm', split_low)

        addGLIMPSE(glimpse1_old, rows_high,'-old',split_high )
        addGLIMPSE(glimpse1,rows_high,'-norm',split_high)
        addPPR2(ppr2_old, rows_high, '-old', split_high)
        addPPR2(ppr2, rows_high, '-norm', split_high)
        print("high")
        old = [a for a in filter(lambda x: 'glimpse-2 -old' in x['Algorithm'],rows_high)]
        old.sort(key=lambda x: x["K in % of |T|"])
        new = [a for a in filter(lambda x: 'glimpse-2 -norm' in x['Algorithm'],rows_high)]
        new.sort(key=lambda x: x["K in % of |T|"])


        increase = [float(i['Accuracy']) for i,j in zip(old,new)][0:4]

        for i in increase:
            print(i)

        #print(json.dumps(old))
        #print("Low")
        #print(json.dumps(rows_low))
    else:
        all =[]
        addGLIMPSE(glimpse1_old, all, '-old', all_indexes)
        addGLIMPSE(glimpse1, all, '-norm', all_indexes)
        addPPR2(ppr2_old, all, '-old', all_indexes)
        addPPR2(ppr2, all, '-norm', all_indexes)
        print(json.dumps(all))



def printDynamic():
    path1 = "experiments_results"
    files = [f for f in listdir(path1) if
     isfile(join(path1, f)) and f.endswith(".csv") and "v5" in f and "S0.1" in f ]
    file = files[0]
    rows = []

    df = pd.read_csv(path1+"/"+file)
    for i,c in enumerate(df.columns[1:]):
        rows.append({'split':int(0.1*(i+1)*10)/10 , 'accuracy':np.mean(np.array(df[c].values)), 'method':'static'})
    print(json.dumps(rows))


def printDynamicRetrain():
    """
    The results were put into the csv file in a weird way, so we have to extract them differently.
    :return:
    """

    path1 = "experiments_results"
    files = [f for f in listdir(path1) if
     isfile(join(path1, f)) and f.endswith(".csv") and "v7" in f and "S0.1" in f ]
    file = files[0]
    rows = []

    df = pd.read_csv(path1 + "/" + file)
    for i,c in enumerate(df.columns[1:]):
        values = []
        for j in range(6):
            values.append(df[c][i+9*j])
        rows.append({'split':int(0.1*(i+1)*10)/10 , 'accuracy': np.mean(np.array(values)), 'method':'dynamic'})
    print(json.dumps(rows))


def printRDF():
    path1 = "user_query_log_answersRDF"
    files = [f for f in listdir(path1) if
             isfile(join(path1, f)) and f.endswith(".csv")]
    for f in files:
        df = pd.read_csv(path1 + "/" +f)
        print(len(df))

def printRDFResultPagerank():
    path1 = "experiments_results_pagerank"
    files = [f for f in listdir(path1) if
             isfile(join(path1, f)) and f.endswith(".csv") and "v4" in f]
    for f in files:
        df = pd.read_csv(path1 + "/" +f)
        print(len(df))

#printDynamic()
#printDynamicRetrain()
#printResults("v2",, key='Entities')
#printResults("v11")
#merge_accuracy_for_old_and_normalization(True)
#runGLIMPSEDynamicExperiment(answers_version="2",k=0.01,e=1e-2, kg_path="../dbpedia3.9/",version=8,split=0.1, retrain=True)
#runGLIMPSEDynamicExperiment(answers_version="2",k=0.01,e=1e-2, kg_path="../dbpedia3.9/",version=7,split=0.2)
#printRDFResultPagerank()
#merge_accuracy_for_old_and_normalization(split_low_high=True)
KG = loadDBPedia('../dbpedia3.9/', include_properties=False)
for k in [10*(10**-i) for i in range(2, 7)]:
#    runPagerankExperimentOnceRDF(k, 2, "10",answers_version="RDF", KG_in=KG)
#    KG.reset()
    runGLIMPSEExperimentOnceRDF(k, 1e-5, "13", "RDF", KG_in=KG)
#    KG.reset()
#    runGLIMPSEExperimentOnceRDF(k, 1e-5, "12", "RDF", KG_in=KG, include_relationship_prob=True)
#    KG.reset()

def fixRDFResults():
    path = "experiments_results/"
    path2 = "experiments_results_pagerank/"
    files = [f for f in listdir(path) if
                 isfile(join(path, f)) and f.endswith(".csv") and "v10" in f or "v11" in f]

    files2 = [f for f in listdir(path2) if
                 isfile(join(path2, f)) and f.endswith(".csv") and "v10" in f]

    for file in files:
        print(file)
        df = pd.read_csv(path+file)
        df1 = df
        if type(df['entities'][0]) == np.int64:
            print("did not change: " + file)
            continue
        else:
            print("Changed: " + file)
            df1.entities = [e.count("http") for e in df['entities']]
            df1.to_csv(path+file)

    for file in files2:
        print(file)
        df = pd.read_csv(path2+file)
        df1 = df
        if type(df['entities'][0]) == np.int64:
            print("did not change: " + file)
            continue
        else:
            print("Changed: " + file)
            df1.entities = [e.count("http") for e in df['entities']]
            df1.to_csv(path2+file)


def versionStats():
    path = "experiments_results/"
    path2 = "experiments_results_pagerank/"
    files = [path+f for f in listdir(path) if
             isfile(join(path, f)) and f.endswith(".csv") and "v10" in f]

    files_prob = [path+f for f in listdir(path) if
             isfile(join(path, f)) and f.endswith(".csv") and "v11" in f]

    files2 = [path2+f for f in listdir(path2) if
              isfile(join(path2, f)) and f.endswith(".csv") and "v10" in f]

    rows = []
    div = 10000
    tms = 1000000
    kg_triples = 47408000

    def pctf(x):
        f = math.ceil(int(x) / kg_triples * tms) / div
        if f > 1:
            print(f)
            return str(math.floor(f))
        f = str(f)
        return f[0:f.find("1") + 1]

    def makeEntitiesStats(files, algorithm):
        rows = []
        for f in files:
            df = pd.read_csv(f)
            avg  = df['entities'].sum()/len(df['entities'])
            if "PPR" in f:
                k = str(f.split("K#")[1].split("_PPR")[0])
            else:
                k = str(f.split("K#")[1].split("e#")[0])
            k = pctf(int(k))
            rows.append({'entities':avg,'k':k, 'algorithm':algorithm})
        print(json.dumps(rows))

    def makeRelationshipStats(files, algorithm):
        rows = []
        for f in files:
            df = pd.read_csv(f)
            avg = df['relationships'].sum() / len(df['relationships'])
            if "PPR" in f:
                k = str(f.split("K#")[1].split("_PPR")[0])
            else:
                k = str(f.split("K#")[1].split("e#")[0])
            k = pctf(int(k))
            rows.append({'relationships': avg, 'k': k, 'algorithm': algorithm})
        print(json.dumps(rows))

    makeRelationshipStats(files, 'glimpse-2')
    makeRelationshipStats(files_prob, 'glimpse-2 prob')
    makeRelationshipStats(files2, 'ppr2')

#versionStats()