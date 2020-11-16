import json
import math
from os import listdir
from os.path import join, isfile

import numpy as np
import pandas as pd

from experiments import runGLIMPSEExperiment, runGLIMPSEDynamicExperiment, printResults


def f1skew(acc):
    return (2*acc)/(1 + acc)

def merge_accuracy_for_old_and_normalization():
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
    print("median: " + str(np.median(unique_entities)))
    rows_low = []
    rows_high = []

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

    addGLIMPSE(glimpse1_old, rows_low,'-old',split_low )
    addGLIMPSE(glimpse1,rows_low,'-norm',split_low)
    addPPR2(ppr2_old,rows_low,'-old',split_low)
    addPPR2(ppr2, rows_low, '-norm', split_low)

    addGLIMPSE(glimpse1_old, rows_high,'-old',split_high )
    addGLIMPSE(glimpse1,rows_high,'-norm',split_high)
    addPPR2(ppr2_old, rows_high, '-old', split_high)
    addPPR2(ppr2, rows_high, '-norm', split_high)


    print(json.dumps(rows_high))
    #print(json.dumps(rows_low))


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
    #print(df.columns)
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
#printResults("v4",include_properties=True)
#printResults("v6",include_properties=True)
#merge_accuracy_for_old_and_normalization()
runGLIMPSEDynamicExperiment(answers_version="2",k=0.01,e=1e-2, kg_path="../dbpedia3.9/",version=8,split=0.1, retrain=True)
#runGLIMPSEDynamicExperiment(answers_version="2",k=0.01,e=1e-2, kg_path="../dbpedia3.9/",version=7,split=0.2)
#printRDFResultPagerank()

