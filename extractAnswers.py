import pandas as pd
from SPARQLWrapper import RDFXML
from rdflib import Literal

from virtuoso_connector import makeQueryLogsUserList, VirtuosoConnector


def extractAnswersToQueryInRDF():
    def removeOptional(query):
        o_idx = query.lower().find("optional")
        rem_o_end = query[o_idx:len(query)].find("}")
        result = query[0: o_idx] + query[o_idx + rem_o_end: len(query)]
        print(result)
        if "optional" in result.lower():
            print("MORE")
            return removeOptional(result)
        return result

    def makeConstructQuery(q):

        select_idx = q.lower().find("select")
        # remove select
        first = q[:select_idx]
        # extract where
        brac_start_idx = q.find('{')
        brac_end_idx = len(q)  - q[::-1].find('}') - 1
        where  = q[brac_start_idx:brac_end_idx + 1]
        if "filter" in where.lower():
            f = where.lower().find("filter")
            where_c = where[0:f] + "}"
        else:
            where_c = where
        if "optional" in where.lower():
            where_c = removeOptional(where_c)
        return first + "CONSTRUCT " + where_c + "WHERE" + where + q[brac_end_idx + 2: len(q)+1]

    def extractTriples(results_rdf_lib):
        results_triple = []
        for triple in results_rdf_lib:
            e1 = triple[0]
            r = triple[1]
            e2 = triple[2]
            # Skip triples with property values
            if type(e1) == Literal or type(e2) == Literal:
                break
            result = str(e1) + " " + str(r) + " " + str(e2)
            results_triple.append(result)
        return results_triple

    a = "SELECT ?name ?description_en ?description_de ?musician WHERE { \
         ?musician <http://purl.org/dc/terms/subject> <http://dbpedia.org/resource/Category:German_musicians> .\
         ?musician foaf:name ?name .\
         OPTIONAL {\
             ?musician rdfs:comment ?description_en .\
             FILTER (LANG(?description_en) = 'en') .\
         }\
         OPTIONAL {\
             ?musician rdfs:comment ?description_de .\
             FILTER (LANG(?description_de) = 'de') .\
         }\
       }\
        "

    user_list = makeQueryLogsUserList()
    df = pd.read_csv("user_stats2.csv")
    df = df[df['answers'] >= 20]
    v = VirtuosoConnector(format=RDFXML)
    v1 = VirtuosoConnector()
    for uid in df['uid']:
        rows = []
        i = 0
        success = 0
        for q in user_list[uid]:
            try:
                results = v1.query(q)
                if len(results) == 0:
                    continue
                else:
                    success += 1
            except:
                continue
            try:
                qc = makeConstructQuery(q)
                results = v.query(qc)

                if len(results) == 0:
                    continue
                results = extractTriples(results)
                if len(results) != 0:
                    rows.append({'id': i, 'answers': " ".join(results)})
                    i += 1
            except Exception as e:
                print("ERROR")
                print(e)
                print("\n####################")
                print(q)
                print("&&&&&&&&&&&&&&&&&&&")
                print(qc)
                print("####################\n")
        print("Number of success queries:" + str(success))
        print("Queries with results:" + str(len(rows)))
        pd.DataFrame(rows).to_csv("user_query_log_answersRDF/" + uid + ".csv")


def extractAnswersToQuery():
    user_list = makeQueryLogsUserList()
    df = pd.read_csv("user_stats2.csv")
    df = df[df['answers'] >= 15]
    v = VirtuosoConnector()
    for uid in df['uid']:
        rows = []
        i = 0
        for q in user_list[uid]:
            try:
                results = v.query(q)
                results = v.extractIRIsFromJsonResults(results)
                results = [r for r in results if "http" in r]
                if len(results)>0:
                    # Filter out non iri answers
                    rows.append({'id': i, 'answers': " ".join(results)})
                i += 1
            except Exception as e:
                print(e)

        pd.DataFrame(rows).to_csv("user_query_log_answers2/" + uid + ".csv")
