import pandas as pd
from SPARQLWrapper import RDFXML
from rdflib import Literal

from virtuoso_connector import makeQueryLogsUserList, VirtuosoConnector


def extractAnswersToQueryInRDF():
    total_succes = 0
    total_answers = 0
    total_nested_select_count = 0
    total_union_count = 0
    total_property_path_count = 0
    total_rdf_no_result = 0
    total_errors = 0
    square_count_total = 0

    def removeOptional(query):
        o_idx = query.lower().find("optional")
        rem_o_end = query[o_idx:len(query)].find("}")

        result = query[0: o_idx] + query[o_idx + rem_o_end + 1: len(query)]

        if "optional" in result.lower():
            return removeOptional(result)
        return result

    def handleUnion(where, debug=False):
        parts = where.lower().replace("}","").replace("{","").split("union")
        where_new = ""
        for part in parts:
            if debug:
                print("     \n part")
                print("     " + part)
                print("\n")

                print(part.replace(" ", "")[::-1][0])

            if part.replace(" " ,"")[::-1][0] == ".":
                if debug:
                    print("     DOT")
                where_new += part.replace("}","").replace("{","") + " \n"
            else:
                if debug:
                    print("     NO DOT")
                where_new += part.replace("}", "").replace("{", "") + " .\n"
        return "{" + where_new + "}"

    def makeConstructQuery(q, debug=False):
        # Remove line comments
        lines = q.splitlines()
        q = ""
        for line in lines:
            if len(line) > 0 and line[0] == "#":
                continue
            q += line + "\n"

        select_idx = q.lower().find("select")
        # remove select
        first = q[:select_idx]
        # extract where
        brac_start_idx = q.find('{')
        brac_end_idx = len(q)  - q[::-1].find('}') - 1
        where  = q[brac_start_idx:brac_end_idx + 1]

        # More brachets not supported in construct
        where_c = where
        if debug:
            print("     Where START:")
            print(where_c)
            print("     Where END")

        if (where.replace(" ","")[0] == "{" and where.replace(" ","")[1] == "{") and "union" not in where_c.lower():
            if debug:
                print("NO UNION")
                print(where[1:len(where)-1])
            where_c = where[1:len(where)-1]
        if "optional" in where.lower():
            where_c = removeOptional(where_c)
        if "filter" in where_c.lower():
            f = where_c.lower().find("filter")
            where_c = where[0:f] + "}"
        if "union" in where_c.lower():
            where_c = handleUnion(where_c,debug)

        return first + "CONSTRUCT " + where_c + "WHERE" + where + " LIMIT 1000"

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

    user_list = makeQueryLogsUserList()
    df = pd.read_csv("user_stats2.csv")
    df = df[df['answers'] >= 15]

    print("total answers in select:" + str(df['answers'].sum()))
    v = VirtuosoConnector(format=RDFXML)
    v1 = VirtuosoConnector()

    for uid in df['uid']:
        rows = []
        i = 0
        success = 0
        nested_select_count = 0
        union_count = 0
        property_path_count = 0
        errors_count = 0
        rdf_no_result = 0
        square_count = 0

        for q in user_list[uid]:
            try:
                results = v1.query(q)
                results = v1.extractIRIsFromJsonResults(results)
                if len(results) == 0:
                    continue
                else:
                    success += 1
            except:
                continue

            where_idx = q.lower().find("where")
            if "select" in q[where_idx:len(q)].lower():
                nested_select_count +=1
                continue
            if "*" in q[where_idx:len(q)].lower():
                property_path_count +=1
                continue
            if "[]" in q[where_idx:len(q)].lower():
                square_count +=1
                continue

            try:
                qc = makeConstructQuery(q)
                if "union" in qc.lower():
                    union_count += 1
                results = v.query(qc)
                results = extractTriples(results)
                if len(results) == 0:
                    rdf_no_result += 1
                    continue
                if len(results) != 0:
                    rows.append({'id': i,'query':qc, 'answers': " ".join(results)})
                    i += 1
            except Exception as e:
                errors_count +=1
                print("ERROR")
                print(e)
                makeConstructQuery(q, True)
        total_rdf_no_result += rdf_no_result
        total_errors += errors_count
        total_nested_select_count += nested_select_count
        total_union_count += union_count
        total_property_path_count += property_path_count
        total_succes += success
        total_answers += len(rows)
        square_count_total += square_count
        print("     Number of successfull normal queries :" + str(success))
        print("     Queries with results:" + str(len(rows)))
        print("     Queries with no results:" + str(rdf_no_result))
        print("     Queries with nested: " + str(nested_select_count))
        print("     Queries with union: " + str(union_count))
        print("     Queries with prop path: " + str(property_path_count))
        print("     Queries with error: " + str(errors_count))
        print("     Queries with square" + str(square_count))
        print("\n")
        pd.DataFrame(rows).to_csv("user_query_log_answersRDF/" + uid + ".csv")
    print("Total normal success " + str(total_succes))
    print("total usefull answers " + str(total_answers))
    print("total union queries " + str(total_union_count))
    print("total prop path queries" +str(total_property_path_count))
    print("Total nested select " + str(total_nested_select_count))
    print("Total no result rdf " + str(total_rdf_no_result))
    print("Total errors" + str(total_errors))
    print("Total square" + str(square_count_total))

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