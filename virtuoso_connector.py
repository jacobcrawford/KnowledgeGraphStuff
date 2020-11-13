from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions, RDF

import QueryLogReader


def makeQueryLogsUserList():
    mypath = "dbpedia3.9"

    ### PARSE LOGS ###
    parsed_logs = QueryLogReader.parseDirectoryOfLogs(mypath)
    user_list = {}

    ### Divide the log into personal logs in dict format {uid: list<string>} ###
    for d in parsed_logs:
        query = d['query'].replace("dbpprop","dbo")
        if "describe" in query.lower() or "ask" in query.lower() or "select" not in query.lower() or "dbo:wikiPageDisambiguates" in query:
            continue
        try:
            user_list[d['uid']].append(query)
        except KeyError:
            user_list[d['uid']] = [query]

    delete = []
    for k in user_list.keys():
        if len(user_list[k]) < 20: # CHANGED FROM 100
            delete.append(k)
    for k in delete:
        del user_list[k]
    return user_list

class VirtuosoConnector:

    def __init__(self, url="http://localhost:8890/sparql", defaultgraph='http://www.purl.com/KG_SUMMARY/DBPedia3.9_SMALL', format=JSON):
        self.sparql = SPARQLWrapper(url)
        self.sparql.setTimeout(100)
        self.sparql.setReturnFormat(format)
        if defaultgraph:
            self.sparql.addParameter('default-graph-uri', defaultgraph)


    def query(self,query:str):
        self.sparql.setQuery(query)
        try:
            return self.sparql.query().convert()
        except SPARQLExceptions.QueryBadFormed:
            raise SPARQLExceptions.QueryBadFormed()



    def extractIRIsFromJsonResults(self,results):
        answer_iris = []
        answers = results['results']['bindings']
        for a in answers:
            for k in a.keys():
                answer_iris.append(a[k]['value'])
        return answer_iris

    def extractAnswersToUsersQueryLog(self,queries):
        answers = []
        errors = 0
        no_ans = 0
        for q in queries:
            try:
                answer = self.query(q)
                answer = self.extractIRIsFromJsonResults(answer)
                if len(answer) > 0:
                    answers.append(answer)
                else:
                    no_ans += 1
            except SPARQLExceptions.QueryBadFormed:
                print("ERROR BAD QUERY")
                errors +=1
            except Exception as e:
                print("Error", e)
                errors += 1
        return answers, errors, len(queries), no_ans





#main()
#main3("e501b61d45deb9937635cacba1c3b87a")


