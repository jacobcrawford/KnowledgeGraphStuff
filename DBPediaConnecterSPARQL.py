import sparql
import QueryLogReader
import analyse
import pandas as pd


mypath = "dbpedia3.9"

### PARSE LOGS ###
parsed_logs = QueryLogReader.parseDirectoryOfLogs(mypath)
user_list = {}
print(analyse.keyword_analysis(pd.DataFrame(parsed_logs)))

### Divide the log into personal logs in dict format {uid: list<string>} ###
for d in parsed_logs:
    try:
        user_list[d['uid']].append(d['query'])
    except KeyError:
        user_list[d['uid']] = [d['query']]


### Get uid analysis ###
df = pd.DataFrame(parsed_logs)
df_unique = analyse.uid_analysis(df)
# filter on count
df_unique = df_unique[df_unique['count'] < 1000]
df_unique = df_unique[df_unique['count'] > 200]

print("size:" + str(len(df_unique)))
print(df_unique)

uids = [df_unique['uid'].iloc[i] for i in range(len(df_unique))]


analysis_tuples = []
for uid in ["0126ace40b97576faafdc04d352bf2e2"]:#uids:
    queries = user_list[uid]

    # See if the queries contain known automated agent keywords
    automated_agent = False
    for q in queries:
        automated_agent = "dbo:wikiPageDisambiguates" in q
        if automated_agent:
            print("############## AUTO AGENT "+ str(uid) + " ##########")
            print(q)
            break
    if automated_agent:
        continue

    answerable_queries = []
    rows = []
    errors = 0
    no_ans = 0
    for q in queries:

        row = {'query': q, 'results': None, 'error': None}

        try:
            result = sparql.query('http://dbpedia.org/sparql', q, 30)
            results = [r for r in result]
            row['results'] = len(results)

            if len(results) > 0:
                print(results)
                exit(1)
                answerable_queries.append(q)
            else:
                no_ans +=1
        except Exception as e:
            print("Error happened for query: " + str(q) + ", error: " + str(e))
            errors += 1
            row['error'] = e
        rows.append(row)

    df_for_id = pd.DataFrame(rows,columns=['query','results','error'])
    total = int(float(df_for_id['results'].sum()))
    df_for_id.to_csv("user_queries/" + str(uid) + "(a" + str(len(answerable_queries)) + "#t" + str (len(queries)) + "#e" + str(errors) + "#na" + str(no_ans) + "#t" + str(total) + ").csv")
    analysis_tuples.append((uid, answerable_queries, queries))


for uid,ans,queries in analysis_tuples:
    print("uid:" + uid)
    print("Answerable queries: " + str(len(ans))+ "/" + str(len(queries)))




