import pandas as pd
from bs4 import BeautifulSoup
import requests
import urllib.request

import QueryLogReader
from analyse import queryDBPediaResourceAnalysis, getFileContaining


#mypath = "dbpedia3.9"
#parsed_logs = QueryLogReader.parseDirectoryOfLogs(mypath)
#df = pd.DataFrame(parsed_logs)
#d = queryDBPediaResourceAnalysis(df)

def listFD(url, ext=''):
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]

def downLoadDBPedia39():
    url = 'http://downloads.dbpedia.org/3.9/en/'
    ext = 'nt.bz2'
    print('Beginning file download with urllib2...')
    download = False
    for file in listFD(url, ext):
        name = str(file)
        print(name)
        if name == "http://downloads.dbpedia.org/3.9/en//old_interlanguage_links_see_also_chapters_en.nt.bz2":
            print("Resume download")
            download = True
        if download:
            name = file.split("http://downloads.dbpedia.org/3.9/en//")[1]

            urllib.request.urlretrieve(file, './DBPedia3.9Full/' + name)
            print("Downloaded")

getFileContaining("<http://purl.org/dc/terms/subject>")


