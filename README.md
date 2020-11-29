# KnowledgeGraphStuff

This is the readme for the implementation work dome in the master thesis of Jacob Crawford.


The folders of the repo are:

* GLIMPSE_personalized_KGsummarization:
Contains the implementation of the GLIMPSE framework. Code has been modified from the original project.

* experiments_results:
Contains all results of GLIMPSE experiments

* experiments_results_pagerank:
Contains all results of PageRank experiments

* log_files:
Folder used to output logging of experiments

* statistics:
Folder for holding different statistics of the data and experiments

* user_query_log_answers:
Holding the first extracted answers of the query logs

* user_query_log_answers2:
Holding the second extracted answers of the query logs. These are the ones used in experiments.

* user_query_log_answersRDF:
Holding the extracted answers of the query logs translated to CONSTRUCT queries. These are the ones used in experiments.

* visualizations:
Holds files where the content can be copied to https://vega.github.io/editor/#/edited, to see visualizations.

The files of the repo are:

* QueryLogReader.py:
Used to read the query logs 

* abstract_shorter.py:
Used to shorten the file long_abstracts.nt to reduce memory load.

* analyse.py: 
Used to run analysis 

* experiments.py:
Holds all setups of experiments

* extractAnswers.py:
Used to extract entity answers of SELECT queries and triple answers of CONSTRUCT queries.

* main.py:
Used to run scripts and functions and all sort of hacks.

* parallel_experiments.py
Used to try to run expreiments in parallel. Failes because of lack of memory.

* util.py 
Utility functions

* virtuoso_connector.py
Connector for a remote or local virtuoso endpoint



