# KnowledgeGraphStuff

This is the readme for the implementation work dome in the master thesis of Jacob Crawford.
The project can be cloned from: https://github.com/jacobcrawford/KnowledgeGraphStuff


The folders of the repo are:

* GLIMPSE_personalized_KGsummarization:
Contains the implementation of the GLIMPSE framework. Code has been modified from the original project. The original project can be found here: https://github.com/GemsLab/GLIMPSE-personalized-KGsummarization

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


## Running the code

The original user log files are to big to commit in git.

The user logs used in entity based experiments are in the folder **user_query_log_answers2** and for RDF based experiments they are found in **user_query_log_answersRDF**

The knowledge graph used shoud be downloaded with the **downLoadDBPedia39()** function in **util.py**
This will download all the files of DBPedia3.9.   

Select the files described in the paper and move them to another folder.   

To run GLIMPSE experiments use the function **runGLIMPSEExperimentOnce** in **experiments.py**. <br/>
To run GLIMPSE rdf experiments use the function **runGLIMPSEExperimentOnceRDF** in  **experiments.py**.  <br/>
To run dynamic glimpse experiment use function **runGLIMPSEDynamicExperiment** in **experiments.py**.  <br/><br/>
  
To run PageRank experiments use function **pageRankExperimentOnce** in **experiments.py**. <br/>
To run PageRank rdf experiments use the function **runPagerankExperimentOnceRDF** in  **experiments.py**. <br/>
  
ALWAYS use answers_version=2 for normal experiments and answers_version=RDF for RDF experiments.  <br/>
Set the version to output experiments results to a .cvs file with the version number.  <br/><br/>



