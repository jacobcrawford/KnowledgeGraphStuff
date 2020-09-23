import urllib.parse
from os import listdir
from os.path import isfile, join

def parse_log_file(file_name):
    """
    :param file_name: The name of the file to parse
    :return: a dict of uid:query
    """
    unparsed_log_file = open(file_name)
    unparsed_log = unparsed_log_file.read()

    parsed_log = urllib.parse.unquote_plus(urllib.parse.unquote(unparsed_log))

    parsed_log_http_split_list = parsed_log.split("\"-\" \"R\" \"-\"")

    def extract_id(log_entry):
        """
        Extracts an id from the query log entry string
        :param log_entry:
        :return: the id of the agent who executed the query
        """
        return log_entry.split("-", maxsplit=2)[0].replace("\n", "").strip()

    def extract_query(log_entry):
        """
        Extract the query from a log entry
        :param log_entry:
        :return:
        """
        split_list = log_entry.split("query=")
        if len(split_list) == 1:
            query_string = log_entry.split("qtxt=")[1].split("&format")[0].split("&callback")[0].split("&debug")[0].split("HTTP/1")[0].split("&output=")[0].split("&_")[0]
        else:
            query_string = log_entry.split("query=")[1].split("&format")[0].split("&callback")[0].split("&debug")[0].split("HTTP/1")[0].split("&output=")[0].split("&_")[0]

        return query_string

    # Remove calls where there are no query
    parsed_log_http_split_list_clean = [x for x in parsed_log_http_split_list if ("query=" in x or "qtxt=" in x) ]

    return [{'uid': extract_id(log_split), 'query': extract_query(log_split)} for log_split in
            parsed_log_http_split_list_clean]


def parseDirectoryOfLogs(path:str):
    """
    Parse a whole directory of log files
    :param path: The relative path to the directory
    :return: A list of {uid:query} objects
    """
    files = [f for f in listdir(path) if isfile(join(path, f))]
    parsed_logs = []
    for f in files:
        parsed_logs = parsed_logs + parse_log_file(path + "/" + f)
    return parsed_logs
