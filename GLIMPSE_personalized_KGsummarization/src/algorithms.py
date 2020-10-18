import numpy as np


def query_vector(KG, query_log_answers):
    """
    :param KG: KnowledgeGraph
    :param query_log: list of answers to queries in iri format
    :return x: query vector (n_entities,)
    """
    x = np.zeros(KG.number_of_entities())
    for i in range(len(query_log_answers)):
        for entity in query_log_answers[i]:
            entity_id = KG.entity_id(entity)
            x[entity_id] += 1/len(query_log_answers[i])
    return x

def random_walk_with_restart(M, x, c=0.15, power=1):
    """
    :param M: scipy sparse transition matrix
    :param x: np.array (n_entities,) seed initializations
    :param c: float in [0, 1], optional restart prob
    :param power: number of terms in Taylor expansion
    :return r: np.array (n_entities,) random walk vector

    Approximates the matrix inverse using the Taylor expansion:
        (I - M)^-1 = I + M + M^2 + M^3 ...
    """
    q = c * np.copy(x)
    r = np.copy(q) # result vector

    for _ in range(power):
        q = (1 - c) * M * q
        r += q
        r /= np.sum(r)
    return r
