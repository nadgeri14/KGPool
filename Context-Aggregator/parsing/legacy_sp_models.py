# coding: utf-8
# Copyright (C) 2016 UKP lab
#
# Author: Daniil Sorokin (ukp.tu-darmstadt.de/ukp-home/)
#
import itertools
import numpy as np
np.random.seed(1)

import tqdm
import sys
import pdb

#sys.path.insert(0, '..') 
#sys.path.insert(0, '../..') # maybe troublesome when on windows


from utils import embedding_utils, graph
from semanticgraph import graph_utils
from utils.conversion_util import calculate_order_conversion

RESOURCES_FOLDER = "resources/"
property_blacklist = embedding_utils.load_blacklist(RESOURCES_FOLDER + "property_blacklist.txt")

def get_negative_edges(g, limit=1):
    """
    Generate negative edges for every entity pair if no relation is available. If generated set is bigger that limit, it will be dropped randomly.
    :param g: graphs a dictionary
    :return: a list of negative edges
    >>> get_negative_edges({'edgeSet': [{'kbID': 'P397', 'left': [8], 'right': [23]}, \
    {'kbID': 'P376', 'left': [80], 'right': [8]}], 'vertexSet': [{'tokenpositions': [8]}, {'tokenpositions': [23]}, {'tokenpositions': [80]}]}) \
    == [{'left': [23], 'kbID': 'P0', 'right': [80]}]
    True
    """
    # get all combinations of vertex set
    # combinations('ABCD', 2) => AB AC AD BC BD CD
    vertex_pairs = itertools.combinations(g["vertexSet"], 2)
    existing_edges = [p for e in g["edgeSet"] for p in [(e['left'], e['right']), (e['right'], e['left'])]]
    negative_edges = []
    for vertex_pair in vertex_pairs:
        left_right = (vertex_pair[0]['tokenpositions'], vertex_pair[1]['tokenpositions'])
        if left_right not in existing_edges:
            negative_edges.append({'kbID': 'P0', 'left': left_right[0], 'right': left_right[1]})
    if len(negative_edges) > limit:
        negative_edges = np.random.choice(negative_edges, limit, replace=False)
    return list(negative_edges)

def get_all_negative_edges(g, limit=100000):
    """
    Generate negative edges for every entity pair if no relation is available. If generated set is bigger that limit, it will be dropped randomly.
    :param g: graphs a dictionary
    :return: full list of edges
    >>> get_negative_edges({'edgeSet': [{'kbID': 'P397', 'left': [8], 'right': [23]}, \
    {'kbID': 'P376', 'left': [80], 'right': [8]}], 'vertexSet': [{'tokenpositions': [8]}, {'tokenpositions': [23]}, {'tokenpositions': [80]}]}) \
    == [{'left': [23], 'kbID': 'P0', 'right': [80]}]
    True
    """
    # get all products of vertex set
    # combinations('ABC', 2) => AB AC BC
    vertex_pairs = itertools.combinations(g["vertexSet"], 2)
    existing_edges = [p for e in g["edgeSet"] for p in [(e['left'], e['right']), (e['right'], e['left'])]]
    negative_edges = []
    for vertex_pair in vertex_pairs:
        left_right = (vertex_pair[0]['tokenpositions'], vertex_pair[1]['tokenpositions'])
        if left_right not in existing_edges:
            negative_edges.append({'kbID': 'P0', 'left': left_right[0], 'right': left_right[1]})
    if len(negative_edges) > limit:
        negative_edges = np.random.choice(negative_edges, limit, replace=False)
    return list(negative_edges)


def to_indices(graphs, word2idx, property2idx, max_sent_len, replace_entities_with_unkown = False, mode='train', **kwargs):
    """
    :param graphs:
    :param word2idx:
    :param property2idx:
    :param max_sent_len:
    :return:
    """
    num_edges = len([e for g in graphs for e in g['edgeSet'] if e['kbID'] not in property_blacklist])
    print("Dataset number of edges: {}".format(num_edges))
    sentences_matrix = np.zeros((num_edges, max_sent_len), dtype="int32")
    entity_matrix = np.zeros((num_edges, max_sent_len), dtype="int8")
    y_matrix = np.zeros(num_edges, dtype="int16")
    index = 0
    for g in tqdm.tqdm(graphs, ascii=True):
        token_ids = embedding_utils.get_idx_sequence(g["tokens"], word2idx)
        if len(token_ids) > max_sent_len:
            token_ids = token_ids[:max_sent_len]
        for edge in g["edgeSet"]:
            if edge['kbID'] not in property_blacklist:
                sentences_matrix[index, :len(token_ids)] = \
                    [word2idx[embedding_utils.unknown] if i in edge["left"] + edge["right"] else t for i, t in enumerate(token_ids)] \
                        if replace_entities_with_unkown else token_ids
                entity_matrix[index, :len(token_ids)] = \
                    [m for _, m in graph_utils.get_entity_indexed_vector(token_ids, edge, mode="mark-bi")]
                if mode == "train":
                    _, property_kbid, _ = graph_utils.edge_to_kb_ids(edge, g)
                    property_kbid = property2idx.get(property_kbid, property2idx[embedding_utils.unknown])
                    y_matrix[index] = property_kbid
                index += 1
    return [sentences_matrix, entity_matrix, y_matrix]

def to_indices_and_entity_pair(graphs, word2idx, property2idx, max_sent_len, replace_entities_with_unkown = False, mode='train', **kwargs):
    """
    :param graphs:
    :param word2idx:
    :param property2idx:
    :param max_sent_len:
    :return:
    """
    num_edges = len([e for g in graphs for e in g['edgeSet'] if e['kbID'] not in property_blacklist])
    print("Dataset number of edges: {}".format(num_edges))
    sentences_matrix = np.zeros((num_edges, max_sent_len), dtype="int32")
    entity_matrix = np.zeros((num_edges, max_sent_len), dtype="int8")
    y_matrix = np.zeros(num_edges, dtype="int16")
    index = 0
    entity_cnt = []
    pos2id = dict()
    entity_pair = []
    for g in tqdm.tqdm(graphs, ascii=True):
        token_ids = embedding_utils.get_idx_sequence(g["tokens"], word2idx)
        try:    
            entity_cnt.append(len(g["vertexSet"]))  
            for i in g['vertexSet']:
                pos2id[tuple(i['tokenpositions'])] = i['kbID']
        except:
            continue
        if len(token_ids) > max_sent_len:
            token_ids = token_ids[:max_sent_len]
        for edge in g["edgeSet"]:
            if edge['kbID'] not in property_blacklist:
                sentences_matrix[index, :len(token_ids)] = \
                    [word2idx[embedding_utils.unknown] if i in edge["left"] + edge["right"] else t for i, t in enumerate(token_ids)] \
                        if replace_entities_with_unkown else token_ids
                entity_matrix[index, :len(token_ids)] = \
                    [m for _, m in graph_utils.get_entity_indexed_vector(token_ids, edge, mode="mark-bi")]
                if mode == "train":
                    _, property_kbid, _ = graph_utils.edge_to_kb_ids(edge, g)
                    property_kbid = property2idx.get(property_kbid, property2idx[embedding_utils.unknown])
                    y_matrix[index] = property_kbid
                entity_pair.append((pos2id[tuple(edge['left'])], pos2id[tuple(edge['right'])]))    
                index += 1
    return [sentences_matrix, entity_matrix, y_matrix, entity_pair]

MAX_EDGES_PER_GRAPH = 72

def to_indices_with_real_entities(graphs, word2idx, property2idx, max_sent_len, mode='train', **kwargs):
    """
    :param graphs:
    :param word2idx:
    :param property2idx:
    :param max_sent_len:
    :return:
    """
    graphs_to_process = []
    for g in graphs:
        if len(g['edgeSet']) > 0:
            if len(g['edgeSet']) <= MAX_EDGES_PER_GRAPH:
                graphs_to_process.append(g)
            else:
                for i in range(0, len(g['edgeSet']), MAX_EDGES_PER_GRAPH):
                    graphs_to_process.append({"tokens": g["tokens"], "edgeSet": g["edgeSet"][i:i+ MAX_EDGES_PER_GRAPH]})
    graphs = graphs_to_process
    sentences_matrix = np.zeros((len(graphs), max_sent_len), dtype="int32")
    entity_matrix = np.zeros((len(graphs), MAX_EDGES_PER_GRAPH, max_sent_len), dtype="int8")
    y_matrix = np.zeros((len(graphs), MAX_EDGES_PER_GRAPH), dtype="int16")
    for index, g in enumerate(tqdm.tqdm(graphs, ascii=True)):
        token_ids = embedding_utils.get_idx_sequence(g["tokens"], word2idx)
        if len(token_ids) > max_sent_len:
            token_ids = token_ids[:max_sent_len]
        sentences_matrix[index, :len(token_ids)] = token_ids
        for j, edge in enumerate(g["edgeSet"][:MAX_EDGES_PER_GRAPH]):
            entity_matrix[index, j, :len(token_ids)] = \
                [m for _, m in graph_utils.get_entity_indexed_vector(token_ids, edge, mode="mark-bi")]
            _, property_kbid, _ = graph_utils.edge_to_kb_ids(edge, g)
            property_kbid = property2idx.get(property_kbid, property2idx[embedding_utils.unknown])
            y_matrix[index, j] = property_kbid
    return sentences_matrix, entity_matrix, y_matrix

def to_indices_with_real_entities_and_entity_nums(graphs, word2idx, property2idx, max_sent_len, mode='train', **kwargs):
    """
    :param graphs:
    :param word2idx:
    :param property2idx:
    :param max_sent_len:
    :return:
    """
    graphs_to_process = []
    for g in graphs:
        if len(g['edgeSet']) > 0:
            if len(g['edgeSet']) <= MAX_EDGES_PER_GRAPH:
                graphs_to_process.append(g)
            else:
                continue # here we discard these data points
                for i in range(0, len(g['edgeSet']), MAX_EDGES_PER_GRAPH):
                    graphs_to_process.append({"tokens": g["tokens"], "edgeSet": g["edgeSet"][i:i+ MAX_EDGES_PER_GRAPH]})
    graphs = graphs_to_process
    sentences_matrix = np.zeros((len(graphs), max_sent_len), dtype="int32")
    entity_matrix = np.zeros((len(graphs), MAX_EDGES_PER_GRAPH, max_sent_len), dtype="int8")
    y_matrix = np.zeros((len(graphs), MAX_EDGES_PER_GRAPH), dtype="int16")
    entity_cnt = []
    for index, g in enumerate(tqdm.tqdm(graphs, ascii=True)):
        try:    
            entity_cnt.append(len(g["vertexSet"]))    
        except:
            continue
        token_ids = embedding_utils.get_idx_sequence(g["tokens"], word2idx)
        if len(token_ids) > max_sent_len:
            token_ids = token_ids[:max_sent_len]
        sentences_matrix[index, :len(token_ids)] = token_ids
        for j, edge in enumerate(g["edgeSet"][:MAX_EDGES_PER_GRAPH]):
            entity_matrix[index, j, :len(token_ids)] = \
                [m for _, m in graph_utils.get_entity_indexed_vector(token_ids, edge, mode="mark-bi")]
            _, property_kbid, _ = graph_utils.edge_to_kb_ids(edge, g)
            property_kbid = property2idx.get(property_kbid, property2idx[embedding_utils.unknown])
            y_matrix[index, j] = property_kbid
    entity_cnt = np.array(entity_cnt, dtype=np.int32)        
         
    return sentences_matrix, entity_matrix, y_matrix, entity_cnt

def to_indices_with_real_entities_and_entity_nums_with_vertex_padding(graphs, word2idx, property2idx, max_sent_len, mode='train', **kwargs):
    """
    :param graphs:
    :param word2idx:
    :param property2idx:
    :param max_sent_len:
    :return:
    """
    graphs_to_process = []
    for g in graphs:
        if len(g['edgeSet']) > 0:
            if len(g['edgeSet']) <= MAX_EDGES_PER_GRAPH:
                graphs_to_process.append(g)
            else:
                continue # here we discard these data points
                for i in range(0, len(g['edgeSet']), MAX_EDGES_PER_GRAPH):
                    graphs_to_process.append({"tokens": g["tokens"], "edgeSet": g["edgeSet"][i:i+ MAX_EDGES_PER_GRAPH]})
    graphs = graphs_to_process
    sentences_matrix = np.zeros((len(graphs), max_sent_len), dtype="int32")
    entity_matrix = np.zeros((len(graphs), MAX_EDGES_PER_GRAPH, max_sent_len), dtype="int8")
    y_matrix = np.zeros((len(graphs), MAX_EDGES_PER_GRAPH), dtype="int16")
    kbID_matrix = np.empty((len(graphs), MAX_EDGES_PER_GRAPH), dtype=object)
    entity_cnt = []
    for index, g in enumerate(tqdm.tqdm(graphs, ascii=True)):
        try:    
            entity_cnt.append(len(g["vertexSet"]))    
        except:
            continue
        token_ids = embedding_utils.get_idx_sequence(g["tokens"], word2idx)
        if len(token_ids) > max_sent_len:
            token_ids = token_ids[:max_sent_len]
        sentences_matrix[index, :len(token_ids)] = token_ids
        for j, edge in enumerate(g["edgeSet"][:MAX_EDGES_PER_GRAPH]):
            new_j = calculate_order_conversion(j, len(g["vertexSet"]))
            entity_matrix[index, new_j, :len(token_ids)] = \
                [m for _, m in graph_utils.get_entity_indexed_vector(token_ids, edge, mode="mark-bi")]
            left_entity, property_kbid, right_entity = graph_utils.edge_to_kb_ids(edge, g)
            relationID = property_kbid
            property_kbid = property2idx.get(property_kbid, property2idx[embedding_utils.unknown])
            y_matrix[index, new_j] = property_kbid
            kbID_matrix[index, new_j] = {"graph":g,"left_entity":left_entity,"right_entity":right_entity,"relation":relationID}
    entity_cnt = np.array(entity_cnt, dtype=np.int32)     
    return sentences_matrix, entity_matrix, y_matrix, entity_cnt, kbID_matrix

def to_indices_with_real_entities_and_entity_nums_with_vertex_padding_and_entity_pair(graphs, word2idx, property2idx, max_sent_len, mode='train', **kwargs):
    """
    :param graphs:
    :param word2idx:
    :param property2idx:
    :param max_sent_len:
    :return:
    """
    graphs_to_process = []
    for g in graphs:
        if len(g['edgeSet']) > 0:
            if len(g['edgeSet']) <= MAX_EDGES_PER_GRAPH:
                graphs_to_process.append(g)
            else:
                continue # here we discard these data points
                for i in range(0, len(g['edgeSet']), MAX_EDGES_PER_GRAPH):
                    graphs_to_process.append({"tokens": g["tokens"], "edgeSet": g["edgeSet"][i:i+ MAX_EDGES_PER_GRAPH]})
    graphs = graphs_to_process
    sentences_matrix = np.zeros((len(graphs), max_sent_len), dtype="int32")
    entity_matrix = np.zeros((len(graphs), MAX_EDGES_PER_GRAPH, max_sent_len), dtype="int8")
    y_matrix = np.zeros((len(graphs), MAX_EDGES_PER_GRAPH), dtype="int16")
    entity_cnt = []
    kbID_matrix = np.empty((len(graphs), MAX_EDGES_PER_GRAPH), dtype=object)
    pos2id = dict()
    entity_pair = []
    for index, g in enumerate(tqdm.tqdm(graphs, ascii=True)):
        try:    
            entity_cnt.append(len(g["vertexSet"]))  
            for i in g['vertexSet']:
                pos2id[tuple(i['tokenpositions'])] = i['kbID']
        except:
            continue
        token_ids = embedding_utils.get_idx_sequence(g["tokens"], word2idx)
        if len(token_ids) > max_sent_len:
            token_ids = token_ids[:max_sent_len]
        sentences_matrix[index, :len(token_ids)] = token_ids
        entity_pair_instance = []
        for j, edge in enumerate(g["edgeSet"][:MAX_EDGES_PER_GRAPH]):
            new_j = calculate_order_conversion(j, len(g["vertexSet"]))
            entity_matrix[index, new_j, :len(token_ids)] = \
                [m for _, m in graph_utils.get_entity_indexed_vector(token_ids, edge, mode="mark-bi")]
            left_entity, property_kbid, right_entity = graph_utils.edge_to_kb_ids(edge, g)
            relationID = property_kbid
            property_kbid = property2idx.get(property_kbid, property2idx[embedding_utils.unknown])
            y_matrix[index, new_j] = property_kbid
            kbID_matrix[index, new_j] = {"graph":g,"left_entity":left_entity,"right_entity":right_entity,"relation":relationID}
            entity_pair_instance.append((pos2id[tuple(edge['left'])], pos2id[tuple(edge['right'])]))
        entity_pair.append(entity_pair_instance)    
    entity_cnt = np.array(entity_cnt, dtype=np.int32)        
         
    return sentences_matrix, entity_matrix, y_matrix, entity_cnt, entity_pair, kbID_matrix

def to_indices_with_real_entities_completely(graphs, word2idx, property2idx, max_sent_len, mode='train', **kwargs):
    """
    This function add N/A relations to all entity pairs with no relation in dataset
    :param graphs:
    :param word2idx:
    :param property2idx:
    :param max_sent_len:
    :return:
    """
    graphs_to_process = []
    for g in graphs:
        if len(g['edgeSet']) > 0:
            if len(g['edgeSet']) <= MAX_EDGES_PER_GRAPH:
                graphs_to_process.append(g)
            else:
                for i in range(0, len(g['edgeSet']), MAX_EDGES_PER_GRAPH):
                    graphs_to_process.append({"tokens": g["tokens"], "edgeSet": g["edgeSet"][i:i+ MAX_EDGES_PER_GRAPH]})
    graphs = graphs_to_process
    sentences_matrix = np.zeros((len(graphs), max_sent_len), dtype="int32")
    entity_matrix = np.zeros((len(graphs), MAX_EDGES_PER_GRAPH, max_sent_len), dtype="int8")
    y_matrix = np.zeros((len(graphs), MAX_EDGES_PER_GRAPH), dtype="int16")
    for index, g in enumerate(tqdm.tqdm(graphs, ascii=True)):
        token_ids = embedding_utils.get_idx_sequence(g["tokens"], word2idx)
        if len(token_ids) > max_sent_len:
            token_ids = token_ids[:max_sent_len]
        sentences_matrix[index, :len(token_ids)] = token_ids
        for j, edge in enumerate(g["edgeSet"][:MAX_EDGES_PER_GRAPH]):
            entity_matrix[index, j, :len(token_ids)] = \
                [m for _, m in graph_utils.get_entity_indexed_vector(token_ids, edge, mode="mark-bi")]
            _, property_kbid, _ = graph_utils.edge_to_kb_ids(edge, g)
            property_kbid = property2idx.get(property_kbid, property2idx[embedding_utils.unknown])
            y_matrix[index, j] = property_kbid
    return sentences_matrix, entity_matrix, y_matrix


def graphs_for_evaluation(graphs, graphs_tagged):
    for_evaluation = []
    for i, g in enumerate(tqdm.tqdm(graphs, ascii=True, ncols=100)):
        for edge in g["edgeSet"]:
            new_g = {"edgeSet": [edge], "tokens": g['tokens']}
            entities = [ne for ne, t in graph.extract_entities(graphs_tagged[i])]
            entities += [edge['left'], edge['right']]
            new_g['vertexSet'] = [{'tokenpositions': ne} for ne in entities]
            new_g['edgeSet'].extend(get_negative_edges(new_g, limit=6))
            for_evaluation.append(new_g)
    return for_evaluation


def to_indices_with_ghost_entities(graphs, word2idx, property2idx, max_sent_len, embeddings, **kwargs):
    sentences_matrix, entity_matrix, y_matrix = to_indices(graphs, word2idx, property2idx, max_sent_len, **kwargs)
    ghost_entity_matrix = create_ghost_edges(sentences_matrix, entity_matrix, embeddings)
    entity_matrix = entity_matrix.reshape((entity_matrix.shape[0], 1, entity_matrix.shape[1]))
    entity_matrix = np.concatenate([entity_matrix, ghost_entity_matrix], axis = 1)
    return [sentences_matrix, entity_matrix, y_matrix]


def to_indices_with_relative_positions(graphs, word2idx, property2idx, max_sent_len, position2idx, **kwargs):
    num_edges = len([e for g in graphs for e in g['edgeSet']])
    sentences_matrix = np.zeros((num_edges, max_sent_len), dtype="int32")
    entity_matrix = np.zeros((num_edges, 2, max_sent_len), dtype="int8")
    y_matrix = np.zeros(num_edges, dtype="int16")
    index = 0
    max_entity_index = max_sent_len - 1
    for g in tqdm.tqdm(graphs, ascii=True):
        token_ids = embedding_utils.get_idx_sequence(g["tokens"], word2idx)
        if len(token_ids) > max_sent_len:
            token_ids = token_ids[:max_sent_len]
        for edge in g["edgeSet"]:
            sentences_matrix[index, :len(token_ids)] = token_ids
            _, property_kbid, _ = graph_utils.edge_to_kb_ids(edge, g)
            try:
                property_kbid = property2idx.get(property_kbid, property2idx[embedding_utils.unknown])
            except:
                pdb.set_trace()    
            entity_vector = graph_utils.get_entity_indexed_vector(token_ids, edge, mode="position")
            entity_vector = [(-max_entity_index if m1 < -max_entity_index else max_entity_index if m1 > max_entity_index else m1,
                              -max_entity_index if m2 < -max_entity_index else max_entity_index if m2 > max_entity_index else m2) for _, m1,m2  in entity_vector]
            entity_matrix[index, :, :len(token_ids)] = [[position2idx[m] for m,_  in entity_vector],[position2idx[m] for _, m  in entity_vector]]

            y_matrix[index] = property_kbid
            index += 1
    return [sentences_matrix, entity_matrix, y_matrix]

def to_indices_with_relative_positions_and_entity_pair(graphs, word2idx, property2idx, max_sent_len, position2idx, **kwargs):
    num_edges = len([e for g in graphs for e in g['edgeSet']])
    sentences_matrix = np.zeros((num_edges, max_sent_len), dtype="int32")
    entity_matrix = np.zeros((num_edges, 2, max_sent_len), dtype="int8")
    y_matrix = np.zeros(num_edges, dtype="int16")
    index = 0
    max_entity_index = max_sent_len - 1
    entity_pair = []
    pos2id = dict()
    for g in tqdm.tqdm(graphs, ascii=True):
        try:    
            for i in g['vertexSet']:
                pos2id[tuple(i['tokenpositions'])] = i['kbID']
        except:
            continue
        token_ids = embedding_utils.get_idx_sequence(g["tokens"], word2idx)
        if len(token_ids) > max_sent_len:
            token_ids = token_ids[:max_sent_len]
        entity_pair_instance = []
        for edge in g["edgeSet"]:
            sentences_matrix[index, :len(token_ids)] = token_ids
            _, property_kbid, _ = graph_utils.edge_to_kb_ids(edge, g)
            try:
                property_kbid = property2idx.get(property_kbid, property2idx[embedding_utils.unknown])
            except:
                pdb.set_trace()    
            entity_vector = graph_utils.get_entity_indexed_vector(token_ids, edge, mode="position")
            entity_vector = [(-max_entity_index if m1 < -max_entity_index else max_entity_index if m1 > max_entity_index else m1,
                              -max_entity_index if m2 < -max_entity_index else max_entity_index if m2 > max_entity_index else m2) for _, m1,m2  in entity_vector]
            entity_matrix[index, :, :len(token_ids)] = [[position2idx[m] for m,_  in entity_vector],[position2idx[m] for _, m  in entity_vector]]

            y_matrix[index] = property_kbid
            index += 1
            entity_pair_instance.append((pos2id[tuple(edge['left'])], pos2id[tuple(edge['right'])]))
        entity_pair += entity_pair_instance
    return [sentences_matrix, entity_matrix, y_matrix, entity_pair]

def to_indices_with_relative_positions_and_pcnn_mask_and_entity_pair(graphs, word2idx, property2idx, max_sent_len, position2idx, **kwargs):
    num_edges = len([e for g in graphs for e in g['edgeSet']])
    sentences_matrix = np.zeros((num_edges, max_sent_len), dtype="int32")
    entity_matrix = np.zeros((num_edges, 2, max_sent_len), dtype="int8")
    pcnn_mask = np.zeros((num_edges, 3, max_sent_len), dtype="float32")
    y_matrix = np.zeros(num_edges, dtype="int16")
    index = 0
    max_entity_index = max_sent_len - 1
    entity_pair = []
    pos2id = dict()
    for g in tqdm.tqdm(graphs, ascii=True):
        try:    
            for i in g['vertexSet']:
                pos2id[tuple(i['tokenpositions'])] = i['kbID']
        except:
            continue
        token_ids = embedding_utils.get_idx_sequence(g["tokens"], word2idx)
        if len(token_ids) > max_sent_len:
            token_ids = token_ids[:max_sent_len]
        entity_pair_instance = []
        for edge in g["edgeSet"]:
            sentences_matrix[index, :len(token_ids)] = token_ids
            _, property_kbid, _ = graph_utils.edge_to_kb_ids(edge, g)
            try:
                property_kbid = property2idx.get(property_kbid, property2idx[embedding_utils.unknown])
            except:
                pdb.set_trace()    
            entity_vector = graph_utils.get_entity_indexed_vector(token_ids, edge, mode="position")
            entity_vector = [(-max_entity_index if m1 < -max_entity_index else max_entity_index if m1 > max_entity_index else m1,
                              -max_entity_index if m2 < -max_entity_index else max_entity_index if m2 > max_entity_index else m2) for _, m1,m2  in entity_vector]
            entity_matrix[index, :, :len(token_ids)] = [[position2idx[m] for m,_  in entity_vector],[position2idx[m] for _, m  in entity_vector]]
            pcnn_mask[index, 0, :len(token_ids)], pcnn_mask[index, 1, :len(token_ids)], pcnn_mask[index, 2, :len(token_ids)] = graph_utils.get_pcnn_mask(token_ids, edge)
            y_matrix[index] = property_kbid
            index += 1
            entity_pair_instance.append((pos2id[tuple(edge['left'])], pos2id[tuple(edge['right'])]))
        entity_pair += entity_pair_instance
    return [sentences_matrix, entity_matrix, y_matrix, pcnn_mask, entity_pair]

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=0)


def create_ghost_edges(sentences_matrix, entity_matrix, embeddings):
    ghost_matrix = np.zeros((entity_matrix.shape[0], 2, entity_matrix.shape[1]))
    for i in range(sentences_matrix.shape[0]):
        entity_vector = entity_matrix[i][entity_matrix[i].nonzero()]
        sentence_vector = sentences_matrix[i][sentences_matrix[i].nonzero()]
        e1_one_hot = entity_vector == 2
        e2_one_hot = entity_vector == 3
        entity_embs = np.dot(np.asarray([e1_one_hot,e2_one_hot]), embeddings[sentence_vector])

        e1_index = np.nonzero(e1_one_hot)[0]
        e2_index = np.nonzero(e2_one_hot)[0]
        entity_attention = np.dot(entity_embs, embeddings[sentence_vector].T)
        entity_attention = softmax(entity_attention.T).T

        entity_attention[:,[np.concatenate([e1_index, e2_index])]] = -np.Inf
        ghost_markers = np.tile(entity_vector, (2,1))
        ghost_markers[0][e1_index] =  1
        ghost_markers[1][e2_index] =  1
        if entity_attention.shape[-1] > 0:
            selected_entities = np.argmax(entity_attention, axis=-1)
            ghost_markers[0][selected_entities[0]] = 2
            ghost_markers[1][selected_entities[1]] = 3
            ghost_matrix[i,:,:entity_vector.shape[0]] = ghost_markers

    return ghost_matrix

def makeup_missing_edges(g):
    '''
    make up missing edges with N/A relations
    ============
    Arguments:
        - g: an instance with tokens, edgeSet, vertexSet
    Returns:
        - new_g: g with missing edges made up with N/A  
    '''

    negedges = get_all_negative_edges(g)
    full_edgeset = g['edgeSet'] + negedges
    full_edgeset = sorted(full_edgeset, key = lambda x:(x['left'], x['right']))
    new_g = g
    new_g['edgeSet'] = full_edgeset
    return new_g

def detect_bidirectional_edges(g):
    '''
    detect bidirectional edges in the data
    ==========
    Arguments:
        - g: an instance with tokens, edgeSet, vertexSet
    Returns:
        - exist: boolean value representing if there exist bidirectional or replicated edges in this instance
    '''
    cache = set()
    for i in g['edgeSet']:
        if((tuple(i['left']), tuple(i['right'])) in cache):
            return True
        else:
            cache.add((tuple(i['left']), tuple(i['right'])))
            cache.add((tuple(i['right']), tuple(i['left'])))
    return False   

def remove_replicated_vertices(g):
    '''
    remove vertices with same tokenpos in the graph 
    ===========
    Arguments:
        - g: an instance with tokens, edgeSet, vertexSet
    Returns:
        - new_g: a graph with no vertices of the same tokenpos
    '''
    new_g = {}
    new_g['tokens'] = g['tokens']
    new_g['vertexSet'] = []
    new_g['edgeSet'] = []
    tokenposSet = set()
    for i in g['vertexSet']:
        if(not tuple(i['tokenpositions']) in tokenposSet):
            tokenposSet.add(tuple(i['tokenpositions']))
            new_g['vertexSet'].append(i)
    tokenpospairSet = set()
    for i in g['edgeSet']:
        if(not (tuple(i['left']), tuple(i['right'])) in tokenpospairSet and not tuple(i['left']) == tuple(i['right'])):
            tokenpospairSet.add((tuple(i['left']), tuple(i['right'])))
            new_g['edgeSet'].append(i)       
    return new_g        
            
def add_reverse_edge(g):
    '''
    remove vertices with same tokenpos in the graph 
    ===========
    Arguments:
        - g: an instance with tokens, edgeSet, vertexSet
    Returns:
        - new_g: a graph with no vertices of the same tokenpos
    '''
    def compare(item1, item2):
        return ((item1['left'], item1['right']) < (item2['left'], item2['right']))
    new_g = {}
    new_g['tokens'] = g['tokens']
    new_g['vertexSet'] = g['vertexSet']
    new_g['edgeSet'] = []
    tokenpospairSet = set()
    for i in g['edgeSet']:
        j = dict(i)
        new_g['edgeSet'].append(i) 
        if(i['kbID'] != "P0"):
            j['kbID'] = "~" + i['kbID']
        tmp = j['left']
        j['left'] = j['right']
        j['right'] = tmp
        new_g['edgeSet'].append(j)
    # pdb.set_trace()
    # new_g['vertexSet'] = sorted(new_g['vertexSet'])
    new_g['edgeSet'] = sorted(new_g['edgeSet'], key=lambda x : (x['left'], x['right']))               
    return new_g  

if __name__ == "__main__":
    # Testing
    import doctest
    print(doctest.testmod())
