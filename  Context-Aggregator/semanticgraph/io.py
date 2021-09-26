
# IO for graphs of relations

import json
from nltk import word_tokenize
import re

def get_tokenpositions_from_sentence(sent_tokens, sent, rsent, entity):
    # i = 0
    # tokenpositions = []
    # lsent = sent.lower()
    # start_idx = lsent.find(entity.replace('_',' '))
    # if start_idx>-1:
    #   end_idx = start_idx + len(entity)
    #   entity_sent = sent[start_idx:end_idx]
    #   entity_tokens = [entity_word.strip().lower() for entity_word in word_tokenize(entity_sent)]
    # else:
    #   entity_tokens = [entity_word.strip() for entity_word in entity.split('_')]
    # for j, word in enumerate(sent_tokens):
    #   if entity_tokens[i]==word.lower():
    #     tokenpositions.append(j)
    #     i += 1
    #     if i>=len(entity_tokens):
    #       break

    # if i<len(entity_tokens):
    #   import pdb; pdb.set_trace()
    #   raise("Token match not found for {} and token {}".format(rsent,entity))
    lsent = sent.lower()
    max_attempts = len(re.findall('{}'.format(entity.replace('_',' ').lower()),lsent))
    start = 0
    attempt = 0
    while attempt<max_attempts:
        tokenpositions = []
        match = re.search(r'{}'.format(entity.replace('_',' ').lower()),lsent[start:])
        if match:
            start_idx = match.span()[0] + start
            end_idx = match.span()[1] + start
            left_sent_nospace = lsent[0:start_idx].replace(' ','')
            right_sent_nospace = lsent[0:end_idx].replace(' ','')
            start_pos = len(left_sent_nospace)
            end_pos = len(right_sent_nospace)
            next_pos = 0
            i = 0
            for j, word in enumerate(sent_tokens):
                if next_pos>=start_pos and next_pos<end_pos:
                    tokenpositions.append(j)
                if next_pos>=end_pos:
                    break
                next_pos += len(word)

            # print(entity)
            # print(sent_tokens)
            # print(tokenpositions)
            # print(''.join([sent_tokens[k] for k in tokenpositions]).lower(),entity.lower().replace('_',''))
            # print()

            start = end_idx
            attempt += 1
            if ''.join([sent_tokens[k] for k in tokenpositions]).lower()!=entity.lower().replace('_',''):
                continue
            else:
                return tokenpositions
        else:
            raise("Exception no match")     
    raise("Exception match but no match")
    return tokenpositions

def load_relation_graphs_from_files(json_files, val_portion=0.0, load_vertices=True, data = 'wikidata'):
    """
    Load semantic graphs from multiple json files and if specified reserve a portion of the data for validation.

    :param json_files: list of input json files
    :param val_portion: a portion of the data to reserve for validation
    :return: a tuple of the data and validation data
    """
    if(data == 'wikidata'):
        data = []
        for json_file in json_files:
            with open(json_file) as f:
                if load_vertices:
                    data = data + json.load(f)
                else:
                    data = data + json.load(f, object_hook=dict_to_graph_with_no_vertices)
        print("Loaded data size:", len(data))
    
        val_data = []
        if val_portion > 0.0:
            val_size = int(len(data)*val_portion)
            rest_size = len(data) - val_size
            val_data = data[rest_size:]
            data = data[:rest_size]
            print("Training and dev set sizes:", (len(data), len(val_data)))

    elif(data=='NYT'):
        data = []
        compatible_data = []
        for json_file in json_files:
            with open(json_file) as f:
                if load_vertices:
                    data = data + json.load(f)['data']
                else:
                    data = data + json.load(f, object_hook=dict_to_graph_with_no_vertices)['data']

                for d in data:
                    tokens = word_tokenize(d['sent'])
                    tokenpositions_sub = get_tokenpositions_from_sentence(tokens,d['sent'],d['rsent'],d['sub'])
                    tokenpositions_obj = get_tokenpositions_from_sentence(tokens,d['sent'],d['rsent'],d['obj'])
                    vertexSet = [\
                    {"kbID": d['sub_id'], "tokenpositions": tokenpositions_sub, "lexicalInput": d['sub'].replace('_',' ')},\
                    {"kbID": d['obj_id'], "tokenpositions": tokenpositions_obj, "lexicalInput": d['obj'].replace('_',' ')}\
                    ]
                    edgeSet = [
                    {"kbID": d['rel'], "right": tokenpositions_obj, "left": tokenpositions_sub}
                    ]
                    compatible_data.append({
                        'tokens': tokens,
                        'vertexSet': vertexSet,
                        'edgeSet': edgeSet
                    })

        data = compatible_data
        print("Loaded data size:", len(data))

        val_data = []
        if val_portion > 0.0:
            val_size = int(len(data)*val_portion)
            rest_size = len(data) - val_size
            val_data = data[rest_size:]
            data = data[:rest_size]
            print("Training and dev set sizes:", (len(data), len(val_data)))
    else:
        raise("Expected data to be one of {}".format(['wikidata','NYT']))
    return data, val_data


def load_relation_graphs_from_file(json_file, val_portion=0.0, load_vertices=True, data='wikidata'):
    """
    Load semantic graphs from a json file and if specified reserve a portion of the data for validation.

    :param json_file: the input json file
    :param val_portion: a portion of the data to reserve for validation
    :return: a tuple of the data and validation data
    """
    return load_relation_graphs_from_files([json_file], val_portion, load_vertices, data = data)


def dict_to_graph_with_no_vertices(d):
    if 'vertexSet' in d:
        del d['vertexSet']
    return d
