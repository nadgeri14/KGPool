import numpy as np
import json
from nltk.tokenize import word_tokenize
import utils
from collections import OrderedDict
import tqdm

unknown = "_UNKNOWN"
MAX_SENT_LEN = 32
MAX_CHAR_LEN = 10
CONV_FILTER_SIZE = 3
MAX_INSTANCE = 5 #increase depending on memory 
MAX_ALIAS = 5 #increase depending on memory 

train_path = '../../NYT/dataset_triples_train.json' 
test_path = '../../NYT/dataset_triples_test.json' 
entity_detials_path = '../../NYT/dataset_context_en_all_TOKENS.json'
relation_idx_path = '../../NYT/relation2id.txt'

def make_char_vocab(data):
    char_vocab = OrderedDict()
    char_idx = 2
    data = data['data']
    for d in data:
        sentence = d['sent']
        for c in sentence:
            if c not in char_vocab:
                char_vocab[c] = char_idx
                char_idx += 1

    chars = {c for c, _ in char_vocab.items()}
    chars = sorted(chars)
    char2idx = {'<PAD>': 0, '<UNK>':1}
    char2idx.update({el: idx for idx, el in enumerate(chars, start=len(char2idx))})

    return char2idx

def load_embeddings(path = '../../NYT/glove.6B/glove.6B.300d.txt'):
    """
    Loads pre-trained embeddings from the specified path.
    
    @return (embeddings as an numpy array, word to index dictionary)
    """
    word2idx = {}  # Maps a word to the index in the embeddings matrix
    embeddings = []

    with open(path, 'r', encoding='utf8') as fIn:
        idx = 1               
        for line in fIn:
            split = line.strip().split(' ')                
            embeddings.append(np.array([float(num) for num in split[1:]]))
            word2idx[split[0]] = idx
            idx += 1
    
    word2idx['<PAD>'] = 0
    embedding_size = embeddings[0].shape[0]
    print("Emb. size: {}".format(embedding_size))
    embeddings = np.asarray([[0.0]*embedding_size] + embeddings, dtype='float32')
    
    rare_w_ids = list(range(idx-101,idx-1))
    unknown_emb = np.average(embeddings[rare_w_ids,:], axis=0)
    embeddings = np.append(embeddings, [unknown_emb], axis=0)
    word2idx[unknown] = idx
    idx += 1

    print("Loaded: {}".format(embeddings.shape))
    
    return embeddings, word2idx

def find_vertex(vertexs, kbID):
    for vertex in vertexs:
        if(vertex['kbID'] == kbID):
            return vertex
    return None

def find_desc(vertex, entities_context, sentence_tokens):
    if(vertex == None):
        return ""
    qid = vertex['kbID']
    if(qid in entities_context):
        description = entities_context[qid]['desc']
        if(description == "" or description == " " or description == []):
            description = word_tokenize(vertex['lexicalInput'])
    else:
        description = word_tokenize(vertex['lexicalInput'])
    return description

def find_triples(vertex, triple_file):
    qid = vertex['kbID']
    if(qid in triple_file):
        triples = triple_file[qid]
    else:
        triples = []
    return triples

def find_instancesof(vertex, entities_context):
    if(vertex == None):
        return []
    instancesof = []
    qid = vertex['kbID']
    if(qid in entities_context):
        instancesof = entities_context[qid]['en_instances']
    return instancesof[:MAX_INSTANCE]

def find_alias(vertex, entities_context):
    if(vertex == None):
        return []
    alias = []
    qid = vertex['kbID']
    if(qid in entities_context):
        alias = entities_context[qid]['alias']
    return alias[:MAX_ALIAS]

def find_surfaceform(vertex):
    if(vertex == None):
        return ""
    return word_tokenize(vertex['lexicalInput'])

def get_word_indices(max_sent_len, word_sequence, word2idx):
    token_ids = np.ones((max_sent_len))*word2idx['<PAD>']
    for i, word in enumerate(word_sequence):
        if i>=max_sent_len:
          break        
        token_ids[i] = word2idx.get(word, word2idx['_UNKNOWN'])

    return token_ids

def get_char_indices(tokens, max_sent_len, max_char_len, char2idx, conv_filter_size):
    char_indices = np.ones( (
      conv_filter_size-1 + max_sent_len*(max_char_len+conv_filter_size-1) ) )
    char_indices = char_indices*char2idx['<PAD>']

    words = tokens[:max_sent_len]
    cur_idx = conv_filter_size - 1

    for index_word, word in enumerate(words):
        for index_char, c in enumerate(word[0:min(len(word), max_char_len)]): 
            if char2idx.get(c,None) is not None:
                char_indices[cur_idx+index_char] = char2idx[c]
            else:
                char_indices[cur_idx+index_char] = char2idx['<UNK>']
        cur_idx += max_char_len + conv_filter_size-1

    return char_indices

def create_dataset(data_array, entity_file, rel2idx):
    dataset_words = []
    dataset_chars = []
    num_nodes = []
    edge_indices_r1 = []
    edge_indices_r2 = []
    num_edges = []
    labels = []
    count = 0
    entity_nodes = []
    sentence_nodes = []
    data_list = []
    indices_list = []
    row = len(data_array)
    col = len(data_array[0])
    for i in range(row):
        for j in range(col):
            if(data_array[i][j]!= None):
                data_list.append(data_array[i][j])
                indices_list.append((i,j))
    for data in data_list:
        graph = data['graph']
        left_entity = data['left_entity']
        relation = data['relation']
        sentence_tokens = graph['tokens']
        edges = graph['edgeSet']
        vertexs = graph['vertexSet']

        left_vertex = find_vertex(vertexs, left_entity)
        right_vertex = find_vertex(vertexs, right_entity)

        left_desc = find_desc(left_vertex, entity_file, sentence_tokens)
        right_desc = find_desc(right_vertex, entity_file, sentence_tokens)

        left_surfaceform = find_surfaceform(left_vertex)
        right_surfaceform = find_surfaceform(right_vertex)

        left_instancesof = find_instancesof(left_vertex, entity_file)
        right_instancesof = find_instancesof(right_vertex, entity_file)

        left_alias = find_alias(left_vertex, entity_file)
        right_alias = find_alias(right_vertex, entity_file)
 
        cur_num_nodes = 5 + len(left_instancesof) + len(right_instancesof) + len(left_alias) + len(right_alias) 

        # add egde between left entity and left desc nodes
        list1 = []
        list2 = []
        list1.append(0)
        list2.append(1)
        # add egde between left entity and left instances nodes
        n1 = 2+len(left_instancesof)
        for i in range(2,n1):
            list1.append(0)
            list2.append(i)
        # add egde between left entity and left alias nodes
        n2 = n1 + len(left_alias)
        for i in range(n1,n2):
            list1.append(0)
            list2.append(i)
        #add edge between left entity and left triples 
        n21 = n2 

        # add egde between left entity and sentence node
        list1.append(0)
        list2.append(n21)
        # add egde between right entity and sentence nodes
        n3 = n21 + 1
        list1.append(n3)
        list2.append(n21)
        # add egde between right entity and right desc nodes
        list1.append(n3)
        list2.append(n3+1)
        # add egde between right entity and right instance nodes
        n4 = n3 + 2 + len(right_instancesof)
        for i in range(n3+2,n4):
            list1.append(n3)
            list2.append(i)
        # add egde between right entity and right alias nodes
        n5 = n4 + len(right_alias)
        for i in range(n4,n5):
            list1.append(n3)
            list2.append(i)

        cur_edges = np.asarray([list1,list2], dtype=np.float)

        left_desc_tokens = left_desc
        right_desc_tokens = right_desc
        left_surfaceform_tokens = []
        right_surfaceform_tokens = []
        if(left_vertex != None):
            for i in left_vertex['tokenpositions']:
                left_surfaceform_tokens.append(sentence_tokens[i])

        if(right_vertex != None):        
            for i in right_vertex['tokenpositions']:
                right_surfaceform_tokens.append(sentence_tokens[i])
        left_instancesof_tokens = []
        for instanceof in left_instancesof:
          left_instancesof_tokens.append(instanceof)
        right_instancesof_tokens = []
        for instanceof in right_instancesof:
          right_instancesof_tokens.append(instanceof)
        left_alias_tokens = []
        for alias in left_alias:
          left_alias_tokens.append(alias)
        right_alias_tokens = []
        for alias in right_alias:
          right_alias_tokens.append(alias)

        dataset_words.extend([left_surfaceform_tokens,left_desc_tokens])
        dataset_words.extend(left_instancesof_tokens)
        dataset_words.extend(left_alias_tokens)

        dataset_words.extend([sentence_tokens,right_surfaceform_tokens,right_desc_tokens])
        dataset_words.extend(right_instancesof_tokens)
        dataset_words.extend(right_alias_tokens)
        
        
        num_nodes.append(cur_num_nodes)
        num_edges.append(cur_edges.shape[1])

        edge_indices_r1 += list1
        edge_indices_r2 += list2

        labels.append(rel2idx[relation])

        #Store all the entity node positions
        entity_nodes.append([0,n3])
        sentence_nodes.append([n21])

    edge_indices = np.asarray([edge_indices_r1,edge_indices_r2], dtype=np.float)
    entity_sent_nodes = [entity_nodes,sentence_nodes]
    return dataset_words, dataset_chars, num_nodes, edge_indices, num_edges, labels, entity_sent_nodes

def load_static_data(load_char_vocab=False,char_vocab_path='char_vocab.json'):
    with open(train_path) as f:
        train_file = json.load(f)
    with open(entity_detials_path) as f:
        entity_file = json.load(f)

    if load_char_vocab:
        with open(char_vocab_path, 'r') as f:
            char2idx = json.load(f)
    else:
        char2idx = make_char_vocab(train_file)
        with open(char_vocab_path, 'w') as f:
            json.dump(char2idx, f, indent=True)

    relation_idx_file = open(relation_idx_path, 'r') 
    data = relation_idx_file.readlines()
    rel2idx = {}
    for mapping in data:
        key = mapping.split(" ")[0]
        val = int(mapping.split(" ")[1])
        rel2idx[key] = val

    return train_file, entity_file, char2idx, rel2idx


def load_dataset(data, train_file, entity_file, relation_file = None, load_char_vocab=False, char_vocab_path='char_vocab.json', max_sent_len=MAX_SENT_LEN, max_char_len=MAX_CHAR_LEN, conv_filter_size=CONV_FILTER_SIZE):

    train_data = create_dataset(data, entity_file, relation_file)#relation_file = rel2idx

    return train_data

def data_generator(data, batch_size=32, shuffle=True, max_sent_len=MAX_SENT_LEN, max_char_len=MAX_CHAR_LEN, conv_filter_size=CONV_FILTER_SIZE, word2idx=None, char2idx=None):
  indices = np.arange(len(data[-2]))
  if shuffle:
    np.random.shuffle(indices)
  num_iters = int(np.ceil(len(data[-2])/batch_size))
  np_data = [[],[]]
  for i in range(2,len(data) - 1):
    np_data.append(np.array(data[i]))
  
  entity_node_indexs = data[-1][0]
  sent_node_indexs = data[-1][1]
  entity_node_indexs =  np.array(entity_node_indexs)
  sent_node_indexs = np.array(sent_node_indexs)

  cum_nodes_sum = np.cumsum(np_data[2])
  cum_nodes_sum = np.append([0],cum_nodes_sum,axis=0)
  cum_edges_sum = np.cumsum(np_data[4])
  cum_edges_sum = np.append([0],cum_edges_sum,axis=0)
  for i in range(num_iters):
    start_idx = i*batch_size
    end_idx = min(len(data[-2]),(i+1)*batch_size)

    batch_indices = indices[start_idx:end_idx]

    num_nodes = np_data[2][batch_indices]
    num_edges = np_data[4][batch_indices]
    edges = []
    dataset_words = []
    dataset_chars = []
    for j in batch_indices:
      edges.append(np_data[3][:,cum_edges_sum[j]:cum_edges_sum[j+1]])

      for idx in range(cum_nodes_sum[j],cum_nodes_sum[j+1]):
        tokens_indices = get_word_indices(max_sent_len,data[0][idx],word2idx)
        dataset_words.append(tokens_indices)

        tokens_char_indices = get_char_indices(data[0][idx], max_sent_len, max_char_len, char2idx, conv_filter_size)
        dataset_chars.append(tokens_char_indices)

    word_indices = np.array(dataset_words)
    char_indices = np.array(dataset_chars)

    labels = np_data[-1][batch_indices]
    e_node_indices = entity_node_indexs[batch_indices]
    s_node_indices = sent_node_indexs[batch_indices]

    batches = np.empty((0))
    batch_edges = np.empty((2,0))
    offset = 0
    for j in range(end_idx-start_idx):
      e_node_indices[j] += offset
      s_node_indices[j] += offset
      cur_offset_edges = edges[j] + offset
      batch_edges = np.append(batch_edges,cur_offset_edges,axis=1)
      batches = np.append(batches,np.ones((num_nodes[j]))*j)
      offset = offset + num_nodes[j]

    return np.array(word_indices).astype('long'), np.array(char_indices).astype('long'), batch_edges.astype('long'), batches.astype('long'), e_node_indices, s_node_indices, labels
