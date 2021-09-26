# coding: utf-8
# Copyright (C) 2017 Hao Zhu
#
# Author: Hao Zhu (ProKil.github.io)
#

from utils import evaluation_utils, embedding_utils
from semanticgraph import io
from parsing import legacy_sp_models as sp_models
from models import baselines
import numpy as np
from sacred import Experiment
import json
import torch
from torch import nn
from torch.autograd import Variable
from tqdm import *
import ast
from models.factory import get_model
import argparse

import sys
sys.path.insert(1, '../KGPool')
import data_loader

from sklearn.metrics import precision_recall_curve
import torch.nn.functional as F
try:
    from functools import reduce
except:
    pass

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152

ex = Experiment("test")

np.random.seed(1)

p0_index = 1

def to_np(x):
    return x.data.cpu().numpy()

@ex.config
def main_config():
    """ Main Configurations """
    model_name = "GPGNN"
    load_model = "../models/model.out" # you should choose the proper model to load
    device_id = 0

    data_folder = "../NYT/"
    save_folder = "../NYT/"
    result_folder = "../result/"
    model_params = "model_params.json"
    word_embeddings = "../data/glove.6B.50d.txt"

    test_set = "NYT_validation.json"
    # a file to store property2idx
    # if is None use model_name.property2idx
    property_index = None

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)


    kgpool_parser = argparse.ArgumentParser()

    kgpool_parser.add_argument('--nhid', type=int, default=128,
                        help='hidden size')
    kgpool_parser.add_argument('--pooling_ratio', type=float, default=1.00,
                        help='pooling ratio')
    kgpool_parser.add_argument('--dynamic_pooling1', type=float, default=False,
                        help='keep dynamic pooling for layer 1')
    kgpool_parser.add_argument('--dynamic_pooling2', type=float, default=False,
                        help='keep dynamic pooling for layer 2')
    kgpool_parser.add_argument('--dynamic_pooling3', type=float, default=True,
                        help='keep dynamic pooling for layer 3') 
    
    kgpool_parser.add_argument('--input_dim', type=int, default=50, help='The input word embedding dimension')
    kgpool_parser.add_argument('--hidden_dim', type=int, default=50, help='The RNN hidden dimension')
    kgpool_parser.add_argument('--layers', type=int, default=1, help='Number of RNN layers')
    kgpool_parser.add_argument('--is_bidirectional', type=bool, default=True, help='Whether the RNN is bidirectional')
    kgpool_parser.add_argument('--drop_out_rate', type=float, default=0.5, help='The dropout probability')
    kgpool_parser.add_argument('--entity_embed_dim', type=int, default=100, help='The output embedding dimension')
    kgpool_parser.add_argument('--conv_filter_size', type=int, default=3, help='The size of the convolution filters for character encoding')
    kgpool_parser.add_argument('--char_embed_dim', type=int, default=50, help='The character embedding dimension')
    kgpool_parser.add_argument('--max_word_len_entity', type=int, default=10, help='The max number of characters per word')
    kgpool_parser.add_argument('--char_feature_size', type=int, default=50, help='The feature dimension of the character convolution')
    
    kgpool_parser.add_argument('--load_char_vocab', type=bool, default=False,
                        help='Whether to load the char vocab from file')
    kgpool_parser.add_argument('--char_vocab_path', type=str, default='char_vocab.json',
                        help='The path to (re)store the char vocab')
    kgpool_parser.add_argument('--embeddings_path', type=str, default='../data/glove.6B.50d.txt',
                        help='The path to (re)store the char vocab')
    kgpool_args = kgpool_parser.parse_args()



@ex.automain
def main(model_params, model_name, data_folder, word_embeddings, test_set, property_index, save_folder, load_model, result_folder, kgpool_args):

    test_epoch = 1
    
    with open(model_params) as f:
        model_params = json.load(f)

    embeddings, word2idx = embedding_utils.load(word_embeddings)
    print("Loaded embeddings:", embeddings.shape)

    
    test_file, entity_file, char2idx, rel2idx = data_loader.load_static_data()
    kgpool_args.num_classes = len(rel2idx)
    kgpool_args.num_features = kgpool_args.entity_embed_dim

    def check_data(data):
        for g in data:
            if(not 'vertexSet' in g):
                print("vertexSet missed\n")


    print("Reading the property index")
    with open(data_folder + "models/" + model_name + ".property2idx") as f:
        property2idx = ast.literal_eval(f.read())
    idx2property = { v:k for k,v in property2idx.items() }

    test_file, entity_file, char2idx, rel2idx = data_loader.load_static_data()
    kgpool_args.num_classes = len(rel2idx)

    max_sent_len = 36
    print("Max sentence length set to: {}".format(max_sent_len))

    graphs_to_indices = sp_models.to_indices_and_entity_pair
    if model_name == "ContextAware":
        graphs_to_indices = sp_models.to_indices_with_real_entities_and_entity_nums_with_vertex_padding_and_entity_pair
    elif model_name == "PCNN":
        graphs_to_indices = sp_models.to_indices_with_relative_positions_and_pcnn_mask_and_entity_pair  
    elif model_name == "CNN":
        graphs_to_indices = sp_models.to_indices_with_relative_positions_and_entity_pair
    elif model_name == "GPGNN":
        graphs_to_indices = sp_models.to_indices_with_real_entities_and_entity_nums_with_vertex_padding

    _, position2idx = embedding_utils.init_random(np.arange(-max_sent_len, max_sent_len), 1, add_all_zeroes=True)

    training_data = None

    n_out = len(property2idx)
    print("N_out:", n_out)

    model = get_model(model_name)(model_params, embeddings, max_sent_len, n_out, kgpool_args, char2idx).cuda()
    model.load_state_dict(torch.load(load_model))
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("model params:",pytorch_total_params)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable model params : ",pytorch_total_params)

    test_set, _ = io.load_relation_graphs_from_file(data_folder + test_set, data='NYT')
    test_as_indices = list(graphs_to_indices(test_set, word2idx, property2idx, max_sent_len, embeddings=embeddings, position2idx=position2idx))
    
    print("Start testing!")
    result_file = open("../result/" + "_" + model_name, "w")
    test_f1 = 0
    scores = []
    y_hot = []
    for i in tqdm(range(int(test_as_indices[0].shape[0] / model_params['batch_size']))):
    #for i in tqdm(range(5))
        sentence_input = test_as_indices[0][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]
        entity_markers = test_as_indices[1][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]
        labels = test_as_indices[2][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]
        kbIDs = test_as_indices[4][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]
        
       kg_pool_test_data = data_loader.load_dataset(data= kbIDs, train_file= test_file, entity_file=entity_file, relation_file = rel2idx)
        batch_sizes = len(kg_pool_test_data[-2])
       kg_pool_test_data_loader = data_loader.data_generator(kg_pool_test_data, batch_sizes, shuffle=False, word2idx=word2idx, char2idx= char2idx)
       kg_pool_test_data_loader = list(kg_pool_test_data_loader)
        for j in range(len(kg_pool_test_data_loader)):
            if (torch.cuda.is_available()):
               kg_pool_test_data_loader[j] = torch.tensor(kg_pool_test_data_loader[j]).cuda()
            else:
               kg_pool_test_data_loader[j] = torch.tensor(kg_pool_test_data_loader[j])
        
        if model_name == "GPGNN":
            output = model(Variable(torch.from_numpy(sentence_input.astype(int)), volatile=True).cuda(),
                            Variable(torch.from_numpy(entity_markers.astype(int)), volatile=True).cuda(),
                            test_as_indices[3][i * model_params['batch_size']: (i + 1) * model_params['batch_size']],
                           kg_pool_test_data_loader,kbIDs,test_epoch)
        elif model_name == "PCNN":
            output = model(Variable(torch.from_numpy(sentence_input.astype(int)), volatile=True), 
                            Variable(torch.from_numpy(entity_markers.astype(int)), volatile=True), 
                            Variable(torch.from_numpy(np.array(test_as_indices[3][i * model_params['batch_size']: (i + 1) * model_params['batch_size']])).float(), requires_grad=False, volatile=True))        
        else:
            output = model(Variable(torch.from_numpy(sentence_input.astype(int)), volatile=True),
                            Variable(torch.from_numpy(entity_markers.astype(int)), volatile=True))


        _, predicted = torch.max(output, dim=1)
        labels_copy = labels.reshape(-1).tolist()
        predicted = predicted.data.tolist()
        p_indices = np.array(labels_copy) != 0
        predicted = np.array(predicted)[p_indices].tolist()
        labels_copy = np.array(labels_copy)[p_indices].tolist()

        _, _, add_f1 = evaluation_utils.evaluate_instance_based(
            predicted, labels_copy, empty_label=p0_index)
        test_f1 += add_f1

        score = F.softmax(output)
        score = to_np(score).reshape(-1, n_out)
        labels = labels.reshape(-1)
        p_indices = labels != 0
        score = score[p_indices].tolist()
        labels = labels[p_indices].tolist()
        scores.extend(score)
        cur_one_hot = np.zeros((len(labels),n_out))
        for i, l in enumerate(labels):
            cur_one_hot[i,l] = 1
        y_hot.extend(cur_one_hot)

        pred_labels = r = np.argmax(score, axis=-1)
        indices = [i for i in range(len(p_indices)) if p_indices[i]]
        start_idx = i * model_params['batch_size']
        for index, (i, j) in enumerate(zip(score, labels)):
            sent = ' '.join(test_set[ start_idx + indices[index]//(model_params['max_num_nodes']*(model_params['max_num_nodes']-1)) ]['tokens']).strip()
            result_file.write("{} | {} | {} | {} \n".format(sent, idx2property[pred_labels[index]], idx2property[labels[index]], score[index][pred_labels[index]]))

    result_file.close()
    print("Test f1: ", test_f1 /
            (test_as_indices[0].shape[0] / model_params['batch_size']))