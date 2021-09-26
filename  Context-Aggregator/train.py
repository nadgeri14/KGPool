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
    device_id = 0
    # 
    model_name = "GPGNN"
    data_folder = "../realtion_extraction/NYT/"
    save_folder = "../realtion_extraction/NYT/save_folder"
    load_folder = "../realtion_extraction/NYT/kgppool_gpgnn/"
    load_model = "model-80.out"
    model_params = "model_params.json"
    word_embeddings = "../data/glove.6B.50d.txt"
    train_set = "dataset_triples_train.json"
    val_set = ""
    test_set = "dataset_triples_test.json"

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



    # a file to store property2idx
    # if is None use model_name.property2idx
    property_index = None
    learning_rate = 1e-3
    shuffle_data = True
    save_model = True
    grad_clip = 0.25
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)


@ex.automain
def main(model_params, model_name, data_folder, word_embeddings, train_set, val_set, test_set, property_index, learning_rate, shuffle_data, save_folder, load_folder, load_model, save_model, grad_clip, kgpool_args):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    with open(model_params) as f:
        model_params = json.load(f)

    embeddings, word2idx = embedding_utils.load(word_embeddings)
    print("Loaded embeddings:", embeddings.shape)

    
    train_file, entity_file, char2idx, rel2idx = data_loader.load_static_data()
    kgpool_args.num_classes = len(rel2idx)
    kgpool_args.num_features = kgpool_args.entity_embed_dim
    

    def check_data(data):
        for g in data:
            if(not 'vertexSet' in g):
                print("vertexSet missed\n")

    training_data, _ = io.load_relation_graphs_from_file(data_folder + train_set, load_vertices=True, data='NYT')

    check_data(training_data)
    check_data(test_set)

    if property_index:
        print("Reading the property index from parameter")
        with open(data_folder + args.property_index) as f:
            property2idx = ast.literal_eval(f.read())
    else:
        _, property2idx = embedding_utils.init_random({e["kbID"] for g in training_data
                                                    for e in g["edgeSet"]} | {"P0"}, 1, add_all_zeroes=True, add_unknown=True)
    
    max_sent_len = max(len(g["tokens"]) for g in training_data)
    print("Max sentence length:", max_sent_len)

    max_sent_len = 48
    print("Max sentence length set to: {}".format(max_sent_len))

    graphs_to_indices = sp_models.to_indices
    if model_name == "ContextAware":
        graphs_to_indices = sp_models.to_indices_with_real_entities_and_entity_nums_with_vertex_padding
    elif model_name == "PCNN":
        graphs_to_indices = sp_models.to_indices_with_relative_positions_and_pcnn_mask   
    elif model_name == "CNN":
        graphs_to_indices = sp_models.to_indices_with_relative_positions
    elif model_name == "GPGNN":
        graphs_to_indices = sp_models.to_indices_with_real_entities_and_entity_nums_with_vertex_padding

    _, position2idx = embedding_utils.init_random(np.arange(-max_sent_len, max_sent_len), 1, add_all_zeroes=True)

    train_as_indices = list(graphs_to_indices(training_data, word2idx, property2idx, max_sent_len, embeddings=embeddings, position2idx=position2idx))

    training_data = None

    n_out = len(property2idx)
    print("N_out:", n_out)

    test_as_indices = list(graphs_to_indices(test_set, word2idx, property2idx, max_sent_len, embeddings=embeddings, position2idx=position2idx))
    test_set = None

    print("Save property dictionary.")
    with open(data_folder + "models/" + model_name + ".property2idx", 'w') as outfile:
        outfile.write(str(property2idx))

    print("Training the model")

    print("Initialize the model")
    if (torch.cuda.is_available()):
        model = get_model(model_name)(model_params, embeddings, max_sent_len, n_out, kgpool_args, char2idx).cuda()

    else:
        model = get_model(model_name)(model_params, embeddings, max_sent_len, n_out, kgpool_args, char2idx)

    if (torch.cuda.is_available()):
        loss_func = nn.CrossEntropyLoss(ignore_index=0).cuda()
    else:
        loss_func = nn.CrossEntropyLoss(ignore_index=0)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=model_params['weight_decay'])

    indices = np.arange(train_as_indices[0].shape[0])

    step = 0
    for train_epoch in range(model_params['nb_epoch']):

        if(shuffle_data):
            np.random.shuffle(indices)
        f1 = 0
        for i in tqdm(range(int(train_as_indices[0].shape[0] / model_params['batch_size']))):
            opt.zero_grad()

            sentence_input = train_as_indices[0][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]]
            entity_markers = train_as_indices[1][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]]
            labels = train_as_indices[2][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]]
            kbIDs = train_as_indices[4][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]]

            #data for kgpool
            kgpool_train_data = data_loader.load_dataset(data= kbIDs, train_file= train_file, entity_file=entity_file, relation_file = rel2idx)
            batch_sizes = len(kgpool_train_data[-2])
            kgpool_train_data_loader = data_loader.data_generator(kgpool_train_data, batch_sizes, shuffle=False, word2idx=word2idx, char2idx= char2idx)
            kgpool_train_data_loader = list(kgpool_train_data_loader)
            for j in range(len(kgpool_train_data_loader)):
                if (torch.cuda.is_available()):
                    kgpool_train_data_loader[j] = torch.tensor(kgpool_train_data_loader[j]).cuda()
                else:
                    kgpool_train_data_loader[j] = torch.tensor(kgpool_train_data_loader[j])
            
            if model_name == "GPGNN":
                if (torch.cuda.is_available()):
                    output = model(Variable(torch.from_numpy(sentence_input.astype(int))).cuda(), 
                                    Variable(torch.from_numpy(entity_markers.astype(int))).cuda(), 
                                    train_as_indices[3][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]],
                                    kgpool_train_data_loader,kbIDs, train_epoch)
                else:
                    output = model(Variable(torch.from_numpy(sentence_input.astype(int))), 
                                    Variable(torch.from_numpy(entity_markers.astype(int))), 
                                    train_as_indices[3][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]],
                                    kgpool_train_data_loader,kbIDs)
            elif model_name == "PCNN":
                output = model(Variable(torch.from_numpy(sentence_input.astype(int))).cuda(), 
                                Variable(torch.from_numpy(entity_markers.astype(int))).cuda(), 
                                Variable(torch.from_numpy(np.array(train_as_indices[3][i * model_params['batch_size']: (i + 1) * model_params['batch_size']])).float(), requires_grad=False).cuda())
            else:
                output = model(Variable(torch.from_numpy(sentence_input.astype(int))).cuda(),
                                Variable(torch.from_numpy(entity_markers.astype(int))).cuda())

            if (torch.cuda.is_available()):
                loss = loss_func(output, Variable(torch.from_numpy(labels.astype(int))).view(-1).cuda())
            else:
                loss = loss_func(output, Variable(torch.from_numpy(labels.astype(int))).view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), grad_clip)
            opt.step()

            _, predicted = torch.max(output, dim=1)
            labels = labels.reshape(-1).tolist()
            predicted = predicted.data.tolist()
            p_indices = np.array(labels) != 0
            predicted = np.array(predicted)[p_indices].tolist()
            labels = np.array(labels)[p_indices].tolist()

            _, _, add_f1 = evaluation_utils.evaluate_instance_based(predicted, labels, empty_label=p0_index)
            f1 += add_f1
            

        print("Train f1: ", f1 / (train_as_indices[0].shape[0] / model_params['batch_size']))
