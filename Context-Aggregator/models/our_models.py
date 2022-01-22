import torch
from torch import nn
from torch.autograd import Variable
import sys
from parsing.legacy_sp_models import MAX_EDGES_PER_GRAPH
from utils.build_adjecent_matrix import adjecent_matrix
from models.layers import GraphConvolution
from utils.embedding_utils import make_start_embedding, get_head_indices, get_tail_indices
import torch.nn.functional as F

import sys
sys.path.insert(1,'../../KGPool')
from networks import  Net


class GPGNN(nn.Module):
    """
    Our model
    """

    def __init__(self, p, embeddings, max_sent_len, n_out, kgpool_args, kgpool_char2idx):
        super(GPGNN, self).__init__()

        print("Parameters:", p)
        self.p = p

        # Input shape: (max_sent_len,)
        # Input type: int
        self.max_sent_len = max_sent_len

        self.word_embedding = nn.Embedding(
            embeddings.shape[0], embeddings.shape[1], padding_idx=0)
        self.word_embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.word_embedding.weight.requires_grad = False

        self.dropout1 = nn.Dropout(p=p['dropout1'])

        self.pos_embedding = nn.Embedding(4, p['position_emb'], padding_idx=0)
        nn.init.orthogonal_(self.pos_embedding.weight)

        # Merge word and position embeddings and apply the specified amount of RNN layers
        self.rnn1 = nn.LSTM(batch_first=True, input_size=embeddings.shape[1] + int(p['position_emb']),
                                           hidden_size=int(p['units1']),
                                           num_layers=int(p['rnn1_layers']), bidirectional=bool(p['bidirectional']))

        for parameter in self.rnn1.parameters():
            if(len(parameter.size()) >= 2):
                nn.init.orthogonal_(parameter)

        self.dropout2 = nn.Dropout(p=p['dropout1'])

        if(p['layer_number'] == 1 or p['projection_style'] == 'tie'):
            self.representation_to_adj = nn.Linear(
                in_features=p['units1'] * 2, out_features=(p['embedding_dim'] * 2) ** 2)
            nn.init.xavier_uniform_(self.representation_to_adj.weight)
        else:
            self.representation_to_adj = nn.ModuleList([nn.Linear(
                in_features=p['units1'] * 2, out_features=(p['embedding_dim'] * 2) ** 2) for i in range(p['layer_number'])])
            for i in self.representation_to_adj:
                nn.init.xavier_uniform_(i.weight)
        
        self.identity_transformation = nn.Parameter(
            torch.eye(p['embedding_dim'] * 2), requires_grad=True)
        
        self.start_embedding = nn.Parameter(torch.from_numpy(
            make_start_embedding(9, p['embedding_dim'])).float(), requires_grad=False)
        
        self.head_indices = nn.Parameter(torch.LongTensor(
            get_head_indices(9, p['embedding_dim'])), requires_grad=False)
        
        self.tail_indices = nn.Parameter(torch.LongTensor(
            get_tail_indices(9, p['embedding_dim'])), requires_grad=False)
        
        self.linear3 = nn.Linear(
            in_features=1456, out_features=n_out)
        nn.init.xavier_uniform_(self.linear3.weight)

        self.kgpool = Net(kgpool_args, embeddings, kgpool_char2idx)


    def forward(self, sentence_input, entity_markers, num_entities, kgpool_data, kbIDs, epoch = None):
        """
        Model forward function.
        Input:
        sentence_input: (batch_size, max_sent_len)
        entity_markers: (batch_size, MAX_EDGES_PER_GRAPH, max_sent_len) ? edge markers?
        num_entities: (batch_size,) a list of number of entities of each instance in the batch
        kgpool_data: tuple of data required by kgpool
        
        Output:
        main_output: (batch_size * MAX_EDGES_PER_GRAPH, n_out)
        """
        # Repeat the sentences for MAX_EDGES_PER_GRAPH times. As we will need it to be encoded differently
        # with difference target entity pairs.
        # [[1, 2, 3], [3, 4, 5]] => [[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[3, 4, 5], [3, 4, 5], [3, 4, 5]]]
        # shape: (batch_size, max_sent_len) => (batch_size, MAX_EDGES_PER_GRAPH, max_sent_len)
        expanded_sentence_input = torch.transpose(
            sentence_input.expand(MAX_EDGES_PER_GRAPH, sentence_input.size()[0], self.max_sent_len), 0, 1)
        
        # Word and position embeddings lookup.
        # shape: batch, MAX_EDGES_PER_GRAPH, max_sent_len, wordemb_dim
        word_embeddings = self.word_embedding(expanded_sentence_input.contiguous().view(-1, self.max_sent_len)).view(
            sentence_input.size()[0], MAX_EDGES_PER_GRAPH, self.max_sent_len, -1)
        word_embeddings = self.dropout1(word_embeddings)
        pos_embeddings = self.pos_embedding(entity_markers.contiguous().view(-1, self.max_sent_len)).view(
            sentence_input.size()[0], MAX_EDGES_PER_GRAPH, self.max_sent_len, -1)
        # Merge them together!
        merged_embeddings = torch.cat([word_embeddings, pos_embeddings], dim=3)
        merged_embeddings = merged_embeddings.view(-1, self.max_sent_len, merged_embeddings.size()[-1])

        # Encode the setntences with LSTM. 
        # NOTE that the api of LSTM, GRU and RNN are different, the following only works for LSTM
        rnn_output, _ = self.rnn1(merged_embeddings)
        # rnn_output shape: batch * MAX_EDGES_PER_GRAPH, max_sent_len, hidden
        rnn_result = torch.cat([rnn_output[:, -1, :self.p['units1']], rnn_output[:, 0, self.p['units1']:]], dim=1).view(sentence_input.size()[0], MAX_EDGES_PER_GRAPH, -1)
        rnn_result = self.dropout2(rnn_result)


        if(self.p['layer_number'] == 1 or self.p['projection_style'] == 'tie'):
            # 1 layer case or tied-matrices cases
            rnn_result = self.representation_to_adj(rnn_result).view(sentence_input.size()[0], 8, 9, (self.p['embedding_dim'] * 2) ** 2)  # magic number here
            if(self.p['non-linear'] != "linear"):
                try:
                    rnn_result = getattr(F, self.p['non-linear'])(rnn_result)
                except:
                    raise NotImplementedError
            identity_stuffing = self.identity_transformation.repeat(sentence_input.size()[0] * 8, 1).view(sentence_input.size()[0], 8, 1, (self.p['embedding_dim'] * 2) ** 2)
            rnn_result = torch.cat([identity_stuffing, rnn_result], dim=2).view(sentence_input.size()[0], 80, (self.p['embedding_dim'] * 2) ** 2)
            identity_stuffing = self.identity_transformation.repeat(sentence_input.size()[0], 1).view(sentence_input.size()[0], 1, (self.p['embedding_dim'] * 2) ** 2)
            adjecent_matrix = torch.cat([rnn_result, identity_stuffing], dim=1).view(sentence_input.size()[0], 9, 9, self.p['embedding_dim'] * 2, self.p['embedding_dim']
                                                                                     * 2).transpose(dim0=2, dim1=3).contiguous().view(sentence_input.size()[0], self.p['embedding_dim'] * 18, self.p['embedding_dim'] * 18)

            # adjecent_matrix = torch.matmul(adjecent_matrix, block_matrix).view(sentence_input.size()[0], 1, self.p['embedding_dim'] * 18, self.p['embedding_dim'] * 18)
            adjecent_matrix = adjecent_matrix.view(sentence_input.size()[0], 1, self.p['embedding_dim'] * 18, self.p['embedding_dim'] * 18)
            if(self.p['layer_number'] == 1):
                layer_1 = torch.matmul(adjecent_matrix, self.start_embedding).view(
                    sentence_input.size()[0], 72, self.p['embedding_dim'] * 18)
                if(self.p['non-linear1'] != 'linear'):
                    try:
                        layer_1 = getattr(F, self.p['non-linear1'])(layer_1)
                    except:
                        raise NotImplementedError

                heads = torch.gather(layer_1, 2, self.head_indices)
                tails = torch.gather(layer_1, 2, self.tail_indices)
                relation = heads * tails
                main_output = self.linear3(relation).view(
                    sentence_input.size()[0] * MAX_EDGES_PER_GRAPH, -1)
                return main_output
            else:
                layer_tmp = self.start_embedding
                relation_list = []
                for i in range(self.p['layer_number']):
                    layer_tmp = torch.matmul(adjecent_matrix, layer_tmp)
                    if(self.p['non-linear1'] != 'linear'):
                        try:
                            layer_tmp = getattr(
                                F, self.p['non-linear1'])(layer_tmp)
                        except:
                            raise NotImplementedError
                    layer_result = layer_tmp.view(
                        sentence_input.size()[0], 72, self.p['embedding_dim'] * 18)
                    
                    heads = torch.gather(layer_result, 2, self.head_indices)
                    tails = torch.gather(layer_result, 2, self.tail_indices)
                    relation = heads * tails
                    relation_list.append(relation)
                main_output = self.linear3(torch.cat(relation_list, dim=-1)).view(
                    sentence_input.size()[0] * MAX_EDGES_PER_GRAPH, -1)
                return main_output
        else:
            adjecent_matrix = []
            relation_list = []
            for i in range(self.p['layer_number']):
                rnn_result_tmp = self.representation_to_adj[i](rnn_result).view(sentence_input.size(
                )[0], 8, 9, (self.p['embedding_dim'] * 2) ** 2)  # 9 is the num of node, 8 * 9 is the edge_num
                
                if(self.p['non-linear1'] != "linear"):
                    try:
                        rnn_result_tmp = getattr(
                            F, self.p['non-linear1'])(rnn_result_tmp)
                    except:
                        raise NotImplementedError
                        
                identity_stuffing = self.identity_transformation.repeat(sentence_input.size()[0] * 8, 1).view(sentence_input.size()[0], 8, 1, (self.p['embedding_dim'] * 2) ** 2)
                
                rnn_result_tmp = torch.cat([identity_stuffing, rnn_result_tmp], dim=2).view(sentence_input.size()[0], 80, (self.p['embedding_dim'] * 2) ** 2)
                identity_stuffing = self.identity_transformation.repeat(sentence_input.size()[0], 1).view(sentence_input.size()[0], 1, (self.p['embedding_dim'] * 2) ** 2)
                adjecent_matrix.append(None)
                adjecent_matrix[i] = torch.cat([rnn_result_tmp, identity_stuffing], dim=1).view(sentence_input.size()[0], 9, 9, self.p['embedding_dim'] * 2, self.p['embedding_dim']
                                                                                                * 2).transpose(dim0=2, dim1=3).contiguous().view(sentence_input.size()[0], self.p['embedding_dim'] * 18, self.p['embedding_dim'] * 18)

                adjecent_matrix[i] = adjecent_matrix[i].view(sentence_input.size()[0], 1, self.p['embedding_dim'] * 18, self.p['embedding_dim'] * 18)
            layer_tmp = self.start_embedding
            for i in range(self.p['layer_number']):
                layer_tmp = torch.matmul(adjecent_matrix[i], layer_tmp)
                if(self.p['non-linear1'] != 'linear'):
                    try:
                        layer_tmp = getattr(
                            F, self.p['non-linear1'])(layer_tmp)
                    except:
                        raise NotImplementedError
                layer_result = layer_tmp.view(
                    sentence_input.size()[0], 72, self.p['embedding_dim'] * 18)
                
                #from IPython import embed; embed()
                heads = torch.gather(layer_result, 2, self.head_indices)
                tails = torch.gather(layer_result, 2, self.tail_indices)
                relation_list.append(heads * tails)

            #call kgpool here and append the values here
            gpgnn_out = torch.cat(relation_list, dim=-1)
            out = self.kgpool(kgpool_data[0],kgpool_data[1],kgpool_data[2],kgpool_data[3],kgpool_data[4],kgpool_data[5], epoch)
            row = len(kbIDs)
            col = len(kbIDs[0])

            count = 0
            #keep the size same as that of gpgnn_out
            kgpool_out = torch.zeros([row, col, 1408], dtype=torch.float32)
            for i in range(row):
                for j in range(col):
                    if(kbIDs[i][j]!= None):
                        kgpool_out[i,j,:] = out[count]
                        count += 1
            if (torch.cuda.is_available()):
                kgpool_out = kgpool_out.cuda()
            else:
                kgpool_out = kgpool_out

            #concat both kgpool and gpgnn output
            x = torch.cat([gpgnn_out,kgpool_out],dim=-1)


            main_output = self.linear3(x).view(
                sentence_input.size()[0] * MAX_EDGES_PER_GRAPH, -1)

            return main_output

