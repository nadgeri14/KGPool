import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from layers import KGPool

class CharEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, drop_out_rate):
        super(CharEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, words_seq):
        char_embeds = self.embeddings(words_seq)
        char_embeds = self.dropout(char_embeds)
        return char_embeds

class NodeFeature(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, is_bidirectional, drop_out_rate, entity_embed_dim, conv_filter_size, embeddings, char_embed_dim, max_word_len_entity, char_vocab, char_feature_size):
        super(NodeFeature, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.is_bidirectional = is_bidirectional
        self.drop_rate = drop_out_rate

        self.word_embeddings = nn.Embedding(
            embeddings.shape[0], embeddings.shape[1], padding_idx=0)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(embeddings))
        self.word_embeddings.weight.requires_grad = False
        self.char_embeddings = CharEmbeddings(len(char_vocab), char_embed_dim, self.drop_rate)
        self.lstm = nn.LSTM(embeddings.shape[1]+char_feature_size, self.hidden_dim, self.layers, batch_first=True,
          bidirectional=bool(self.is_bidirectional))

        self.conv1d = nn.Conv1d(char_embed_dim, char_feature_size, conv_filter_size,padding=0)
        self.max_pool = nn.MaxPool1d(max_word_len_entity + conv_filter_size - 1, max_word_len_entity + conv_filter_size - 1)

    def forward(self, words, chars):
        batch_size = words.shape[0]
        if len(words.shape)==3:
          # max_batch_len = words.shape[1]
          words = words.view(words.shape[0]*words.shape[1],words.shape[2])
          chars = chars.view(chars.shape[0]*chars.shape[1],chars.shape[2])
        src_word_embeds = self.word_embeddings(words)
        try:
         char_embeds = self.char_embeddings(chars)
        except Exceprion as e:
          import pdb; pdb.set_trace()
        char_embeds = char_embeds.permute(0, 2, 1)

        char_feature = torch.tanh(self.max_pool(self.conv1d(char_embeds)))
        char_feature = char_feature.permute(0, 2, 1)

        words_input = torch.cat((src_word_embeds, char_feature), -1)
        outputs, hc = self.lstm(words_input)

        h_n = hc[0].view(self.layers, 2, words.shape[0], self.hidden_dim)
        h_n = h_n[-1,:,:,:].squeeze() # (num_dir,batch,hidden)
        h_n = h_n.permute((1,0,2)) # (batch,num_dir,hidden)
        h_n = h_n.contiguous().view(h_n.shape[0],h_n.shape[1]*h_n.shape[2]) # (batch,num_dir*hidden)
        
        return h_n



class Net(torch.nn.Module):
    def __init__(self, args, embeddings, char_vocab):
        super(Net, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.pooling_ratio = args.pooling_ratio
        self.dynamic_pooling1 = args.dynamic_pooling1
        self.dynamic_pooling2 = args.dynamic_pooling2
        self.dynamic_pooling3 = args.dynamic_pooling3

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.pool1 = KGPool(self.nhid, ratio=self.pooling_ratio_1_2, dynamic_pooling=self.dynamic_pooling1)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.pool2 = KGPool(self.nhid, ratio=self.pooling_ratio_1_2, dynamic_pooling=self.dynamic_pooling2)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.pool3 = KGPool(self.nhid, ratio=self.pooling_ratio_3, dynamic_pooling=self.dynamic_pooling3)

        self.nf = NodeFeature(args.input_dim, args.hidden_dim, args.layers, args.is_bidirectional, args.drop_out_rate, args.entity_embed_dim, args.conv_filter_size, embeddings, args.char_embed_dim, args.max_word_len_entity, char_vocab, args.char_feature_size)

    def forward(self, words, chars, edge_index, batch, entity_indices, sent_indices, epoch = None):
        #s = get the nodes for s too and ensure that it is not getting pooled
        #(ensure that e1 and e2 are not getting pooled in pool1)

        node1_indices = entity_indices[:,0]
        node2_indices = entity_indices[:,1]
        sent_indices = torch.flatten(sent_indices)


        x = self.nf(words, chars)

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, node1_indices, node2_indices, sent_indices = self.pool1(x, edge_index, None, batch, node1_indices, node2_indices, sent_indices)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        e1_x1 = x[node1_indices]
        e2_x1 = x[node2_indices]
        s_x1 = x[sent_indices]

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, node1_indices, node2_indices, sent_indices = self.pool2(x, edge_index, None, batch, node1_indices, node2_indices, sent_indices)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        e1_x2 = x[node1_indices]
        e2_x2 = x[node2_indices]
        s_x2 = x[sent_indices]

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, node1_indices, node2_indices, sent_indices = self.pool3(x, edge_index, None, batch, node1_indices, node2_indices, sent_indices)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        e1_x3 = x[node1_indices]
        e2_x3 = x[node2_indices]
        s_x3 = x[sent_indices]

        e1_cat = torch.cat([e1_x1,e1_x2,e1_x3], dim=1)
        e2_cat = torch.cat([e2_x1,e2_x2,e2_x3], dim=1)
        s_cat = torch.cat([s_x1,s_x2,s_x3], dim=1)

        x = x1 + x2 + x3

        x = torch.cat([e1_cat,e2_cat,s_cat,x],dim=1)

        return x

    
