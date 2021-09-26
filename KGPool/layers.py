from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
import torch
import numpy as np

class KGPool(torch.nn.Module):
    def __init__(self,in_channels,ratio=1.0,Conv=GCNConv,non_linearity=torch.tanh,dynamic_pooling=True):
        super(KGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels,1)
        self.non_linearity = non_linearity
        self.dynamic_pooling = dynamic_pooling

    def forward(self, x, edge_index, edge_attr=None, batch=None, node1_indices = None, node2_indices = None, sent_indices = None, alpha = 4):
        #Here we only pool the context nodes
        #The entity nodes and sentence nodes are not pooled
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        score = self.score_layer(x,edge_index).squeeze()

        if(self.dynamic_pooling):
            perm = []
            score_list = score.tolist()
            temp_node1_indices = node1_indices.tolist() + [len(score_list)]
            off_set = 0
            for i in range(len(temp_node1_indices)-1):
                scores_per_sentence = score_list[temp_node1_indices[i]:temp_node1_indices[i+1]]
                norm_scores = np.exp(scores_per_sentence) / np.sum(np.exp(scores_per_sentence), axis=0)
                max_scores = np.max(norm_scores)
                std_scores = np.std(norm_scores)
                cut_off = max_scores - (alpha * std_scores)
                indices = [i for i,v in enumerate(norm_scores) if v >= cut_off]
                indices = [x+off_set for x in indices]
                perm += indices 
                off_set += len(norm_scores)
            perm = torch.Tensor(perm).type(torch.LongTensor).cuda()
        else:
            perm = topk(score, self.ratio, batch)

        perm, indices = torch.sort(perm)
        perm = perm.tolist()
        node1_indices = node1_indices.tolist()
        node2_indices = node2_indices.tolist()
        sent_indices = sent_indices.tolist() 
        all_unique_nodes = list(set(perm + node1_indices + node2_indices + sent_indices))

        all_unique_nodes.sort()
        node1_indices.sort()
        node2_indices.sort()
        sent_indices.sort()

        new_node1_indices = []
        new_node2_indices = []
        new_sent_indices = []

        node1_iterator = 0
        node2_iterator = 0
        sentence_iterator = 0

        for i in range(len(all_unique_nodes)):
            if(node1_iterator < len(node1_indices) and all_unique_nodes[i] == node1_indices[node1_iterator]):
                new_node1_indices.append(i)
                node1_iterator += 1
            if(node2_iterator < len(node2_indices) and all_unique_nodes[i] == node2_indices[node2_iterator]):
                new_node2_indices.append(i)
                node2_iterator += 1
            if(sentence_iterator < len(sent_indices) and all_unique_nodes[i] == sent_indices[sentence_iterator]):
                new_sent_indices.append(i)
                sentence_iterator += 1


        if (torch.cuda.is_available()):
            perm = torch.tensor(all_unique_nodes).cuda()
            new_node1_indices = torch.tensor(new_node1_indices).cuda()
            new_node2_indices = torch.tensor(new_node2_indices).cuda()
            new_sent_indices = torch.tensor(new_sent_indices).cuda()
        else:
            perm = torch.tensor(all_unique_nodes)
            new_node1_indices = torch.tensor(new_node1_indices)
            new_node2_indices = torch.tensor(new_node2_indices)
            new_sent_indices = torch.tensor(new_sent_indices)

        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, new_node1_indices, new_node2_indices, new_sent_indices
