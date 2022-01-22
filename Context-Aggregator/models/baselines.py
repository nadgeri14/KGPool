import torch
from torch import nn
from torch.autograd import Variable
import sys
from parsing.legacy_sp_models import MAX_EDGES_PER_GRAPH
import torch.nn.functional as F


class ContextAware(nn.Module):
    """
    model described in 'Context-Aware Representations for Knowledge Base Relation Extraction'
    """

    def __init__(self, p, embeddings, max_sent_len, n_out):
        super(ContextAware, self).__init__()

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
        nn.init.orthogonal(self.pos_embedding.weight)

        # Merge word and position embeddings and apply the specified amount of RNN layers
        self.rnn1 = nn.LSTM(embeddings.shape[1] + int(p['position_emb']), int(p['units1']),
                                           num_layers=int(p['rnn1_layers']), batch_first=True, bidirectional = bool(p['bidirectional']))

        self.linear1 = nn.Linear(
            in_features=p['units1']*2, out_features=p['units1']*2, bias=False)

        nn.init.xavier_uniform(self.linear1.weight)

        self.dropout2 = nn.Dropout(p=p['dropout1'])
        self.linear2 = nn.Linear(
            in_features=p['units1'] * 4, out_features=n_out)
        nn.init.xavier_uniform(self.linear2.weight)

    def forward(self, sentence_input, entity_markers):
        """
        Model forward function.
        Input:
        sentence_input: (batch_size, max_sent_len)
        entity_markers: (batch_size, MAX_EDGES_PER_GRAPH, max_sent_len)
        Output:
        main_output: (batch_size * MAX_EDGES_PER_GRAPH, n_out)
        """
        # Repeat the input MAX_EDGES_PER_GRAPH times as will need it once for the target entity pair and twice for the ghost pairs
        # [[1, 2, 3], [3, 4, 5]] => [[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[3, 4, 5], [3, 4, 5], [3, 4, 5]]]
        # shape: (batch_size, MAX_EDGES_PER_GRAPH, max_sent_len)
        expanded_sentence_input = torch.transpose(
            sentence_input.expand(MAX_EDGES_PER_GRAPH, sentence_input.size()[0], self.max_sent_len), 0, 1)
        word_embeddings = self.word_embedding(expanded_sentence_input.contiguous().view(-1, self.max_sent_len)).view(
            sentence_input.size()[0], MAX_EDGES_PER_GRAPH, self.max_sent_len, -1)
        word_embeddings = self.dropout1(word_embeddings)
        pos_embeddings = self.pos_embedding(entity_markers.contiguous().view(-1, self.max_sent_len)).view(
            sentence_input.size()[0], MAX_EDGES_PER_GRAPH, self.max_sent_len, -1)
        merged_embeddings = torch.cat([word_embeddings, pos_embeddings], dim=3)

        # prepare to input rnn cell
        merged_embeddings = merged_embeddings.view(-1,
                                                   self.max_sent_len, merged_embeddings.size()[-1])

        rnn_output, _ = self.rnn1(merged_embeddings)

        rnn_result = torch.cat([rnn_output[:, -1, :self.p['units1']], rnn_output[:, 0, self.p['units1']:]], dim=1)

        # return format
        rnn_result = rnn_result.view(
            sentence_input.size()[0], MAX_EDGES_PER_GRAPH, -1)

        ### Attention over ghosts ###
        layers_to_concat = []

        for i in range(MAX_EDGES_PER_GRAPH):
            # Compute a memory vector for the target entity pair
            sentence_vector = rnn_result[:, i, :]
            target_sentence_memory = self.linear1(sentence_vector)
            if i == 0:
                context_vectors = rnn_result[:, 1:, :]
            elif i == MAX_EDGES_PER_GRAPH - 1:
                context_vectors = rnn_result[:, :i, :]
            else:
                context_vectors = torch.cat(
                    [rnn_result[:, :i, :], rnn_result[:, i + 1:, :]], dim=1)
            # Compute the score between each memory and the memory of the target entity pair
            sentence_scores = torch.matmul(target_sentence_memory.unsqueeze(
                1).unsqueeze(1), context_vectors.unsqueeze(3)).squeeze()
            sentence_scores = nn.functional.softmax(sentence_scores)

            # Compute the final vector by taking the weighted sum of context vectors and the target entity vector
            context_vector = torch.matmul(
                sentence_scores.unsqueeze(1), context_vectors).squeeze()
            edge_vector = torch.cat([sentence_vector, context_vector], dim=1)
            layers_to_concat.append(edge_vector.unsqueeze(1))

        edge_vectors = torch.cat(layers_to_concat, dim=1)
        edge_vectors = edge_vectors.view(sentence_input.size()[0] * MAX_EDGES_PER_GRAPH, -1)

        # Apply softmax
        edge_vectors = self.dropout2(edge_vectors)

        main_output = self.linear2(edge_vectors)

        return main_output


class LSTM(nn.Module):
    """
    LSTM baseline model
    """

    def __init__(self, p, embeddings, max_sent_len, n_out):
        super(LSTM, self).__init__()

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

        # Merge word and position embeddings and apply the specified amount of RNN layers
        self.rnn1 = getattr(nn, p['rnn1'])(batch_first=True,
                                           input_size=embeddings.shape[1] +
                                           p['position_emb'],
                                           hidden_size=p['units1'],
                                           num_layers=p['rnn1_layers'], bidirectional=p['bidirectional'])
        for parameter in self.rnn1.parameters():
            if(len(parameter.size()) >= 2):
                nn.init.orthogonal(parameter)                                   

        self.linear1 = nn.Linear(
            in_features=p['units1'], out_features=p['units1'], bias=False)
        nn.init.xavier_uniform(self.linear1.weight)    

        self.dropout2 = nn.Dropout(p=p['dropout1'])
        self.linear2 = nn.Linear(
            in_features=p['units1'] * 2, out_features=n_out)
        nn.init.xavier_uniform(self.linear2.weight)

    def forward(self, sentence_input, entity_markers):
        """
        Model forward function.
        Input:
        sentence_input: (batch_size, max_sent_len)
        entity_markers: (batch_size, max_sent_len)
        Output:
        main_output: (batch_size, n_out)
        """

        word_embeddings = self.word_embedding(sentence_input)
        word_embeddings = self.dropout1(word_embeddings)
        pos_embeddings = self.pos_embedding(entity_markers)
        merged_embeddings = torch.cat([word_embeddings, pos_embeddings], dim=2)

        rnn_output, _ = self.rnn1(merged_embeddings)

        rnn_result = torch.cat([rnn_output[:, -1, :self.p['units1']], rnn_output[:, 0, self.p['units1']:]], dim=1)

        # Apply softmax
        rnn_result = self.dropout2(rnn_result)
        main_output = self.linear2(rnn_result)

        return main_output

class CNN(nn.Module):
    """
    CNN baseline model
    """

    def __init__(self, p, embeddings, max_sent_len, n_out):
        super(CNN, self).__init__()

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

        self.pos_embedding_0 = nn.Embedding(2 * max_sent_len + 1, p['position_emb'], padding_idx=0)
        self.pos_embedding_1 = nn.Embedding(2 * max_sent_len + 1, p['position_emb'], padding_idx=0)

        # Merge word and position embeddings and apply the specified amount of RNN layers
        self.cnn_3 = nn.Conv1d(embeddings.shape[1] + p['position_emb'] * 2, p['units1'], p['window_size'], bias=True)
        self.cnn_5 = nn.Conv1d(embeddings.shape[1] + p['position_emb'] * 2, p['units1'], 5, bias=True)
        self.cnn_7 = nn.Conv1d(embeddings.shape[1] + p['position_emb'] * 2, p['units1'], 7, bias=True)
        nn.init.xavier_uniform(self.cnn_3.weight)
        nn.init.xavier_uniform(self.cnn_5.weight)
        nn.init.xavier_uniform(self.cnn_7.weight)

        self.linear1 = nn.Linear(
            in_features=p['units1'] * 3, out_features=n_out, bias=False)
        nn.init.xavier_uniform(self.linear1.weight)    

    def forward(self, sentence_input, entity_markers):
        """
        Model forward function.
        Input:
        sentence_input: (batch_size, max_sent_len)
        entity_markers: (batch_size, max_sent_len)
        Output:
        main_output: (batch_size, n_out)
        """

        word_embeddings = self.word_embedding(sentence_input)
        word_embeddings = self.dropout1(word_embeddings)
        pos_embeddings_0 = self.pos_embedding_0(entity_markers[:, 0, :])
        pos_embeddings_1 = self.pos_embedding_1(entity_markers[:, 1, :])
        merged_embeddings = torch.cat([word_embeddings, pos_embeddings_0, pos_embeddings_1], dim=2)

        merged_embeddings = merged_embeddings.transpose(1, 2)
        cnn_3_output = F.tanh(self.dropout1(torch.max(self.cnn_3(merged_embeddings), dim = 2)[0]))
        cnn_5_output = F.tanh(self.dropout1(torch.max(self.cnn_5(merged_embeddings), dim = 2)[0]))
        cnn_7_output = F.tanh(self.dropout1(torch.max(self.cnn_7(merged_embeddings), dim = 2)[0]))
        cnn_output = torch.cat([cnn_3_output, cnn_5_output, cnn_7_output], dim=1)

        # Apply softmax
        main_output = self.linear1(cnn_output)

        return main_output

class PCNN(nn.Module):
    """
    PCNN baseline model
    """

    def __init__(self, p, embeddings, max_sent_len, n_out):
        super(PCNN, self).__init__()

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

        self.pos_embedding_0 = nn.Embedding(2 * max_sent_len + 1, p['position_emb'], padding_idx=0)
        self.pos_embedding_1 = nn.Embedding(2 * max_sent_len + 1, p['position_emb'], padding_idx=0)

        # Merge word and position embeddings and apply the specified amount of RNN layers
        self.cnn = nn.Conv1d(embeddings.shape[1] + p['position_emb'] * 2, p['units1'], p['window_size'], bias=True, padding=int(p['window_size']/2))
        nn.init.xavier_uniform(self.cnn.weight)

        self.linear1 = nn.Linear(
            in_features=p['units1'] * 3, out_features=n_out, bias=False)
        nn.init.xavier_uniform(self.linear1.weight)    

    def forward(self, sentence_input, entity_markers, pcnn_mask):
        """
        Model forward function.
        Input:
        sentence_input: (batch_size, max_sent_len)
        entity_markers: (batch_size, max_sent_len)
        Output:
        main_output: (batch_size, n_out)
        """

        word_embeddings = self.word_embedding(sentence_input)
        word_embeddings = self.dropout1(word_embeddings)
        pos_embeddings_0 = self.pos_embedding_0(entity_markers[:, 0, :])
        pos_embeddings_1 = self.pos_embedding_1(entity_markers[:, 1, :])
        merged_embeddings = torch.cat([word_embeddings, pos_embeddings_0, pos_embeddings_1], dim=2)

        merged_embeddings = merged_embeddings.transpose(1, 2)
        cnn_output = self.cnn(merged_embeddings)
        left = torch.max(cnn_output * pcnn_mask[:, 0].unsqueeze(1).repeat(1, self.p['units1'], 1), dim=2)[0]
        middle = torch.max(cnn_output * pcnn_mask[:, 1].unsqueeze(1).repeat(1, self.p['units1'], 1), dim=2)[0]
        right = torch.max(cnn_output * pcnn_mask[:, 2].unsqueeze(1).repeat(1, self.p['units1'], 1), dim=2)[0]
        cnn_output = F.relu(self.dropout1(torch.cat([left, middle, right], dim=1)))
        

        # Apply softmax
        main_output = self.linear1(cnn_output)

        return main_output
