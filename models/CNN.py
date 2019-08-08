import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, opt):
        super(CNN, self).__init__()

        self.embedding_type = opt.embedding_type
        self.batch_size = opt.batch_size
        self.max_sent_len = opt.max_sent_len
        self.embedding_dim = opt.embedding_dim
        self.vocab_size = opt.vocab_size
        self.CLASS_SIZE = opt.label_size
        self.FILTERS = opt["FILTERS"]
        self.FILTER_NUM = opt["FILTER_NUM"]
        self.keep_dropout = opt.keep_dropout
        self.IN_CHANNEL = 1

        assert (len(self.FILTERS) == len(self.FILTER_NUM))

        # one for UNK and one for zero padding
        self.embedding = nn.Embedding(self.vocab_size + 2, self.embedding_dim, padding_idx=self.vocab_size + 1)
        if self.embedding_type == "static" or self.embedding_type == "non-static" or self.embedding_type == "multichannel":
            self.WV_MATRIX = opt["WV_MATRIX"]
            self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
            if self.embedding_type == "static":
                self.embedding.weight.requires_grad = False
            elif self.embedding_type == "multichannel":
                self.embedding2 = nn.Embedding(self.vocab_size + 2, self.embedding_dim, padding_idx=self.VOCAB_SIZE + 1)
                self.embedding2.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
                self.embedding2.weight.requires_grad = False
                self.IN_CHANNEL = 2

        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.embedding_dim * self.FILTERS[i], stride=self.WORD_DIM)
            setattr(self, 'conv_%d'%i, conv)

        self.fc = nn.Linear(sum(self.FILTER_NUM), self.label_size)

    def get_conv(self, i):
        return getattr(self, 'conv_%d'%i)

    def forward(self, inp):
        x = self.embedding(inp).view(-1, 1, self.embedding_dim * self.max_sent_len)
        if self.embedding_type == "multichannel":
            x2 = self.embedding2(inp).view(-1, 1, self.embedding_dim * self.max_sent_len)
            x = torch.cat((x, x2), 1)

        conv_results = [
            F.max_pool1d(F.relu(self.get_conv(i)(x)), self.max_sent_len - self.FILTERS[i] + 1)
                .view(-1, self.FILTER_NUM[i])
            for i in range(len(self.FILTERS))]

        x = torch.cat(conv_results, 1)
        x = F.dropout(x, p=self.keep_dropout, training=self.training)
        x = self.fc(x)
        return x



#https://github.com/zachAlbus/pyTorch-text-classification/blob/master/Yoon/model.py
class  CNN1(nn.Module):
    
    def __init__(self, opt):
        super(CNN1,self).__init__()
        self.opt = opt
        
        V = opt.vocab_size
        D = opt.embedding_dim
        C = opt.label_size
        Ci = 1
        Co = opt.kernel_num
        Ks = opt.kernel_sizes

        self.embed = nn.Embedding(V, D)
        #self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(opt.dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3) #(N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x


    def forward(self, x):
        x = self.embed(x) # (N,W,D)
        
        if self.args.static:
            x = Variable(x)

        x = x.unsqueeze(1) # (N,Ci,W,D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)


        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x) # (N,len(Ks)*Co)
        logit = self.fc1(x) # (N,C)
        return logit

import torch.nn as nn


#https://github.com/zachAlbus/pyTorch-text-classification/blob/master/Zhang/model.py
class CNN2(nn.Module):
    def __init__(self, opt):
        super(CNN2, self).__init__()
        self.embed = nn.Embedding(opt.vocab_size + 1, opt.embedding_dim)

        self.conv1 = nn.Sequential(
            nn.Conv1d(opt.embedding_dim, 128, kernel_size=100, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=100, stride=3)
        )

        # self.conv2 = nn.Sequential(
        #     nn.Conv1d(128, 128, kernel_size=7, stride=1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=3, stride=3)
        # )

        # self.conv3 = nn.Sequential(
        #     nn.Conv1d(256, 256, kernel_size=3, stride=1),
        #     nn.ReLU()
        # )

        # self.conv4 = nn.Sequential(
        #     nn.Conv1d(256, 256, kernel_size=3, stride=1),
        #     nn.ReLU()
        # )

        # self.conv5 = nn.Sequential(
        #     nn.Conv1d(256, 256, kernel_size=3, stride=1),
        #     nn.ReLU()
        # )

        # self.conv6 = nn.Sequential(
        #     nn.Conv1d(256, 256, kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=3, stride=3)
        # )

        self.fc = nn.Linear(128, opt.label_size)

    def forward(self, x_input):
        # Embedding
        x = self.embed(x_input)  # dim: (batch_size, max_seq_len, embedding_size)
        x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.conv6(x)

        # collapse
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return F.log_softmax(x)


class CNNClassifier(nn.Module):

    def __init__(self, opt):
        super(CNNClassifier, self).__init__()

        self.word_embeddings = nn.Embedding(opt.vocab_size+1, opt.embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.kernel_num = 128
        self.conv1 = nn.ModuleList([
            nn.Conv1d(opt.embedding_dim, self.kernel_num, kernel_size=100)
            for kernel_size in (2,3,4)
        ])
        
        self.maxpool = nn.MaxPool1d(55)
        self.avgpool = nn.AvgPool1d(55)
        self.dropout = nn.Dropout(p=0.5)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2label = nn.Linear(27, opt.label_size)
#         self.hidden2label = nn.Linear(200, 2)
#         self.hidden2label = nn.Sequential(
# #             nn.Linear(1800, 400),
# #             nn.Linear(300, 120),
#             nn.Linear(200, 84),
#             nn.Linear(84, 2)
#         )
#         self.conv1 = nn.Sequential(
#             nn.Conv1d(embedding_dim, kernel_num, kernel_size),
            
#         )
#         self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
#             nn.Conv1d(6, 64, 1)  # output shape (32, 14, 14)
# #             nn.ReLU()  # activation
# #             nn.MaxPool1d(20),  # output shape (32, 7, 7)
# #             nn.AvgPool1d(20)
#         )
        #self.batch_size = batch_size
        #self.hidden = self.init_hidden(batch_size)
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, sentence):
        #Batch_size, word_len, emb_size
        embeds = self.dropout(self.word_embeddings(sentence))
        size = embeds.size()
        
        ##Batch_size, 1, word_len, emb_size
        inputs = embeds.view(size[0], size[2], size[1])
#         inputs = embeds.view((size[0], 1, size[1], size[2]))
        #Batch_size, out_channel, n-stride+1, 1
        xs = []
        for conv in self.conv1:
#             x2 = F.relu(conv(x))        # [B, F, T, 1]
#             x2 = torch.squeeze(x2, -1)  # [B, F, T]
#             x2 = F.max_pool1d(x2, x2.size(2))  # [B, F, 1]
            outputs = F.relu(conv(inputs))
  
            max_pool = self.maxpool(outputs).squeeze()
            max_pool = max_pool.view(max_pool.size()[0], max_pool.size()[1], 1)
            avg_pool = self.avgpool(outputs).squeeze()
            avg_pool = avg_pool.view(avg_pool.size()[0], avg_pool.size()[1], 1)
            min_pool = self.maxpool(-outputs).squeeze()
            min_pool = min_pool.view(min_pool.size()[0], min_pool.size()[1], 1)
            pool = self.dropout(torch.cat([max_pool, avg_pool, min_pool], dim=1))
#             pool = self.dropout(self.conv2(self.dropout(torch.cat([max_pool, avg_pool, min_pool], dim=1))))
            xs.append(pool)
        x = torch.cat(xs, 2) 
        
#         outputs = nn.ReLU(outputs)
#         outputs = outputs.view((size[0], 1, self.kernel_num, -1))
        
        #Concatenate max and average pooling
#         pool = self.dropout(max_pool)
#         pool = self.dropout(self.conv2(self.dropout(torch.cat([max_pool, avg_pool, min_pool], dim=1))))
        
        #fully connected layer
#         pool = self.dropout(self.conv2(pool))
        probs = self.hidden2label(x.view(-1, x.size(1) * x.size(2)))

#         pred_scores = F.log_softmax(probs, dim=1)
        
        return F.log_softmax(probs)


class CNN3(nn.Module):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, args):
        super(CNN3, self).__init__()
        self.args = args

        embedding_dim = args.embed_dim
        embedding_num = args.num_features
        class_number = args.class_num
        in_channel = 1
        out_channel = args.kernel_num
        kernel_sizes = args.kernel_sizes

        self.embed = nn.Embedding(embedding_num+1, embedding_dim)
        self.conv = nn.ModuleList([nn.Conv2d(in_channel, out_channel, (K, embedding_dim)) for K in kernel_sizes])

        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(kernel_sizes) * out_channel, class_number)


    def forward(self, input_x):
        """
        :param input_x: a list size having the number of batch_size elements with the same length
        :return: batch_size X num_aspects tensor
        """
        # Embedding
        x = self.embed(input_x)  # dim: (batch_size, max_seq_len, embedding_size)

        if self.args.static:
            x = F.Variable(input_x)

        # Conv & max pool
        x = x.unsqueeze(1)  # dim: (batch_size, 1, max_seq_len, embedding_size)

        # turns to be a list: [ti : i \in kernel_sizes] where ti: tensor of dim([batch, num_kernels, max_seq_len-i+1])
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv]

        # dim: [(batch_size, num_kernels), ...]*len(kernel_sizes)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        # Dropout & output
        x = self.dropout(x)  # (batch_size,len(kernel_sizes)*num_kernels)
        logit = F.log_softmax(self.fc(x))  # (batch_size, num_aspects)

        return logit