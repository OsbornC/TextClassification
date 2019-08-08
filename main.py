# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from torch.nn import DataParallel
from six.moves import cPickle

import opts
import models
import torch.nn as nn
import utils
import torch.nn.functional as F
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText
from torch.nn.modules.loss import NLLLoss,MultiLabelSoftMarginLoss,MultiLabelMarginLoss,BCELoss
import dataHelper
import time,os


from_torchtext = False

opt = opts.parse_opt()
#opt.proxy="http://xxxx.xxxx.com:8080"






if "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
    os.environ["CUDA_VISIBLE_DEVICES"] =opt.gpu
#opt.model ='lstm'
#opt.model ='capsule'

if from_torchtext:
    train_iter, test_iter = utils.loadData(opt)
else:
    import dataHelper as helper
    train_iter, dev_iter, test_iter = dataHelper.loadData(opt)

opt.lstm_layers=2

torch.manual_seed(0)
model=models.setup(opt)
if torch.cuda.is_available():
    model.cuda()
model.train()
print("# parameters:", sum(param.numel() for param in model.parameters() if param.requires_grad))
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
optimizer.zero_grad()
loss_fun = F.cross_entropy
# model = DataParallel(model)             # 并行化model
# loss  = DataParallel(loss_fun)
#batch = next(iter(train_iter))

#x=batch.text[0]

#x=batch.text[0] #64x200

#print(utils.evaluation(model,test_iter))
max_precision = 0
for i in range(opt.max_epoch):
    model.train()
    for batch, iterator in enumerate(train_iter):

        start= time.time()
        optimizer.zero_grad()
        text = iterator.text[0] if from_torchtext else iterator.text
        predicted = model(text)

        loss= loss_fun(predicted,iterator.label)

        loss.backward()
        utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        if batch% 100==0:
            if  torch.cuda.is_available():
                print("%d epoch %d batch with loss : %.5f in %.4f seconds" % (i+1,batch,loss.cuda().item(),time.time()-start))
            else:
                print("%d epoch %d batch with loss : %.5f in %.4f seconds" % (i+1,batch,loss.data.numpy(),time.time()-start))
                # print("iteration epoch with loss :  in  seconds")
                # print(i,epoch,loss.data.numpy(),time.time()-start)
 
    precision, F1 =utils.evaluation(model,dev_iter,from_torchtext)
    print("%d epoch with precision %.4f and F1-score %.5f on development set" % (i+1,precision, F1))
    if precision > max_precision:
        max_precision = precision
        test_precision, test_F1 =utils.evaluation(model,test_iter,from_torchtext)
        print("%d epoch with precision %.4f and F1-score %.5f on test set" % (i+1,test_precision, test_F1))

    if from_torchtext:
        train_iter, test_iter = utils.loadData(opt)
    else:
        import dataHelper as helper
        train_iter, _, _ = dataHelper.loadData(opt)


