# -*- coding: utf-8 -*-

from .Dataset import Dataset
import os
import pandas as pd
import numpy as np
from codecs import open

class SSTDataset(Dataset):
    def __init__(self,opt=None,**kwargs):
        super(SSTDataset,self).__init__(opt,**kwargs)
        self.urls=['http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip']
        
    
    def process(self):
        # root=self.download()
        # root = os.path.join(os.getcwd(),"TextClassificationBenchmark")
        # print("processing into: "+ root)
##        root = "D:\code\git\TextClassificationBenchmark\.data_waby\\imdb\\aclImdb"
        if not os.path.exists(self.saved_path):
            print("mkdir " + self.saved_path)
            os.makedirs(self.saved_path) # better than os.mkdir
#            
        datas=[]

        filename_train = os.path.join(os.getcwd(),"sst-2","stsa.binary.train.txt") 
        filename_dev = os.path.join(os.getcwd(),"sst-2","stsa.binary.dev.txt") 
        filename_test = os.path.join(os.getcwd(),"sst-2","stsa.binary.test.txt") 
        # print('filename', filename)
        records=[]
        text_data_set = []
        target_label_set = []
        train = []
        dev = []
        test = []
        with open(filename_train,encoding="utf-8",errors="replace") as f:
            for i,line in enumerate(f):
                text_data = line.strip()[2:]
                target_label = int(line.strip()[0:2])
                train.append({"text":text_data, "label":target_label})
        with open(filename_dev,encoding="utf-8",errors="replace") as f:
            for i,line in enumerate(f):
                text_data = line.strip()[2:]
                target_label = int(line.strip()[0:2])
                dev.append({"text":text_data, "label":target_label})
        with open(filename_test,encoding="utf-8",errors="replace") as f:
            for i,line in enumerate(f):
                text_data = line.strip()[2:]
                target_label = int(line.strip()[0:2])
                test.append({"text":text_data, "label":target_label})
        train = pd.DataFrame(train)
        dev = pd.DataFrame(dev)
        test = pd.DataFrame(test)
        df = pd.concat([train, dev])
        # from sklearn.utils import shuffle  
        # df = shuffle(datas, random_state=0).reset_index()
        train = df.iloc[:int(len(df)*5/6)]
        dev = df.iloc[int(len(df)*5/6):]
        # test = df.iloc[int(len(df)*6/7):]

        train_filename=os.path.join(self.saved_path,"train.csv")
        dev_filename = os.path.join(self.saved_path,"dev.csv")
        test_filename = os.path.join(self.saved_path,"test.csv")
        train[["text","label"]].to_csv(train_filename,encoding="utf-8",sep="\t",index=False,header=None)
        dev[["text","label"]].to_csv(dev_filename,encoding="utf-8",sep="\t",index=False,header=None)
        test[["text","label"]].to_csv(test_filename,encoding="utf-8",sep="\t",index=False,header=None)
        print("processing into formated files over")
        return [train_filename, dev_filename, test_filename]








#         root=self.download()
#         root = os.path.join(root,"stanfordSentimentTreebank")
#         print("processing into: "+ root)
# ##        root = "D:\code\git\TextClassificationBenchmark\.data_waby\\imdb\\aclImdb"
#         if not os.path.exists(self.saved_path):
#             print("mkdir " + self.saved_path)
#             os.makedirs(self.saved_path) # better than os.mkdir
# #            
#         datas=[]

#         filename = os.path.join(root,"original_rt_snippets.txt") 
#         filename_for_labels = os.path.join(root,"sentiment_labels.txt")
#         print('filename', filename)
#         records=[]
#         text_data_set = []
#         target_label_set = []
#         with open(filename,encoding="utf-8",errors="replace") as f:
#             for i,line in enumerate(f):
#                 text_data = line.strip()
#                 text_data_set.append(text_data)
#         with open(filename_for_labels,encoding="utf-8",errors="replace") as f:
#             for i,line in enumerate(f):
#                 if i == 0:
#                     continue
#                 if i == 10605:
#                     break
#                 target_label = line.split('|')[1]
#                 target_label = float(target_label)
#                 if target_label <= 0.2:
#                     target_label = 1
#                 elif target_label <= 0.4:
#                     target_label = 2
#                 elif target_label <= 0.6:
#                     target_label = 3
#                 elif target_label <= 0.8:
#                     target_label = 4
#                 else:
#                     target_label = 5

#                 target_label_set.append(target_label)
#                 records.append({"text":text_data_set[i-1], "label":target_label})
#                 # records.append({"text":line.strip(),"label": 1 if polarity == "pos" else 0})
#         print(target_label_set[0:10])
#         print(text_data_set[0:10])
#         print(records[0:10])
#         datas = pd.DataFrame(records)
        

#         from sklearn.utils import shuffle  
#         df = shuffle(datas, random_state=0).reset_index()
#         train = df.iloc[:int(len(df)*5/7)]
#         dev = df.iloc[int(len(df)*5/7):int(len(df)*6/7)]
#         test = df.iloc[int(len(df)*6/7):]

#         train_filename=os.path.join(self.saved_path,"train.csv")
#         dev_filename = os.path.join(self.saved_path,"dev.csv")
#         test_filename = os.path.join(self.saved_path,"test.csv")
#         train[["text","label"]].to_csv(train_filename,encoding="utf-8",sep="\t",index=False,header=None)
#         dev[["text","label"]].to_csv(dev_filename,encoding="utf-8",sep="\t",index=False,header=None)
#         test[["text","label"]].to_csv(test_filename,encoding="utf-8",sep="\t",index=False,header=None)
#         print("processing into formated files over")
#         return [train_filename, dev_filename, test_filename]

if __name__=="__main__":
    import opts
    opt = opts.parse_opt()
    opt.dataset="sst"
    import dataloader
    dataset= dataloader.getDataset(opt)
    dataset.process()
    

    