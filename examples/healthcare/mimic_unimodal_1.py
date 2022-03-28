import torch
from torch import nn
import sys
import os

sys.path.append(os.getcwd())

from unimodals.common_models import MLP, GRUWithLinear # noqa
from datasets.mimic.get_data import get_dataloader # noqa
from training_structures.unimodal import train, test # noqa

# get dataloader for icd9 classification task 7
traindata, validdata, testdata = get_dataloader(
    1, imputed_path='datasets/mimic/im.pk')
modalnum = 1
# build encoders, head and fusion layer
#encoders = [MLP(5, 10, 10,dropout=False).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), GRU(12, 30,dropout=False).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]
encoder = GRUWithLinear(12, 30, 15, flatten=True, batch_first=True).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
head = MLP(360, 40, 2, dropout=False).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


# train
train(encoder, head, traindata, validdata, 20, auprc=False, modalnum=modalnum)

# test
print("Testing: ")
encoder = torch.load('encoder.pt').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
head = torch.load('head.pt').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
# dataset = 'mimic mortality', 'mimic 1', 'mimic 7'
test(encoder, head, testdata, dataset='mimic 1', auprc=False, modalnum=modalnum)
