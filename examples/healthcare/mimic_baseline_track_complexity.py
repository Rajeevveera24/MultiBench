import sys
import os
import torch
from torch import nn
sys.path.append(os.getcwd())

from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test # noqa
from memory_profiler import memory_usage # noqa
from unimodals.common_models import MLP, GRU # noqa
from datasets.mimic.get_data import get_dataloader # noqa
from fusions.common_fusions import Concat # noqa
from training_structures.Supervised_Learning import train, test # noqa

# get dataloader for icd9 classification task 7
traindata, validdata, testdata = get_dataloader(
    7, imputed_path='datasets/mimic/im.pk')

# build encoders, head and fusion layer
encoders = [MLP(5, 10, 10, dropout=False).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), GRU(
    12, 30, dropout=False, batch_first=True).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]
head = MLP(730, 40, 2, dropout=False).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
fusion = Concat().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
allmodules = [encoders[0], encoders[1], head, fusion]

# train


def trainprocess():
    train(encoders, fusion, head, traindata, validdata, 20, auprc=True)


all_in_one_train(trainprocess, allmodules)


# test
print("Testing: ")
model = torch.load('best.pt').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
# dataset = 'mimic mortality', 'mimic 1', 'mimic 7'
test(model, testdata, dataset='mimic 7', auprc=True)
