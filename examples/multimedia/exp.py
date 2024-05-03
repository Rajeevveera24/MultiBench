import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import h5py
from lime.lime_tabular import LimeTabularExplainer
sys.path.append(os.getcwd())
from datasets_1.imdb.get_data import get_dataloader
from imdb_dyn import DynMMNet, DiffSoftmax
from sklearn.metrics import accuracy_score

model = DynMMNet()

traindata, validdata, testdata = get_dataloader("./data/multimodal_imdb.hdf5", "./data/mmimdb", vgg=True, batch_size=128, no_robust=True)

model_file_name = 'log/imdb/default_run_cnn/DynMMNet_freeze_True_reg_0.1.pt'

model = torch.load(model_file_name).cuda()
model.hard_gate = True

def predict(x):
    value = model.gate(torch.from_numpy(x).float().cuda())
    value = DiffSoftmax(value, tau=model.temp, hard=False).cpu().detach().numpy()
    return value

data = h5py.File("./data/multimodal_imdb.hdf5", 'r')
training_data = torch.cat([
    torch.Tensor(data["features"][:15552]), torch.Tensor(data["vgg_features"][:15552])
    ], dim=1)
testing_data = torch.cat([
    torch.Tensor(data["features"][18160:25959]), torch.Tensor(data["vgg_features"][18160:25959])
    ], dim=1)
exp = LimeTabularExplainer(training_data.numpy())
selected_labels = []
predicted_labels = []
for i in range(100):
    x = testing_data[i]
    a = exp.explain_instance(x.numpy(), predict, list(range(3)), num_samples=100)
    max_score = None
    selected_label = None
    for label in range(3):
        if max_score is None:
            max_score = sorted(a.as_map()[label], key=lambda x: x[1])[-1][1]
            selected_label = label
        else:
            score = sorted(a.as_map()[label], key=lambda x: x[1])[-1][1]
            if score > max_score:
                selected_label = label
                max_score = score
    selected_labels.append(selected_label)
    predicted_label = DiffSoftmax(model.gate(x.cuda()), tau=model.temp, hard=True).argmax()
    predicted_labels.append(predicted_label.detach().cpu().numpy().item())

print(selected_labels, predicted_labels)
agreement = accuracy_score(selected_labels, predicted_labels)
print(agreement)