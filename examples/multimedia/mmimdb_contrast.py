import torch
import sys
import os

sys.path.append(os.getcwd())

from unimodals.common_models import MLP, MaxOut_MLP, Linear
from datasets.imdb.get_data import get_dataloader
from fusions.common_fusions import Concat
from objective_functions.objectives_for_supervised_learning import RefNet_objective
from training_structures.Supervised_Learning import train, test


filename = "best_contrast.pt"
traindata, validdata, testdata = get_dataloader(
    "../video/multimodal_imdb.hdf5", "../video/mmimdb", vgg=True, batch_size=128)

encoders = [MaxOut_MLP(512, 512, 300, linear_layer=False),
            MaxOut_MLP(512, 1024, 4096, 512, False)]
head = Linear(1024, 23).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
refiner = MLP(1024, 3072, 4396).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
fusion = Concat().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

train(encoders, fusion, head, traindata, validdata, 1000, [refiner], early_stop=True, task="multilabel", save=filename, objective_args_dict={"refiner": refiner},
      optimtype=torch.optim.AdamW, lr=1e-2, weight_decay=0.01, objective=RefNet_objective(0.1, torch.nn.BCEWithLogitsLoss()))

print("Testing:")
model = torch.load(filename).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
test(model, testdata, method_name="refnet", dataset='imdb',
     criterion=torch.nn.BCEWithLogitsLoss(), task="multilabel")
