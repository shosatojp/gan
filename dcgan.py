# In[1]
from train import train_model, weights_init, validate
from dataloader import train_dataloader
from discriminator import Discriminator
from generator import Generator
import random
import numpy as np
import torch

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

G = Generator()
G.apply(weights_init)

D = Discriminator(64)
D.apply(weights_init)

G_update, D_update = train_model(
    G, D,
    dataloader=train_dataloader,
    num_epochs=500
)
# In[40]:

while True:
    validate(G)
