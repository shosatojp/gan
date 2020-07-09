# In[1]
from sagan_train import train_model, weights_init, check
from dataloader import train_dataloader
from sagan_discriminator import Discriminator
from sagan_generator import Generator
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
    num_epochs=200
)
# In[40]:

while True:
    check(G)

