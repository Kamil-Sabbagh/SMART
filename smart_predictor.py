import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os
from data_reader import DataLoader as CustomDataLoader
from model import TransformerPredictor


class smart_predictor():
    def __init__(self, path, config=None):
            if config is None:
                self.state_embed_dim = 20
                self.state_dim = 2 
                self.action_dim = 1  
                self.num_heads = 2
                self.num_layers = 2
                self.num_action_classes = 5
            
            self.saved_model_path = path
            self.model = TransformerPredictor()
            self.model.load_state_dict((torch.load(path)))
            self.model.eval()

    def predict(self, states, actions):
            # Create a random element with the same h and w dimensions
            random_element = torch.randn(1, 1, self.action_dim)

            # Concatenate the random element to the end of the original tensor
            actions_with_random = torch.cat((actions, random_element), dim=1)
            outputs = self.model(states, actions_with_random)
            action_predictions, _, _, _, _, _, _ = outputs
            return action_predictions
            
         



