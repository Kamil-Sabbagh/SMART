import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class StateActionEmbedding(nn.Module):
    def __init__(self, state_dim, action_dim, embed_dim):
        super(StateActionEmbedding, self).__init__()
        # Define separate linear layers for state and action
        self.state_embed = nn.Linear(state_dim, embed_dim)
        self.action_embed = nn.Linear(action_dim, embed_dim)
    
    def forward_state(self, state):
        # Flatten and embed the state
        state = state.view(state.size(0), -1)  # Reshape to (batch_size, 600)
        return self.state_embed(state)

    def forward_action(self, action):
        # Flatten and embed the action
        action = action.view(action.size(0), -1)  # Reshape to (batch_size, 100)
        return self.action_embed(action)

class TransformerPredictor(nn.Module):
    def __init__(self, state_embed_dim, action_embed_dim, num_heads, num_layers, num_actions):
        super(TransformerPredictor, self).__init__()
        self.embedding = StateActionEmbedding(state_dim=10*60, action_dim=10*10, embed_dim=state_embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=state_embed_dim + action_embed_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(state_embed_dim + action_embed_dim, num_actions)

    def forward(self, states, actions):
        seq_length = states.shape[1]
        embedded = torch.zeros(states.shape[0], seq_length, state_embed_dim + action_embed_dim, device=states.device)
        
        for i in range(seq_length):
            state_embedded = self.embedding.forward_state(states[:, i, :])
            action_embedded = self.embedding.forward_action(actions[:, i, :])
            embedded[:, i, :] = torch.cat((state_embedded, action_embedded), dim=-1)
        
        transformer_output = self.transformer(embedded)
        action_predictions = self.output_layer(transformer_output[:, -1, :])
        return action_predictions

def train_model(model, dataloader, optimizer, loss_fn, epochs=10, save_path=None):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for state_batch, action_batch, target_action_batch in dataloader:
            optimizer.zero_grad()
            action_predictions = model(state_batch, action_batch)
            loss = loss_fn(action_predictions, target_action_batch[:, -1, :])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}')

        if save_path:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            }, save_path)

def load_model(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return start_epoch, loss

# Example initialization and usage
model = TransformerPredictor(embed_dim=128, num_heads=8, num_layers=3, num_actions=3)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

# Assuming dataset and DataLoader setup here
# train_model(model, dataloader, optimizer, loss_fn, epochs=10, save_path='model_checkpoint.pth')

# Example loading (you need to create the optimizer before calling this)
# start_epoch, loss = load_model(model, optimizer, 'model_checkpoint.pth')
