import torch
import torch.nn as nn
import torch.nn.functional as F


num_action_classes = 5


class MaskedPredictionHead(nn.Module):
    def __init__(self, embed_dim, num_action_classes, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.state_output_layer = nn.Linear(embed_dim, 2)  # Output 2 values for state coordinates (regression)
        self.action_output_layer = nn.Linear(embed_dim, num_action_classes)  # Output for actions (classification)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, unmasked_indices, masked_state_indices, masked_action_indices):
        x = x.transpose(0, 1)
        unmasked_inputs = x[unmasked_indices, :]
        masked_state_inputs = x[masked_state_indices, :]
        masked_action_inputs = x[masked_action_indices, :]
        
        # Attention for state and action inputs separately
        state_attn_output, _ = self.attention(masked_state_inputs, unmasked_inputs, unmasked_inputs)
        action_attn_output, _ = self.attention(masked_action_inputs, unmasked_inputs, unmasked_inputs)

        state_attn_output = self.dropout(state_attn_output)
        action_attn_output = self.dropout(action_attn_output)
        
        state_attn_output = state_attn_output.transpose(0, 1)
        action_attn_output = action_attn_output.transpose(0, 1)
        
        # State predictions
        state_predictions = self.state_output_layer(state_attn_output)
        state_predictions = self.sigmoid(state_predictions) * 9  # Scale output to [0, 9]
        
        # Action predictions
        action_predictions = self.action_output_layer(action_attn_output)
        action_predictions = F.softmax(action_predictions, dim=-1)
        
        #print("Predictions from the model:")
        #print(state_predictions, action_predictions)
        return state_predictions, action_predictions

class StateActionEmbedding(nn.Module):
    def __init__(self, state_dim, action_dim, embed_dim):
        super().__init__()
        self.state_embedding = nn.Linear(state_dim, embed_dim)
        self.action_embedding = nn.Linear(action_dim, embed_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.state_embedding.weight)
        nn.init.xavier_uniform_(self.action_embedding.weight)

    def forward_state(self, state):
        return self.state_embedding(state)

    def forward_action(self, action):
        return self.action_embedding(action.float())  # Convert action to float
class ForwardDynamicHead(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.output_layer = nn.Linear(embed_dim, 2)  # Output 2 values for state coordinates
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context_indices, target_index):
        # Ensure the tensors have the correct shape: [sequence_length, batch_size, embed_dim]
        target = x[:, target_index, :].unsqueeze(0)  # Shape: [1, batch_size, embed_dim]
        context = x[:, context_indices, :].transpose(0, 1)  # Shape: [num_context_items, batch_size, embed_dim]
        
        attn_output, _ = self.attention(target, context, context)
        attn_output = self.dropout(attn_output)
        predictions = self.output_layer(attn_output.squeeze(0))  # Remove the sequence length dimension
        predictions = self.sigmoid(predictions) * 9  # Scale output to [0, 9]
        return predictions



class InverseDynamicHead(nn.Module):
    def __init__(self, embed_dim, num_classes, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.output_layer = nn.Linear(embed_dim, num_classes)  # num_classes instead of output_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context_indices, target_index):
        target = x[:, target_index, :].unsqueeze(0)  # Shape: [1, batch_size, embed_dim]
        context = x[:, context_indices, :].transpose(0, 1)  # Shape: [num_context_items, batch_size, embed_dim]
        attn_output, _ = self.attention(target, context, context)
        attn_output = self.dropout(attn_output)
        logits = self.output_layer(attn_output.squeeze(0))
        return logits  # Return raw logits

class TransformerPredictor(nn.Module):
    def __init__(self, state_embed_dim, state_dim, action_dim, num_heads, num_layers, num_action_classes, dropout=0.1):
        super().__init__()
        self.embedding = StateActionEmbedding(state_dim=state_dim, action_dim=action_dim, embed_dim=state_embed_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=state_embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.output_layer = nn.Linear(state_embed_dim, num_action_classes)
        self.forward_dynamic_head = ForwardDynamicHead(state_embed_dim, num_heads, dropout)
        self.inverse_dynamic_head = InverseDynamicHead(state_embed_dim, num_action_classes, num_heads, dropout)
        self.masked_prediction_head = MaskedPredictionHead(state_embed_dim, num_action_classes, num_heads, dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, states, actions):
        batch_size, seq_length, _ = states.size()
        embed = torch.zeros(batch_size, (seq_length * 2), self.embedding.state_embedding.out_features, device=states.device)

        for i in range(seq_length):
            embed[:, i * 2, :] = self.embedding.forward_state(states[:, i, :])
            embed[:, i * 2 + 1, :] = self.embedding.forward_action(actions[:, i].unsqueeze(1))
        
        transformer_output = self.transformer(embed)
        logits = self.output_layer(transformer_output[:, -1, :])
        action_prediction = F.softmax(logits, dim=-1)  # Apply softmax to get probabilities

        forward_predictions = torch.zeros(batch_size, seq_length - 1 , 2, device=states.device)
        inverse_predictions = torch.zeros(batch_size, seq_length - 1, num_action_classes, device=states.device)

        for i in range(0, seq_length-1):
            i = i * 2 
            embed_slice = embed[:,[i,i+1,i+2],:]
            forward_predictions[:,i//2] = self.forward_dynamic_head(embed_slice, [0, 1], 2)
            inverse_predictions[:,i//2] = self.inverse_dynamic_head(embed_slice, [0, 2], 1)

        # Mask some states and actions
        #unmasked_indices, masked_state_indices, masked_action_indices = self.random_mask(embed, seq_length)
        #masked_state_predictions, masked_action_predictions = self.masked_prediction_head(embed, unmasked_indices, masked_state_indices, masked_action_indices)
        #combined_mask_indices = torch.cat([mask_indices_states * 2, mask_indices_actions * 2 + 1])
        #unmasked_indices = torch.tensor([i for i in range((seq_length * 2) + 1) if i not in combined_mask_indices], device=states.device)

        return action_prediction, forward_predictions, F.softmax(inverse_predictions, dim=-1)
        #return action_prediction, forward_predictions, F.softmax(inverse_predictions, dim=-1), masked_state_predictions, masked_action_predictions, unmasked_indices, unmasked_indices

    def random_mask(self, x, seq_length):
                
        # Calculate mask sizes
        mask_size_states = seq_length // 4 + 1
        mask_size_actions = seq_length // 4 - 1
        
        # Ensure the total number of masks does not exceed seq_length
        assert mask_size_states + mask_size_actions <= seq_length, "Total mask size exceeds sequence length"

        # Generate all state indices (odd indices) and action indices (even indices)
        all_state_indices = torch.arange(1, seq_length, 2)
        all_action_indices = torch.arange(0, seq_length, 2)

        # Shuffle indices
        shuffled_state_indices = all_state_indices[torch.randperm(len(all_state_indices))]
        shuffled_action_indices = all_action_indices[torch.randperm(len(all_action_indices))]

        # Select indices for states and actions ensuring no overlap
        mask_indices_states = shuffled_state_indices[:mask_size_states]
        mask_indices_actions = shuffled_action_indices[:mask_size_actions]

        # Combine masked indices for states and actions
        masked_indices = torch.cat([mask_indices_states, mask_indices_actions])

        # Generate unmasked indices
        unmasked_indices = torch.tensor([i for i in range(seq_length) if i not in masked_indices])

        return unmasked_indices, mask_indices_states, mask_indices_actions


