"""
T5 finetuning on materials property prediction using materials text description 
"""
# Import packages
import torch
import torch.nn as nn
import torch.nn.functional as F

class Predictor(nn.Module):
    def __init__(
        self, 
        base_model, 
        base_model_output_size,  
        n_classes=1, 
        drop_rate=0.5, 
        freeze_base_model=False, 
        bidirectional=True, 
        pooling='cls',
        model_name='llmprop'
    ):
        super(Predictor, self).__init__()
        D_in, D_out = base_model_output_size, n_classes
        self.model = base_model
        self.dropout = nn.Dropout(drop_rate)
        self.pooling = pooling
        self.model_name = model_name

        # instantiate a linear layer
        self.linear_layer = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_in, D_out)
        )

    def forward(self, input_ids, attention_masks, x_num=None):
        hidden_states = self.model(input_ids, attention_masks)

        if self.model_name == 'llmprop':
            last_hidden_state = hidden_states.last_hidden_state # [batch_size, input_length, D_in]

            if x_num is not None:
                x_num = x_num.unsqueeze(2) #[batch_size, input_length] -> [batch_size, input_length, 1]
                last_hidden_state = last_hidden_state * x_num # [batch_size, input_length, D_in]

            if self.pooling == 'cls':
                input_embedding = last_hidden_state[:,0,:] # [batch_size, D_in] -- [CLS] pooling
            elif self.pooling == 'mean':
                input_embedding = last_hidden_state.mean(dim=1) # [batch_size, D_in] -- mean pooling
        
        elif self.model_name == 'matbert':
            pooled_hidden_state = hidden_states.pooler_output # [batch_size, D_in]-->the pooled embeddings for the first [CLS] token
            
            if x_num is not None:
                x_num = x_num.unsqueeze(2) #[batch_size, input_length] -> [batch_size, input_length, 1]
                pooled_hidden_state = pooled_hidden_state * x_num # [batch_size, input_length, D_in]
                
            input_embedding = pooled_hidden_state
        
        outputs = self.linear_layer(input_embedding) # [batch_size, D_out]

        return input_embedding, outputs