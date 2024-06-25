from typing import Union, Literal, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup, BertTokenizer, BertModel, BertConfig


class BERTFinetune(nn.Module):
    """
        BERT for classification tasks
    """
    def __init__(self,
                 freeze_bert: bool = False,
                 dim_in: int = 768,
                 dim_out: int = 1,
                 p_dropout: int = 0.1,
                 model_name: str = "bert-base-uncased",
                 pooling_type: Literal["CLS", "ATT", "MEAN", "MAX", "MEAN-MAX", "CONV"] = "CLS",
                 hidden_dropout_prob: float = 0.007,
                 attention_probs_dropout_prob: float = 0.007):

        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.model_name = model_name
        self.pooling_type = pooling_type

        self.linear = nn.Linear(dim_in, dim_out)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p_dropout)

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.linear.apply(init_weights)        

    def forward(self, input_ids, attention_mask):
        """
            Feed input to BERT and the classifier to compute logits.
            @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                            max_length)
            @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                            information with shape (batch_size, max_length)
            @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                            num_labels)
        """

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        if self.pooling_type != "CLS":
            last_hidden_states = outputs[0] # -> torch.Size([1, <n_tokens>, 768])

            if self.pooling_type == "MEAN":
                pooled_last_hidden_state = torch.mean(last_hidden_states, dim=2) # -> torch.Size([1, <n_tokens>, 1])
            
            elif self.pooling_type == "MAX":
                pooled_last_hidden_state = torch.max(last_hidden_states, dim=2) # -> torch.Size([1, <n_tokens>, 1])

            elif self.pooling_type == "MEAN-MAX":
                last_hidden_state_mean = torch.mean(last_hidden_states, dim=2) # -> torch.Size([1, <n_tokens>, 1])
                last_hidden_state_max = torch.max(last_hidden_states, dim=2) # -> torch.Size([1, <n_tokens>, 1])
                pooled_last_hidden_state = torch.cat((last_hidden_state_mean, last_hidden_state_max), dim=3) # -> torch.Size([1, <n_tokens>, 2])

            elif self.pooling_type == "CONV":
                pooled_last_hidden_state = nn.Conv1d(in_channels=last_hidden_states.size()[1], kernel_size=2, padding=1)

            elif self.pooling_type == "ATT":
                raise NotImplementedError
            
            linear = self.linear(pooled_last_hidden_state)
            
        else:
            last_hidden_state = outputs[0][:, 0, :] # -> torch.Size([1, 768])
            # last_hidden_state = last_hidden_state.unsqueeze(1) # Add dimension to match further layers size -> torch.Size([1, 1, 768])
            linear = self.linear(last_hidden_state).squeeze(1)

        output = self.dropout(linear)

        output = self.sigmoid(output)

        return output

        
class BERTDataset(Dataset):
    def __init__(self, input: Union[List[str], ], labels: List[Union[int, float]], tokenizer, max_length=512,
                 padding: Literal[True, False, "max_length"] = "max_length",
                 truncation: Literal[True, False] = True):
        self.labels = torch.tensor(labels, dtype=torch.float)
        self.input_ids, self.attn_masks = self.tokenize_function(input=input, tokenizer=tokenizer, max_length=max_length, padding=padding, truncation=truncation)
    
    def tokenize_function(self,
                      input,
                      tokenizer,
                      padding: Literal[True, False, "max_length"] = False,
                      truncation: Literal[True, False] = True,
                      max_length: int = 512) -> Tuple[torch.tensor, torch.tensor]:

        tokenized = tokenizer(input,
                            padding=padding,
                            truncation=truncation,
                            max_length=max_length)
        
        input_ids = tokenized.get('input_ids')
        attention_masks = tokenized.get('attention_mask')

        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        return input_ids, attention_masks

    @property
    def __max__(self) -> Union[None, float, int]:
        if not self.labels.numel():
            return None
        return torch.max(self.labels).item()
    
    @property
    def __min__(self) -> Union[None, float, int]:
        if not self.labels.numel():
            return None
        return torch.min(self.labels).item()
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.labels[idx]
    