import torch
from torchviz import make_dot
from graphviz import Digraph

def plot_bert_architecture(model, max_size: int = 512,
                            min_index: int = 0,
                            max_index: int = 1500,
                            attention_mask_threshold: float = 0.1,
                            show_graphviz_attrs: bool = False,
                            show_graphviz_saved: bool = False) -> Digraph:
    rand_input_ids = torch.randint(min_index, max_index, (1, max_size))
    rand_attention_mask = rand_input_ids > attention_mask_threshold
    output = model(rand_input_ids, rand_attention_mask)
    return make_dot(output, params=dict(model.named_parameters()), show_attrs=show_graphviz_attrs, show_saved=show_graphviz_saved)