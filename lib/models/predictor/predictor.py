import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers  # 3
        self.input_dim = input_dim
        dims = [hidden_dim] * (num_layers - 1)  # [256, 256]
        
        layers = []
        for i in range(num_layers-1):
            layer = nn.Linear(input_dim, dims[i], bias=True)
            layers.append(layer)
            act_layer = nn.ReLU(inplace=True)
            layers.append(act_layer)
            input_dim = dims[i]

        layer = nn.Linear(input_dim, output_dim, bias=True)
        layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
        return out  # (B,5,1001)



def build_predictor(cfg):
    if cfg.MODEL.HEAD.TYPE == "MLP":
        hidden_dim = cfg.MODEL.HEAD.NUM_CHANNELS
        dim_out = cfg.MODEL.DECODER.VOCAB_SIZE
        mlp_head = MLP(hidden_dim, hidden_dim, dim_out, 3)  # dim_in, dim_hidden, dim_out, 3 layers
        
        return mlp_head
    else:
        raise ValueError("HEAD TYPE %s is not supported." % cfg.MODEL.HEAD_TYPE)