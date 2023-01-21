from torch.nn import Sequential, ReLU, Linear, Module

class MLP(Module):
    def __init__(self, input_size, output_size, hidden_layer_size):
        super().__init__()
        self.layers = Sequential(
            Linear(input_size, hidden_layer_size),
            ReLU(),
            Linear(hidden_layer_size, hidden_layer_size),
            ReLU(),
            Linear(hidden_layer_size, output_size),
        )

    def forward(self, x):
        return self.layers(x)