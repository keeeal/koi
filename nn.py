
import numpy as np
import torch

class Brain(object):
    def __init__(self, input_size):
        super(Brain, self).__init__()

        self.model = torch.nn.Sequential(
                        torch.nn.Linear(input_size, 16),
                        torch.nn.ReLU(),
                        torch.nn.Linear(16, 32),
                        torch.nn.ReLU(),
                        torch.nn.Linear(32, 8),
                        torch.nn.ReLU(),
                        torch.nn.Linear(8, 2),
                        torch.nn.Tanh(),
                    )

        self.parameters = [2*np.random.random(p.shape)-1 for p in self.parameters]

    def __call__(self, input):
        input = torch.tensor(input, dtype=torch.float32)
        output = self.model(input)
        return output[0].item(), (output[1].item() + 1)/2

    def parameters():

        def fget(self):
            return [p.detach().numpy() for p in self.model.parameters()]

        def fset(self, new):
            layers = (layer for layer in self.model if len(list(layer.parameters())))

            for layer, (w, b) in zip(layers, zip(new[::2], new[1::2])):
                layer.weight.data.set_(torch.tensor(w, dtype=torch.float32))
                layer.bias.data.set_(torch.tensor(b, dtype=torch.float32))

        return locals()

    parameters = property(**parameters())

def test():
    pass

if __name__ == '__main__':
    test()
