import numpy as np

class Activation:
    def forward(self,inputs):
        pass

class ReLU(Activation):
    def forward(self,inputs):
        self.output = np.maximum(0, inputs)

class SoftMax(Activation):
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)