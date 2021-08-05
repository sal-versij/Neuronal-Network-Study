class NeuralNetwork:
    def __init__(self):
        self.layers = []
    def addLayer(self, layer, activation):
        self.layers.append((layer,activation))
    def forward(self,inputs):
        current = inputs
        for (layer,activation) in self.layers:
            layer.forward(current)
            activation.forward(layer.output)
            current = activation.output
        self.output = current
    def calculateLoss(self, loss_function, y_true):
        self.loss = loss_function.calculate(self.output, y_true)