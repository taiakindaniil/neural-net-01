import numpy
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
from progress.bar import Bar

class neural_network:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.learning_rate = learning_rate

        # creating weight matrices
        # weights for input -> hidden layers
        # self.wih = numpy.random.rand(self.hidden_nodes, self.input_nodes) - 0.5
        self.wih = numpy.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        # weights for hidden -> output layers
        # self.who = numpy.random.rand(self.output_nodes, self.hidden_nodes) - 0.5
        self.who = numpy.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

        self.activation_function = lambda x: sigmoid(x)

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.learning_rate * numpy.dot((output_errors * final_outputs * (1 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.learning_rate * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), numpy.transpose(inputs))

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

def train_neural_net(n: neural_network):
    with open("./mnist_train.csv", "r") as f:
        lines = f.readlines()
        bar = Bar('Training', max=len(lines))
        for line in lines:
            all_values = line.split(",")
            image_array = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01) #.reshape((28,28))

            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99

            n.train(image_array, targets)
            bar.next()
        bar.finish()

if __name__ == "__main__":
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    learning_rate = 0.1

    n = neural_network(input_nodes, hidden_nodes, output_nodes, learning_rate)

    epochs = 5

    for i in range(epochs):
        print(f"{i} / {epochs} Epochs")
        train_neural_net(n)
    
    scoreboard = []
    with open("./mnist_test.csv", "r") as f:
        lines = f.readlines()
        bar = Bar('Testing', max=len(lines))
        for line in lines:
            all_values = line.split(",")
            correct_label = int(all_values[0])
            image_array = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01) #.reshape((28,28))

            # plt.imshow(image_array, cmap="Greys")
            # plt.show()
            # break

            outputs = n.query(image_array)
            label = numpy.argmax(outputs)

            if label == correct_label:
                scoreboard.append(1)
            else:
                scoreboard.append(0)
            
            bar.next()
        bar.finish()
    
    scoreboard_array = numpy.asfarray(scoreboard)
    print("Network Efficiency:", scoreboard_array.sum() / scoreboard_array.size)