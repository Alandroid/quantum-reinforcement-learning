import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer


dev = qml.device("default.qubit", wires=4)


def layer(W):

    # Rotating all qubits by a tunable parameter

    # TODO: are we hardcoding 4 qubits? Is this the optimal ansatz? Maybe we can vary this
    qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
    qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
    qml.Rot(W[2, 0], W[2, 1], W[2, 2], wires=2)
    qml.Rot(W[3, 0], W[3, 1], W[3, 2], wires=3)

    # Entangling each qubit with its neighbour
    # TODO: test with a 2-layered cell ({{0, 1}, {2, 3}}, {{1, 2}, {3, 0}}) - lighter implementation
    # Symmetry: better for universal computation, but it depends on the problem
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])


# Encoding classical data into the VQC
def state_preparation(x):
    qml.BasisState(x, wires=[0, 1, 2, 3])


@qml.qnode(dev)
def circuit(weights, x): # Using weights to name the tunable parameters, just like ML
# Here x is a keyword arg, so it's ignored by the gradient (not trained)

    state_preparation(x)

    for W in weights:
        layer(W)

    return qml.expval(qml.PauliZ(0))


def variational_classifier(weights, bias, x): # Classical bias b_L: a_L = \sigma{w_L * a_{L-1} + b_L}
    return circuit(weights, x) + bias


def square_loss(labels, predictions):
    """
    For each labeled data point in the training set, return the squared difference
    between the output and the expected label. Then sum over all points in the training
    dataset, and return this sum
    """

    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p)**2

    loss = loss / len(labels)

    return loss


def accuracy(labels, predictions):
    """
    For each input point of the training dataset, return the % that got the
    output matching the expected label
    """

    correct_labels = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            correct_labels += 1
    
    correct_labels = correct_labels / len(labels)

    return correct_labels


def cost(weights, bias, X, Y):
    
    predictions = [variational_classifier(weights, bias, x) for x in X]

    return square_loss(Y, predictions)


#### Training code ####

data = np.loadtxt("data/parity.txt")

X = np.array(data[:, :-1], requires_grad=False)
Y = np.array(data[:, -1], requires_grad=False)
Y = 2 * Y - np.ones(len(Y)) # Shifting labels from {0, 1} to {-1, 1}

# A glimse into the dataset
#for i in range(5):
#    print("X = {}, Y = {: d}".format(X[i], int(Y[i])))
#print("...")

# Initializing the weights and the bias randomly,
# with a fixed seed for reproducibility
np.random.seed(0)
num_qubits = 4
num_layers = 2
weights_init = 0.01 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
bias_init = np.array(0.0, requires_grad=True)

print(weights_init, bias_init)


# Initializing the optimizer

# Nesterov Momentum is an adaptation of the momentum
# optimizer, which is, in turn, similar to the gradient
# descent, but building momentum in the best directions
opt = NesterovMomentumOptimizer(0.5)
batch_size = 5  # Using batched stochastic gradient descent for speedup

weights = weights_init
bias = bias_init

for it in range(25):

    # Updating the weights by one optimizer step - TODO understand
    batch_index = np.random.randint(0, len(X), (batch_size,))
    X_batch = X[batch_index]
    Y_batch = Y[batch_index]
    weights, bias, _, _ = opt.step(cost, weights, bias, X_batch, Y_batch)

    # The signal of the output is taken as the prediction (+-1)
    predictions = [np.sign(variational_classifier(weights, bias, x)) for x in X]
    acc = accuracy(Y, predictions)  # Compute accuracy

    print(
        "Iter: {:5d} | Cost: {:0.7f} | Accuracy: {:0.7f} ".format(
            it + 1, cost(weights, bias, X, Y), acc
        )
    )