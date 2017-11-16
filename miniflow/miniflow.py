import numpy as np


class Node(object):
    def __init__(self, inbound_nodes=[]):
        self.inbound_nodes = inbound_nodes
        self.outbound_nodes = []
        self.gradients = {}
        # when adding out nodes add to forward nodes
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)
        self.value = None

    def forward(self):
        raise NotImplemented

    def backward(self):
        raise NotImplemented


class Input(Node):
    def __init__(self):
        Node.__init__(self)

    def forward(self, value=None):
        if value is not None:
            self.value = value

    def backward(self):
        self.gradients = {self: 0}
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self] += grad_cost * 1


class Add(Node):
    def __init__(self, *input):
        Node.__init__(self, input)

    def forward(self):
        sum = 0
        for n in self.inbound_nodes:
            sum += n.value
        self.value = sum


class Mul(Node):
    def __init__(self, *input):
        Node.__init__(self, input)

    def forward(self):
        mul = 1
        for n in self.inbound_nodes:
            mul *= n.value
        self.value = mul

    def backward(self):
        pass


class Linear(Node):
    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])

    def forward(self):
        X = self.inbound_nodes[0]
        W = self.inbound_nodes[1]
        b = self.inbound_nodes[2]
        self.value = np.dot(X.value, W.value) + b.value

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)
            self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)
            self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)


class Sigmoid(Node):
    def __init__(self, input):
        Node.__init__(self, [input])

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(0 - x))

    def forward(self):
        f = self.inbound_nodes[0]
        self.value = self._sigmoid(f.value)

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}

        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            f = self._sigmoid(self.inbound_nodes[0].value)
            self.gradients[self.inbound_nodes[0]] += grad_cost * f * (1 - f)


class MSE(Node):
    def __init__(self, y, a):
        Node.__init__(self, [y, a])

    def forward(self):
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)
        self.m = self.inbound_nodes[0].value.shape[0]
        self.diff = y - a
        self.value = np.mean(np.square(y - a))

    def backward(self):
        self.gradients[self.inbound_nodes[0]] = (2 / self.m) * self.diff
        self.gradients[self.inbound_nodes[1]] = (-2 / self.m) * self.diff


def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_and_backward(graph):
    """
    Performs a forward pass through a list of sorted nodes.

    Arguments:

        `output_node`: A node in the graph, should be the output node (have no outgoing edges).
        `sorted_nodes`: A topologically sorted list of nodes.

    Returns the output Node's value
    """

    for n in graph:
        n.forward()

    for n in graph[::-1]:
        n.backward()


def sgd_update(trainables, learning_rate=1e-2):
    """
    Updates the value of each trainable with SGD.

    Arguments:

        `trainables`: A list of `Input` Nodes representing weights/biases.
        `learning_rate`: The learning rate.
    """
    # TODO: update all the `trainables` with SGD
    # You can access and assign the value of a trainable with `value` attribute.
    # Example:
    # for t in trainables:
    #   t.value = your implementation here
    for t in trainables:
        t.value = t.value - learning_rate * t.gradients[t]
