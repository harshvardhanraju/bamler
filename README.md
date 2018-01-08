### bamler
**Community built hobby machine learning library**

This repository contains implementations of common machine learning algorithms.

Built by the members of Bangalore Advanced Machine Learning meetup (BAML)[https://www.meetup.com/baml-meetup]


## Installation

`pip install bamler`

## Usage

```python
>>> from bamler.nn import network
>>> from bamler.nn import layer
>>> mnist_network = network()
>>> input_layer = layer(nodes_num=3, nodes_type='input')
>>> first_hidden_layer = layer(nodes_num=10, nodes_type='hidden', activation='linear')
>>> second_hidden_layer = layer(nodes_num=10, nodes_type='hidden', activation='linear')
>>> output_layer = layer(nodes_num=10, nodes_type='output')
>>> mnist_network.add_layer(input_layer)
>>> mnist_network.add_layer(first_hidden_layer)
>>> mnist_network.add_layer(second_hidden_layer)
>>> mnist_network.add_layer(output_layer)
>>> mnist_network.train()
>>> mnist_network.predict([0.1, 0.4, 0.7])
4
```
