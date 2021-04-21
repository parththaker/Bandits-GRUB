<!-- ABOUT THE PROJECT -->

## About The Project

We consider the problem of Best-Arm Identification (BAI) in the prescence of imperfect side information in the form of a graph.

From a practical stand-point, here's why this is interesting:
* This approach helps bandit applications suffering from having too many sub-optimal choices. 
* No strict modelling assumption (like linear bandits, etc.) is needed.
* Superior experimental evidence backed by provably better sample complexity bounds.

This is the code base used for showing the superior experimental evidence.

##Getting started

#### System Requirements :
The following packages in python 3.6+ should be sufficent to run the simulations:
* numpy 1.19+
* scipy 1.5+
* matplotlib
* networkx 2.5+

#### Running base experiment :
The base file is `sample_main.py` can be performed by running
* npm
  ```sh
  python3 sample_main.py
  ```
 
The default configuration is as follows:
 * Total nodes = 101
 * Total clusters = 10
 * Nodes per cluster = 10
 * 1 isolated optimal node
 * Every cluster is a complete graph

#### Other possible setups : 
Apart from tinkering with the node/cluster values, the different type of graphs which can be setup for the experiments are as follows:
* Tree graph
* Star graph
* Wheel graph
* Complete graph (default)
* Erdos-Renyi graph (with parameter p)
* Stochastic Bloch Model (with parameters p, q)
* Barabasi-Albert graph (with m=2)

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

[Parth Thaker](https://parththaker.github.io/)

Project Link: [https://github.com/parththaker/GraphBAI](https://github.com/parththaker/GraphBAI)

 