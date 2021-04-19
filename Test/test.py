"""
Unittest setup.

Placeholder example test for now. Need to build up on it.
"""


import sys
sys.path.append('../')

import graph_algo
import support_func
import unittest
import toml
import numpy as np


class TestGraphAlgo(unittest.TestCase):

    def test_one(self):
        self.assertEqual(1, 1)

    def test_function(self):
        data = toml.load("../sample_config.toml")
        graph = data['1vw_med_graph_SBM_static']
        self.assertEqual(graph['nodes'], 100)


if __name__ == "__main__":

    unittest.main()
