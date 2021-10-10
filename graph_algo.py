"""
Graph Algorithm Library

This file contains all the algorithms used for the GraphBandits simulations.
All algorithms can be used as modules.

"""

import numpy as np
import support_func
import algobase

with_reset = False
imperfect_graph_info = True


class MaxVarianceArmAlgo(algobase.AlgoBase):
    """
    Cyclic algorithm with mean estimation using Laplacian.
    """
    def select_arm(self):
        """
        Spectral bandits [Valko el.at] based arm sampling from the remaining set of arms.
        """
        remaining_width = np.zeros(self.dim)
        for i in self.remaining_nodes:
            remaining_width[i] = self.conf_width[i]
        play_index = np.argmax(remaining_width)

        return play_index


class NoGraphAlgo(algobase.AlgoBase):
    """
    Cyclic algorithm with mean estimation using Laplacian.
    """

    def __init__(self, D, A, mu, eta):
        """

        Parameters
        ----------
        D : Degree matrix
        A : Adjacency matrix
        mu : node-mean vector
        eta : Penalty parameter for mean estimation

        """
        super().__init__(D, A, mu, eta, add_graph=False)

    def select_arm(self):
        """
        Cyclic arm selection from the remaining set of arms.
        """

        # TODO : Switch current algorithm with true cyclic. Currently only works for symmetric graphs.

        remaining_width = np.zeros(self.dim)
        for i in self.remaining_nodes:
            remaining_width[i] = self.conf_width[i]
        play_index = np.argmax(remaining_width)

        return play_index


class CyclicAlgo(algobase.AlgoBase):
    """
    Spectral bandits [Valko el.at] based graph elimination algorithm with mean estimation using Laplacian.
    """

    def select_arm(self):
        """
        Spectral bandits [Valko el.at] based arm sampling from the remaining set of arms.
        """

        next_index= 0
        play_index = self.remaining_nodes[next_index%len(self.remaining_nodes)]

        if len(self.picking_order) > 1:
            last_index = self.picking_order[-1]
            ind = np.where(self.jumping_index == last_index)
            ind = (int(ind[0]) + 1)%len(self.jumping_index)
            while self.jumping_index[ind] not in self.remaining_nodes:
                ind +=1
                ind = ind%len(self.jumping_index)
            play_index = self.jumping_index[ind]

        self.picking_order.append(play_index)

        return play_index


class MaxDiffVarAlgo(algobase.AlgoBase):
    """
    Proposed graph elimination algorithm with mean estimation using Laplacian.
    """

    def opti_selection(self):
        """
        Proposed arm selection criteria based on the ensemble reduction of confidence width.
        """

        # TODO : Replace costly inverse computation using Sherman-Morrison formula.
        A = self.remaining_nodes
        options =[]
        for i in A:
            new_vec = np.zeros(self.dim)
            new_vec[i] = 1
            current = support_func.sherman_morrison_inverse(new_vec, self.inverse_tracker)
            options.append(np.linalg.det(current))
        index = np.argmin(options)
        return np.array(A)[index]

    def select_arm(self):
        """
        Select arm to play based on proposed ensemble confidence width reduction criteria.
        """
        remaining_width = np.zeros(self.dim)
        for i in self.remaining_nodes:
            remaining_width[i] = self.conf_width[i]
        play_index = self.opti_selection()

        return play_index


class OneStepMinDetAlgo(algobase.AlgoBase):
    """
    Proposed graph elimination algorithm with mean estimation using Laplacian.
    """

    def opti_selection(self):
        """
        Proposed arm selection criteria based on the ensemble reduction of confidence width.
        """

        # TODO : Replace costly inverse computation using Sherman-Morrison formula.
        A = self.remaining_nodes
        options =[]
        for i in A:
            new_vec = np.zeros(self.dim)
            new_vec[i] = 1
            current = support_func.sherman_morrison_inverse(new_vec, self.inverse_tracker)
            options.append(np.linalg.det(current))
        index = np.argmin(options)
        return np.array(A)[index]

    def select_arm(self):
        """
        Select arm to play based on proposed ensemble confidence width reduction criteria.
        """
        remaining_width = np.zeros(self.dim)
        for i in self.remaining_nodes:
            remaining_width[i] = self.conf_width[i]
        play_index = self.opti_selection()

        return play_index


class OneStepMinSumAlgo(algobase.AlgoBase):
    """
    Proposed graph elimination algorithm with mean estimation using Laplacian.
    """

    def opti_selection(self):
        """
        Proposed arm selection criteria based on the ensemble reduction of confidence width.
        """

        # TODO : Replace costly inverse computation using Sherman-Morrison formula.
        A = self.remaining_nodes
        options =[]
        for i in A:
            new_vec = np.zeros(self.dim)
            new_vec[i] = 1
            current = support_func.sherman_morrison_inverse(new_vec, self.inverse_tracker)
            options.append(sum([current[j, j] for j in A]))
        index = np.argmin(options)
        return np.array(A)[index]

    def select_arm(self):
        """
        Select arm to play based on proposed ensemble confidence width reduction criteria.
        """
        remaining_width = np.zeros(self.dim)
        for i in self.remaining_nodes:
            remaining_width[i] = self.conf_width[i]
        play_index = self.opti_selection()

        return play_index