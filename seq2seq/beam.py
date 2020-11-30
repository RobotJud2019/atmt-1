import torch
import numpy as np

from itertools import count
from queue import PriorityQueue

import copy


class BeamSearch(object):
    """ Defines a beam search object for a single input sentence. """
    def __init__(self, beam_size, max_len, pad):

        self.beam_size = beam_size
        self.max_len = max_len
        self.pad = pad

        self.alpha = 0.7
        self.gamma = 0.7

        self.nodes = PriorityQueue() # beams to be expanded
        self.final = PriorityQueue() # beams that ended in EOS

        self._counter = count() # for correct ordering of nodes with same score

    def add(self, score, node):
        """ Adds a new beam search node to the queue of current nodes """
        lp = ((5 + node.length)** self.alpha ) / (( 5 + 1 ) **  self.alpha)
        self.nodes.put((score/lp, next(self._counter), node))

    def add_final(self, score, node):
        """ Adds a beam search path that ended in EOS (= finished sentence) """
        # ensure all node paths have the same length for batch ops
        missing = self.max_len - node.length
        node.sequence = torch.cat((node.sequence.cpu(), torch.tensor([self.pad]*missing).long()))
        lp = ((5 + node.length)** self.alpha ) / (( 5 + 1 ) **  self.alpha)
        self.final.put((score/lp, next(self._counter), node))

    def get_current_beams(self):
        """ Returns beam_size current nodes with the lowest negative log probability """
        nodes = []
        while not self.nodes.empty() and len(nodes) < self.beam_size:
            node = self.nodes.get()
            nodes.append((node[0], node[2]))
        return nodes

    def get_bestN(self, N):
        """ Returns final node with the lowest negative log probability """
        # Merge EOS paths and those that were stopped by
        # max sequence length (still in nodes)
        merged = PriorityQueue()
        for i in range(N):
            if self.final.empty(): continue
            node = self.final.get()
            if node[0].data <= self.gamma * i: node[0].data = torch.tensor(0.01)
            else: node[0].data = node[0].data - self.gamma * i
            merged.put(node)

        if not self.nodes.empty:
            for j in range(np.min(N, self.nodes.qsize())):
                node = self.nodes.get() 
                if node[0].data <= self.gamma * j: node[0].data = torch.tensor(0.01)
                else: node[0].data = node[0].data - self.gamma * j
                merged.put(node)

        # modify here to retun a list of N nodes instead of 1 node
        Nnodes = []
        while (merged.qsize() > 0): 
            n = merged.get()
            n = (n[0], n[2])
            Nnodes.append(n)

        return Nnodes

    def get_best(self):
        """ Returns final node with the lowest negative log probability """
        # Merge EOS paths and those that were stopped by
        # max sequence length (still in nodes)
        print("in get_best, self.final.qsize():", self.final.qsize(), "   self.nodes.qsize():", self.nodes.qsize())
        merged = PriorityQueue()
        for _ in range(self.final.qsize()):
            node = self.final.get()
            merged.put(node)

        for _ in range(self.nodes.qsize()):
            node = self.nodes.get()
            merged.put(node)

        node = merged.get()
        node = (node[0], node[2])

        return node

    def prune(self):
        """ Removes all nodes but the beam_size best ones (lowest neg log prob) """
        nodes = PriorityQueue()
        # Keep track of how many search paths are already finished (EOS)
        finished = self.final.qsize()
        for _ in range(self.beam_size-finished):
            node = self.nodes.get()
            nodes.put(node)
        self.nodes = nodes


class BeamSearchNode(object):
    """ Defines a search node and stores values important for computation of beam search path"""
    def __init__(self, search, emb, lstm_out, final_hidden, final_cell, mask, sequence, logProb, length):

        # Attributes needed for computation of decoder states
        self.sequence = sequence
        self.emb = emb
        self.lstm_out = lstm_out
        self.final_hidden = final_hidden
        self.final_cell = final_cell
        self.mask = mask

        # Attributes needed for computation of sequence score
        self.logp = logProb
        self.length = length

        self.search = search

    def eval(self):
        """ Returns score of sequence up to this node """
        return self.logp
