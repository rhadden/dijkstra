#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:51:49 2020

@author: rileyhadden
"""
import numpy as np
import random
from matplotlib import pyplot as plt

class Node():
    def __init__(self, key):
        self.prev = None
        self.dist = np.inf
        self.index = -1
        self.children = []
        self.key = key
    

class MinHeap():
    def __init__(self):
        self._heap = [None]
        self._len = 0
    def check(self):
        for i in range(1, self._len):
            if self._heap[i].index != i:
                return False

            return True
    def make_queue(self, arr):
        for n in arr:
            self.insert(n)

    def insert(self, a):
        self._len += 1
        self._heap.append(a)
        # set index pointer for decrease key operation
        a.index = self._len
        self.bubble_up(self._len)
        
    def bubble_up(self, index):
        # print("BU Index: {}".format(index))
        if index == 1:
            return
        parent_idx = int(index // 2)
        try:
            self._heap[index].dist
            self._heap[parent_idx].dist
        except IndexError:
            print("Uh oh")
        if self._heap[index].dist < self._heap[parent_idx].dist:
            self.swap(index, parent_idx)
            if parent_idx == 1:
                return
            else:
                self.bubble_up(parent_idx)
    
    def pop(self):
        self.swap(self._len, 1)
        self._len -= 1

        out = self._heap.pop()
        self.trickle_down(1)
        return out
    
    def trickle_down(self, index):
        l_idx = index * 2
        r_idx = index * 2 + 1
        
        if l_idx > self._len:
            return
        
        elif l_idx == self._len:
            min_idx = l_idx
            
        else:
            if self._heap[r_idx].dist < self._heap[l_idx].dist:
                min_idx = r_idx
            else:
                min_idx = l_idx
                
        if self._heap[index].dist > self._heap[min_idx].dist:
            self.swap(index, min_idx)
            self.trickle_down(min_idx)
        # if not self.check():
        #     print("Trickle Down Err")
                
    def swap(self, idx1, idx2):
        temp = self._heap[idx1]
        self._heap[idx1] = self._heap[idx2]
        self._heap[idx2] = temp
        # Update pointers within heap
        self._heap[idx1].index = idx1
        self._heap[idx2].index = idx2
        # if not self.check():
        #     print("Swap Err")
        
    def decrease_key(self, node, val):
        node.dist = val
        self.bubble_up(node.index)
        # if not self.check():
        #     print("Decrease key err")

    def count(self):
        return self._len


class PQArray():
    def __init__(self):
        self.q = []

    def make_queue(self, arr):
        for n in arr:
            self.insert(n)

    def insert(self, a):
        self.q.append(a)

    def pop(self):
        min_node = self.q[0]
        min_i = 0
        for i in range(1, len(self.q)):
            if self.q[i].dist < min_node.dist:
                min_node = self.q[i]
                min_i = i

        return self.q.pop(min_i)

    def decrease_key(self, node, val):
        node.dist = val

    def count(self):
        return len(self.q)


def gen_graph(n, p):
    """
    Generate random graph as adjacency list
    :param n: number of nodes
    :param p:
    :return:
    """
    nodes = []
    for i in range(n):
        nodes.append(Node(i))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            elif np.random.uniform() < p:
                # with probability p add an edge from i to j
                # add the node with weight from uniform distribution
                nodes[i].children.append((np.random.uniform(), nodes[j]))

    return nodes


def dijkstra(adj_list, start, end, pq):

    for node in adj_list:
        node.dist = np.inf
        node.prev = None

    start.dist = 0
    pq.make_queue(adj_list)

    while pq.count() > 0:
        # Next closest node
        u = pq.pop()

        # If the next closest node is further than the known distance to the destination
        # node, then end
        if end.dist <= u.dist:
            return
        for length, child in u.children:
            if child.dist > u.dist + length:
                child.dist = u.dist + length
                child.prev = u
                pq.decrease_key(child, u.dist + length)


def test_priority_queue(test_pq):
    nodes = []
    for i in range(1, 11):
        n = Node(i)
        n.dist = 2 * i
        nodes.append(n)

    random.shuffle(nodes)

    print("Insertion order:")
    print([x.dist for x in nodes])
    # for node in nodes:
    #     test_pq.insert(node)

    test_pq.make_queue(nodes)

    test_pq.decrease_key(nodes[3], nodes[3].dist - 3)
    out_order = []
    for i in range(10):
        n = test_pq.pop()
        out_order.append(n.dist)

    print("Pushed and Popped:")
    print(out_order)


def print_graph(adj_list):
    print("Node\tEdges (weight, key)")
    for i in range(len(adj_list)):
        kids = " ".join(["({}, {})".format(w, n.key) for w, n in adj_list[i].children])
        print("{}->\t{}".format(i, kids))


def print_path(node):
    path = []
    while node is not None:
        path.append((node.key, node.dist))
        node = node.prev

    path.reverse()
    print(path)


def test_dijkstras():
    import pickle
    nodes = gen_graph(6, 0.5)
    # with open('bad_graph.pkl', 'rb') as g:
    #     nodes = pickle.load(g)
    print_graph(nodes)

    s = 0
    e = 5

    dijkstra(nodes, nodes[s], nodes[e], MinHeap())
    print("Path found min heap:")
    print_path(nodes[e])

    print("Path found array:")
    dijkstra(nodes, nodes[s], nodes[e], PQArray())
    print_path(nodes[e])

def test_performance(ps, node_counts, samples):
    import time
    import tqdm
    for p in ps:
        heap_runtime = np.zeros((node_counts.shape[0] * samples, 2))
        array_runtime = np.zeros((node_counts.shape[0] * samples, 2))
        run = 0
        for count in node_counts:
            print("Count: {}".format(count))
            for s in tqdm.tqdm(range(samples)):
                nodes = gen_graph(count, p)

                start = random.randint(0, count - 1)
                end = random.randint(0, count - 1)
                while end == start:
                    end = random.randint(0, count - 1)

                tik = time.time()
                dijkstra(nodes, nodes[start], nodes[end], MinHeap())
                tok = time.time() - tik
                heap_runtime[run] = np.array([count, tok])


                tik = time.time()
                dijkstra(nodes, nodes[start], nodes[end], PQArray())
                tok = time.time() - tik
                array_runtime[run] = np.array([count, tok])

                run += 1

        fig, ax = plt.subplots()

        # ax.scatter(heap_runtime[:, 0], heap_runtime[:, 1], c='tab:blue', label='Min Heap', alpha=0.6)
        # ax.scatter(array_runtime[:, 0], array_runtime[:, 1], c='tab:orange', label='Array', alpha=0.6, marker='+')
        #
        # ax.set_xlabel('|V|')
        # ax.set_ylabel('t')

        #ax.scatter(np.log(heap_runtime[:, 0]), np.log(heap_runtime[:, 1]), c='tab:blue', label='Min Heap', alpha=0.6)
        #ax.scatter(np.log(array_runtime[:, 0]), np.log(array_runtime[:, 1]), c='tab:orange', label='Array', alpha=0.6, marker='+')

        #heap_runtime[:, 1].reshape(node_counts.shape[0], samples)
        bp_heap_data = np.log(heap_runtime[:, 1]).reshape(node_counts.shape[0], samples)
        heap_data = [bp_heap_data[i, :] for i in range(node_counts.shape[0])]
        bp_minheap = ax.boxplot(heap_data, showfliers=False, patch_artist=True, labels=list(node_counts), widths=0.5, medianprops={'color': 'blue'})

        bp_array_data = np.log(array_runtime[:, 1]).reshape(node_counts.shape[0], samples)
        bp_array = [bp_array_data[i, :] for i in range(node_counts.shape[0])]
        bp_array = ax.boxplot(bp_array, showfliers=False, patch_artist=True, labels=list(node_counts), widths=0.7)

        for patch in bp_minheap['boxes']:
            patch.set_facecolor('tab:blue')
            patch.set_alpha(0.5)

        for patch in bp_array['boxes']:
            patch.set_facecolor('tab:orange')
            patch.set_alpha(0.5)

        ax.set_xlabel('|V|')
        ax.set_ylabel('log(t)')


        ax.set_title('P(E) = {:03.2f}'.format(p))
        plt.legend([bp_minheap['boxes'][0], bp_array['boxes'][0]], ["Min Heap", "Array"], loc='upper left')

        plt.savefig('p{}.png'.format(int(p * 100)))
        plt.show()



if __name__== "__main__":
    # PQ testing
    # print("Testing Min Heap")
    # test_priority_queue(MinHeap())
    # print("Testing Array")
    # test_priority_queue(PQArray())
    # for i in range(100):
    #     test_dijkstras()

    ps = np.linspace(0.1, 1.0, 10)
    node_counts = np.power(2, np.arange(4, 11))
    samples = 20
    test_performance(ps, node_counts, samples)