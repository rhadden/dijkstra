#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:51:49 2020

@author: rileyhadden
"""
import numpy as np
import random

class Node():
    def __init__(self):
        self.path = None
        self.dist = np.inf
        self.index = -1
    

class MinHeap():
    def __init__(self):
        self._heap = [None]
        self._len = 0
        
    def insert(self, a):
        self._len += 1
        self._heap.append(a)
        # set index pointer for decrease key operation
        a.index = self._len
        self.bubble_up(self._len)
        
    def bubble_up(self, index):
        # print("BU Index: {}".format(index))
        parent_idx = int(index // 2)
        
        if self._heap[index].dist > self._heap[parent_idx].dist:
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
                
        if self._heap[index] > self._heap[min_idx]:
            self.swap(index, min_idx)
            self.trickle_down(min_idx)
                
    def swap(self, idx1, idx2):
        temp = self._heap[idx1]
        self._heap[idx1] = self._heap[idx2]
        self._heap[idx2] = self._heap[idx1]
        # Update pointers within heap
        self._heap[idx1].index = idx2
        self._heap[idx2].index = idx1
        
    def change_val(self, index, val):
        index.dist = val
        self.trickle_down(index)
        
if __name__=="__main__":
    nodes = []
    for i in range(1, 11):
        n = Node()
        n.dist = 2 * i
        nodes.append(n)
        
    
    random.shuffle(nodes)
    
    
    test_heap = MinHeap()
    print("Insertion order:")
    print([x.dist for x in nodes])
    for node in nodes:
        test_heap.insert(node)
        
    
    out_order = []
    for i in range(10):
        n = test_heap.pop()
        out_order.append(n.dist)
        
    print("Pushed and Popped:")
    print(out_order)
        
            