# coding; utf-8
import tensorflow as tf
import numpy as np
import getopt
import sys
import os
import json
import random


def bfs_shortest_path(graph, start, goal):
    # keep track of explored nodes
    explored = []
    # keep track of all the paths to be checked
    queue = [[start]]

    # return path if start is goal
    if start == goal:
        return "That was easy! Start = goal"

    # keeps looping until all possible paths have been checked
    while queue:
        # pop the first path from the queue
        path = queue.pop(0)
        # get the last node from the path
        node = path[-1]
        if node not in explored:
            neighbours = graph[str(node)]
            # go through all neighbour nodes, construct a new path and
            # push it into the queue
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)
                # return path if neighbour is goal
                if neighbour == goal:
                    return new_path

            # mark node as explored
            explored.append(node)

    # in case there's no path between the 2 nodes
    return "So sorry, but a connecting path doesn't exist :("


def search_edge(from_id, to_id, edges):
    for _edge in edges:
        if from_id == _edge[0] and to_id == _edge[1]:
            return _edge
    return None


def generate_data(batch_size, edges, graph, max_length, min_path_len, max_path_len):
    '''
    - batch_size : int
    - seq_length : int
    - edges : np.array
    - graph : dict
    '''
    def convert_one_hot(_arr):
        out_list = []
        for idx, i in enumerate(_arr):
            if idx < 2:
                str_i = "{0:03d}".format(i)
                for char_i in str_i:
                    one_hot_char_i = np.eye(10)[[int(char_i)]].tolist()[0]
                    out_list.extend(one_hot_char_i)
                out_list.append(0)
            else:
                one_hot_char_i = np.eye(10)[[i]].tolist()[0]
                out_list.extend(one_hot_char_i)
        return out_list
    input_vecs = []
    out_vecs = []
    input_modes = []
    for _b in range(batch_size):
        input_vec = []
        out_vec = []
        input_mode = []
        np.random.shuffle(edges)
        for _edge in edges.tolist():
            input_vec.append(convert_one_hot(_edge))
            out_vec.append([0]*72)
            input_mode.append([1]*72)
        for i in range(1):
            shortest_path = [1,1,1,1,1,1,1,1,1,1,1,1,1]
            while len(shortest_path) > max_path_len+1 or len(shortest_path) < min_path_len:
                start_id, goal_id = random.sample(graph.keys(), 2)
                start_id = int(start_id)
                goal_id = int(goal_id)
                shortest_path = bfs_shortest_path(graph, start_id, goal_id)
            input_vec.append(convert_one_hot([start_id, goal_id, 0]))
            out_vec.append([0]*72)
            input_mode.append([1]*72)
            for j in range(5):
                input_vec.append([0]*72)
                out_vec.append([0]*72)
                input_mode.append([1]*72)
            for j in range(len(shortest_path)-1):
                input_vec.append([1]*72)
                out_vec.append(convert_one_hot(search_edge(shortest_path[j], shortest_path[j+1], edges)))
                # if j == 0:
                #     input_mode.append([1]*72)
                # else:
                input_mode.append([0]*72)
        # while len(input_vec) < max_length:
        #     input_vec.append([0]*52)
        #     out_vec.append([0]*52)
        #     input_mode.append([1]*52)
        input_vecs.append(input_vec)
        out_vecs.append(out_vec)
        input_modes.append(input_mode)
        
    return (input_vecs, out_vecs, input_modes)


def generate_patial_graph(edges, graph, start, node_size):
    out_edges = []
    out_graph = {}
    # keep track of explored nodes
    explored = []
    # keep track of all the paths to be checked
    queue = [[start]]
    # keeps looping until all possible paths have been checked
    while queue:
        # pop the first path from the queue
        path = queue.pop(0)
        # get the last node from the path
        for node in path:
            if node not in explored:
                neighbours = graph[str(node)]
                # go through all neighbour nodes, construct a new path and
                # push it into the queue
                new_path = list(path)
                for neighbour in neighbours:
                    if neighbour in new_path:
                        pass
                    else:
                        new_path.append(neighbour)
                    queue.append(new_path)
                    # return path if neighbour is goal
                    if len(new_path) >= node_size:
                        ans_path = new_path
                        for ip in ans_path:
                            _node = graph[str(ip)]
                            _out_node = []
                            for idx_n in _node:
                                if idx_n in ans_path:
                                    _out_node.append(idx_n)
                            out_graph[str(ip)] = _out_node
                            for jp in ans_path:
                                if ip == jp:
                                    continue
                                else:
                                    for _edge in edges:
                                        if ip in _edge[:2] and jp in _edge[:2]:
                                            if _edge in out_edges:
                                                pass
                                            else:
                                                out_edges.append(_edge)
                        return (out_graph, out_edges)
                # mark node as explored
                explored.append(node)

    # in case there's no path between the 2 nodes
    print("So sorry, but a connecting path doesn't exist :(")
    return (None, None)


if __name__ == '__main__':
    with open("./json/metro_training_data.json", "r") as f:
        data_dict = json.load(f)
    edges = data_dict["edge"]
    graph = data_dict["graph"]
    print(generate_data(1, np.array(edges), graph)[0])