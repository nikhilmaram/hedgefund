from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import mpl_toolkits
import pandas as pd
import unittest
import os
from parameterized import parameterized
import networkx as nx

import network
import config as cfg



class NetworkModuleTest(unittest.TestCase):
    # =========================================================================
    # ==================== Checks if matrix and adjacency list are created ====
    # =========================================================================
    def test_create_matrix(self):
        df = pd.DataFrame(
            columns=["sender_user_name", "receiver_user_name", "content", "time_stamp", "sender_in_network",
                     "receiver_in_network", "classify", "sentiment"])
        df = df.append(
            {"sender_user_name": "achacoso_ralph", "receiver_user_name": "ameziane_ghania", "content": "Hello",
             "time_stamp": "01-02-11T03:05:10", "sender_in_network": 1, "receiver_in_network": 0,
             "classify": 1, "sentiment": 1}, ignore_index=True)
        df = df.append(
            {"sender_user_name": "achacoso_ralph", "receiver_user_name": "wolfberg_adam", "content": "Hello World",
             "time_stamp": "01-02-11T03:05:10", "sender_in_network": 1, "receiver_in_network": 1,
             "classify": 1, "sentiment": 1}, ignore_index=True)
        df = df.append(
            {"sender_user_name": "nikhilmaram", "receiver_user_name": "ameziane_ghania", "content": "World",
             "time_stamp": "01-02-11T03:05:10", "sender_in_network": 0, "receiver_in_network": 0,
             "classify": 0, "sentiment": 0}, ignore_index=True)



        file_path = "./tmp.csv"
        df.to_csv(file_path)

        user_count =  cfg.TOTAL_EMPLOYEES + 2
        expected_message_matrix = np.zeros((user_count + 1, user_count + 1))
        expected_message_adj_list = [set() for _ in range(user_count + 1)]
        expected_message_matrix[2][298] = 1 ; expected_message_matrix[299][298] = 1 ; expected_message_matrix[2][289] = 1
        expected_message_adj_list[2].add(289) ; expected_message_adj_list[2].add(298) ; expected_message_adj_list[299].add(298)

        observed_message_matrix, observed_message_adj_list,_ = network.create_matrix(file_path,False)
        self.assertEqual(observed_message_adj_list,expected_message_adj_list)
        self.assertEqual(observed_message_matrix.all(),expected_message_matrix.all())

        ### Only using inside hedgefund employees
        user_count = cfg.TOTAL_EMPLOYEES
        observed_message_matrix, observed_message_adj_list, _ = network.create_matrix(file_path, True)
        expected_message_matrix = np.zeros((user_count + 1, user_count + 1))
        expected_message_adj_list = [set() for _ in range(user_count + 1)]
        expected_message_matrix[2][289] = 1
        expected_message_adj_list[2].add(289)
        self.assertEqual(observed_message_adj_list, expected_message_adj_list)
        self.assertEqual(observed_message_matrix.all(), expected_message_matrix.all())

    # =========================================================================
    # ==================== Checks if graph created is correct =================
    # =========================================================================

    def test_create_graph(self):
        expected_graph = nx.DiGraph()

        ### Only using inside hedgefund employees
        expected_graph.add_edge(64, 242, weight=1.0); expected_graph.add_edge(242, 64, weight=2.0)
        expected_graph.add_edge(280, 242, weight = 1.0)
        message_matrix,_,_ = network.create_matrix(cfg.IM_TEST_FILE,in_network=True)
        observed_graph = network.create_graph(message_matrix,un_directed=False)
        self.assertEqual(nx.is_isomorphic(observed_graph, expected_graph), True)

        ### All the users
        expected_graph.add_edge(181, 299, weight=1.0); expected_graph.add_edge(298, 301, weight=2.0)
        expected_graph.add_edge(280, 302, weight=3.0); expected_graph.add_edge(302,280, weight =1.0)
        expected_graph.add_edge(300,234,weight=1.0)

        message_matrix, _, _ = network.create_matrix(cfg.IM_TEST_FILE, in_network=False)
        observed_graph = network.create_graph(message_matrix, un_directed=False)
        self.assertEqual(nx.is_isomorphic(observed_graph, expected_graph), True)

        ## Check for undirected graph.

        expected_graph = expected_graph.to_undirected()
        message_matrix, _, _ = network.create_matrix(cfg.IM_TEST_FILE, in_network=False)
        observed_graph = network.create_graph(message_matrix, un_directed=True)
        self.assertEqual(nx.is_isomorphic(observed_graph, expected_graph), True)

    @parameterized.expand([
        [1,8,2,5],
        [2,5,1,5],
        [3,4,1,4],
        [4,0,0,0]
    ])
    def test_compute_kcore_values(self,kcore_num,kcore_num_of_nodes,kcore_num_components,kcore_largest_cc_num_nodes):
        input_G = nx.Graph()

        ## Create the input Graph
        input_G.add_edge(1,2) ; input_G.add_edge(1,3) ; input_G.add_edge(2,3) ; input_G.add_edge(2,4) ; input_G.add_edge(3,4)
        input_G.add_edge(3,5) ; input_G.add_edge(4,5) ; input_G.add_edge(1,4)
        input_G.add_edge(6,7) ; input_G.add_edge(6,8)

        kcore_num_obs, kcore_num_of_nodes_obs, kcore_num_components_obs, kcore_largest_cc_num_nodes_obs = network.compute_kcore_values(input_G,kcore_num)
        self.assertEqual(kcore_num_obs,kcore_num)
        self.assertEqual(kcore_num_of_nodes_obs,kcore_num_of_nodes)
        self.assertEqual(kcore_num_components_obs,kcore_num_components)
        self.assertEqual(kcore_largest_cc_num_nodes_obs,kcore_largest_cc_num_nodes)










