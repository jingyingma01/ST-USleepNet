import torch
from tqdm import tqdm
import networkx as nx
import numpy as np
from functools import partial
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import cosine_similarity


class G_data(object):
    def __init__(self, num_class, feat_dim, train_g_list, test_g_list, args):
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.seed = args.seed
        self.fold_idx = args.fold
        self.train_gs = train_g_list
        self.test_gs = test_g_list

class FileLoader(object):
    def __init__(self, args):
        self.args = args
        self.delta_t = args.delta_t
        self.delta_p = args.delta_p
        self.num_class = args.num_class
        self.feat_dim = args.feat_dim
        self.num_patch = args.num_patch
        self.num_node = args.num_node

    def line_genor(self, lines):
        for line in lines:
            yield line

    def gen_graph(self, psg, label):
        g = nx.Graph()
        node_features = []
        for j in range(len(psg)):
            g.add_node(j, features = psg[j])
            node_features.append(psg[j])
        similarity_matrix = cosine_similarity(node_features)
        for i in range(similarity_matrix.shape[0]):
            for j in range(i + 1, similarity_matrix.shape[1]):
                if similarity_matrix[i, j] > 0.5:
                    g.add_edge(i, j, weight=similarity_matrix[i, j])
        g.label = label
        return g

    def process_g(self, g):
        node_features = []
        num_node = self.num_node
        for j in range(self.num_patch * num_node):
            node_features.append(g.nodes[j]['features'])
        node_features = np.array(node_features)
        g.feas = torch.tensor(node_features)
        A = torch.FloatTensor(nx.to_numpy_array(g))
        g.A = A + torch.eye(g.number_of_nodes())
        time_matrix = np.zeros((self.num_patch * num_node, self.num_patch * num_node))
        for i in range(self.num_patch):
            for j in range(self.num_patch):
                time_matrix[i * num_node:(i + 1) * num_node,
                j * num_node:(j + 1) * num_node] = self.delta_t** abs(i - j)
        position_matrix = np.zeros((self.num_patch * num_node, self.num_patch * num_node))
        for i in range(0, self.num_patch * num_node):
            for j in range(0, self.num_patch * num_node):
                if i % num_node == j % num_node:
                    position_matrix[i, j] = 1
                else:
                    position_matrix[i, j] = self.delta_p
        g.A = g.A * time_matrix * position_matrix
        return g

    def load_data(self):
        args = self.args
        print('loading data ...')

        data = np.load('data/%s/%s.npz' % (args.data, args.data), allow_pickle = True)
        fold = args.fold
        test_datas = []
        test_labels = []
        train_datas = []
        train_labels = []

        for key in data.files:
            fold_data = data[key].item()
            if int(key) % 10 == int(fold):
                test_datas.extend(fold_data['datas'])
                test_labels.extend(fold_data['labels'])
            else:
                train_datas.extend(fold_data['datas'])
                train_labels.extend(fold_data['labels'])

        f_n = partial(self.process_g)

        train_g_list = []
        for i in tqdm(range(len(train_datas)), desc="Create train graph", unit='Graph'):
            g = self.gen_graph(train_datas[i], train_labels[i])
            train_g_list.append(g)

        test_g_list = []
        for i in tqdm(range(len(test_datas)), desc="Create test graph", unit='Graph'):
            g = self.gen_graph(test_datas[i], test_labels[i])
            test_g_list.append(g)

        new_train_g_list = []
        for g in tqdm(train_g_list, desc="Process train graph", unit='Graph'):
            new_train_g_list.append(f_n(g))

        new_test_g_list = []
        for g in tqdm(test_g_list, desc="Process test graph", unit='Graph'):
            new_test_g_list.append(f_n(g))

        return G_data(self.num_class, self.feat_dim, new_train_g_list, new_test_g_list, self.args)