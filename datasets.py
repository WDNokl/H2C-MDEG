import numpy as np
import pickle

def load(name, split):
    data = np.load(f'../../Data/{name}/{split}.npz')
    dist_mask = np.load(f'../../Data/{name}/{split}_dist_mask.npy')
    dataset = {
        'x': data['x'],
        'y': data['y'],
        'dist_mask': dist_mask,
        'node_mask': data['node_mask']
    }
    return dataset


def load_sbm(name):
    assert(name in ['PATTERN', 'CLUSTER'])
    train_set = load(name, split='train')
    val_set = load(name, split='val')
    test_set = load(name, split='test')
    return train_set, val_set, test_set

def load_superpixel(name):
    assert(name in ['MNIST', 'CIFAR10'])
    train_set = load(name, split='train')
    val_set = load(name, split='val')
    test_set = load(name, split='test')
    return train_set, val_set, test_set

def load_zinc():
    train_set = load(f'ZINC/subset', split='train')
    train_set['edge_attr'] = np.load(f'../../Data/ZINC/subset/train_edge_attr.npy')
    val_set = load(f'ZINC/subset', split='val')
    val_set['edge_attr'] = np.load(f'../../Data/ZINC/subset/val_edge_attr.npy')
    test_set = load(f'ZINC/subset', split='test')
    test_set['edge_attr'] = np.load(f'../../Data/ZINC/subset/test_edge_attr.npy')
    return train_set, val_set, test_set

def load_peptides(name):
    assert(name in ['peptides-struct', 'peptides-func'])
    train_xy = np.load(f'../../Data/{name}/train.npz')
    train_dist_mask = pickle.load(open(f'../../Data/{name}/train_dist_mask.pkl', 'rb'))
    val_xy = np.load(f'../../Data/{name}/val.npz')
    val_dist_mask = pickle.load(open(f'../../Data/{name}/val_dist_mask.pkl', 'rb'))
    test_xy = np.load(f'../../Data/{name}/test.npz')
    test_dist_mask = pickle.load(open(f'../../Data/{name}/test_dist_mask.pkl', 'rb'))
    return (train_xy, train_dist_mask), (val_xy, val_dist_mask), (test_xy, test_dist_mask)

