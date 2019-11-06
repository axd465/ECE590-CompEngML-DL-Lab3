import numpy as np
from sklearn.cluster import KMeans
from pruned_layers import *
import torch.nn as nn
from collections import Counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class HuffmanNode():
    def __init__(self, key = '<!$>_ANTHONY_<$!>', freq = 0, right = None, left = None, leaf = False):
        if leaf:
            self.key = key
            self.freq = freq
            self.right = None
            self.left = None
            self.leaf = True
            self.encode = ''
        else:
            self.key = key
            self.freq = right.freq + left.freq
            self.right = right
            self.left = left
            self.leaf = False
            right.add_encode('1')
            left.add_encode('0')
        return
    def add_encode(self, addition):
        if self.leaf == False:
            if self.left == None:
                self.right.add_encode(addition)
            elif self.right == None:
                self.left.add_encode(addition)
            else:
                self.right.add_encode(addition)
                self.left.add_encode(addition)
            if self.right == None and self.left == None:
                print('Error: Recursively iterating on a leaf')
        else:
            self.encode = addition + self.encode
        return

def convert_freq_dict_to_encodings(freq):
    original_freq = freq.copy() # Just in Case I want to check something later
    
    leaf_list = []
    for centroid, frequency in freq.items():
        leaf_list.append(HuffmanNode(key = centroid,
                                     freq = frequency,
                                     leaf = True))
    tree = []
    tree.extend(leaf_list)
    
    MaxIter = 500
    iter = 0
    not_root = True
    
    # Forming Huffman Tree and Setting Encoding
    while not_root and iter < MaxIter:
        least_freq_item = tree.pop(-1)
        second_least_freq_item = tree.pop(-1)
        tree.append(HuffmanNode(key = 'Branch ' + str(iter),
                                right = second_least_freq_item,
                                left = least_freq_item))
        iter+=1
        not_root = len(tree) > 1
        if not_root:
            if tree[-1].freq > tree[-2].freq:
                tree = sorted(tree, key=lambda node: node.freq, reverse = True)
    encodings = {}
    for leaf in leaf_list:
        encodings[leaf.key] = leaf.encode
    return encodings


def _huffman_coding_per_layer(weight, centers):
    """
    Huffman coding for each layer
    :param weight: weight parameter of the current layer.
    :param centers: KMeans centroids in the quantization codebook of the current weight layer.
    :return: 
            'encodings': Encoding map mapping each weight parameter to its Huffman coding.
            'frequency': Frequency map mapping each weight parameter to the total number of its appearance.
            'encodings' should be in this format:
            {"0.24315": '0', "-0.2145": "100", "1.1234e-5": "101", ...
            }
            'frequency' should be in this format:
            {"0.25235": 100, "-0.2145": 42, "1.1234e-5": 36, ...
            }
            'encodings' and 'frequency' does not need to be ordered in any way.
    """
    """
    Generate Huffman Coding and Frequency Map according to incoming weights and centers (KMeans centroids).
    --------------Your Code---------------------
    """
    non_zero_weights = list(map(str, weight[np.nonzero(weight)]))# creates string array of non-zero weight values
    ordered = Counter(non_zero_weights)
    # creates dictionary of centroids in decending order of frequency
    frequency = {}
    for item in ordered.most_common(len(ordered)):
        key = item[0]
        value = item[1]
        frequency[key] = value
    encodings = convert_freq_dict_to_encodings(frequency) # converts freq dict to centroid encodings
    return encodings, frequency

def compute_average_bits(encodings, frequency):
    """
    Compute the average storage bits of the current layer after Huffman Coding.
    :param 'encodings': Encoding map mapping each weight parameter to its Huffman coding.
    :param 'frequency': Frequency map mapping each weight parameter to the total number of its appearance.
            'encodings' should be in this format:
            {"0.24315": '0', "-0.2145": "100", "1.1234e-5": "101", ...
            }
            'frequency' should be in this format:
            {"0.25235": 100, "-0.2145": 42, "1.1234e-5": 36, ...
            }
            'encodings' and 'frequency' does not need to be ordered in any way.
    :return (float) a floating value represents the average bits.
    """
    total = 0
    total_bits = 0
    for key in frequency.keys():
        total += frequency[key]
        total_bits += frequency[key] * len(encodings[key])
    return total_bits / total

def huffman_coding(net, centers):
    """
    Apply huffman coding on a 'quantized' model to save further computation cost.
    :param net: a 'nn.Module' network object.
    :param centers: KMeans centroids in the quantization codebook for Huffman coding.
    :return: frequency map and encoding map of the whole 'net' object.
    """
    assert isinstance(net, nn.Module)
    layer_ind = 0
    freq_map = []
    encodings_map = []
    for n, m in net.named_modules():
        if isinstance(m, PrunedConv):
            weight = m.conv.weight.data.cpu().numpy()
            center = centers[layer_ind]
            orginal_avg_bits = round(np.log2(len(center)))
            print("Original storage for each parameter: %.4f bits" %orginal_avg_bits)
            encodings, frequency = _huffman_coding_per_layer(weight, center)
            freq_map.append(frequency)
            encodings_map.append(encodings)
            huffman_avg_bits = compute_average_bits(encodings, frequency)
            print("Average storage for each parameter after Huffman Coding: %.4f bits" %huffman_avg_bits)
            layer_ind += 1
            print("Complete %d layers for Huffman Coding..." %layer_ind)
        elif isinstance(m, PruneLinear):
            weight = m.linear.weight.data.cpu().numpy()
            center = centers[layer_ind]
            orginal_avg_bits = round(np.log2(len(center)))
            print("Original storage for each parameter: %.4f bits" %orginal_avg_bits)
            encodings, frequency = _huffman_coding_per_layer(weight, center)
            freq_map.append(frequency)
            encodings_map.append(encodings)
            huffman_avg_bits = compute_average_bits(encodings, frequency)
            print("Average storage for each parameter after Huffman Coding: %.4f bits" %huffman_avg_bits)
            layer_ind += 1
            print("Complete %d layers for Huffman Coding..." %layer_ind)

    return freq_map, encodings_map