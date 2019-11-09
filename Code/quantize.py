import numpy as np
from sklearn.cluster import KMeans
from pruned_layers import *
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def quantize_whole_model(net, bits=8):
    """
    Quantize the whole model.
    :param net: (object) network model.
    :return: centroids of each weight layer, used in the quantization codebook.
    """
    cluster_centers = []
    assert isinstance(net, nn.Module)
    layer_ind = 0
    for n, m in net.named_modules():
        if isinstance(m, PrunedConv):
            pass
            """
            Apply quantization for the PrunedConv layer.
            --------------Your Code---------------------
            """
            weights = m.conv.weight.data.cpu().numpy()
            original_shape = weights.shape
            non_zero_weights = weights[np.nonzero(weights)]
            weights = weights.reshape(-1,1)
            non_zero_weights = non_zero_weights.reshape(-1,1)
            kmean = KMeans(n_clusters = 2**bits, 
                           init='k-means++', 
                           n_init=25, 
                           max_iter=50)
            kmean.fit(non_zero_weights)
            cluster_centers.append(kmean.cluster_centers_)
            #print(cluster_centers[layer_ind])
            #print(weights.shape)
            quant_weights = np.zeros(weights.shape, dtype=np.float32)
            for i in range(len(weights)):
                weight = weights[i]
                if weight != 0:
                    label = kmean.predict(weight.reshape(1, -1))
                    quant_weights[i] = cluster_centers[layer_ind][label]
            #if layer_ind == 0:
            #    print(quant_weights)
            m.conv.weight.data = torch.from_numpy(quant_weights.reshape(original_shape)).float().to(device)
            
            layer_ind += 1
            print("Complete %d layers quantization..." %layer_ind)
            
        elif isinstance(m, PruneLinear):
            """
            Apply quantization for the PrunedLinear layer.
            --------------Your Code---------------------
            """
            weights = m.linear.weight.data.cpu().numpy()
            original_shape = weights.shape
            non_zero_weights = weights[np.nonzero(weights)]
            weights = weights.reshape(-1,1)
            non_zero_weights = non_zero_weights.reshape(-1,1)
            kmean = KMeans(n_clusters = 2**bits, 
                           init='k-means++', 
                           n_init=25, 
                           max_iter=50)
            kmean.fit(non_zero_weights)
            cluster_centers.append(kmean.cluster_centers_)
            #print(cluster_centers[layer_ind])
            #print(weights.shape)
            quant_weights = np.zeros(weights.shape, dtype=np.float32)
            for i in range(len(weights)):
                weight = weights[i]
                if weight != 0:
                    label = kmean.predict(weight.reshape(1, -1))
                    quant_weights[i] = cluster_centers[layer_ind][label]
            #if layer_ind == 0:
            #    print(quant_weights)
            m.linear.weight.data = torch.from_numpy(quant_weights.reshape(original_shape)).float().to(device)
            
            layer_ind += 1
            print("Complete %d layers quantization..." %layer_ind)
    
    centers = np.array(cluster_centers)
    centers = np.reshape(centers, (centers.shape[0], centers.shape[1]))
    return centers 

