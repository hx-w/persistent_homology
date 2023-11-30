import itertools
import functools
import numpy as np
from tqdm import tqdm
import networkx as nx
from scipy.spatial.distance import squareform, pdist


def build_graph(data, epsilon=1., metric='euclidean'):
    D = squareform(pdist(data, metric=metric))
    D[D >= epsilon] = 0.
    G = nx.Graph(D)
    edges = list(map(set, G.edges()))
    weights = [G.get_edge_data(u, v)['weight'] for u, v in G.edges()]
    return G.nodes(), edges, weights

def rips_filtration(graph, k):
    nodes, edges, weights = graph
    VRcomplex = [{n} for n in nodes]
    filter_values = [0 for j in VRcomplex]  # vertices have filter value of 0
    # add 1-simplices (edges) and associated filter values
    for i in range(len(edges)):
        VRcomplex.append(edges[i])
        filter_values.append(weights[i])
    if k > 1:
        for i in range(k):
            # skip 0-simplices and 1-simplices
            for simplex in tqdm([x for x in VRcomplex if len(x) == i + 2]):
                # for each u in simplex
                nbrs = set.intersection(
                    *[lower_nbrs(nodes, edges, z) for z in simplex])
                for nbr in nbrs:
                    newSimplex = set.union(simplex, {nbr})
                    VRcomplex.append(newSimplex)
                    filter_values.append(get_filter_value(newSimplex, VRcomplex, filter_values))

    # sort simplices according to filter values
    return sort_complex(VRcomplex, filter_values)

def lower_nbrs(nodeSet, edgeSet, node): #lowest neighbors based on arbitrary ordering of simplices
    return {x for x in nodeSet if {x,node} in edgeSet and node > x}

def get_filter_value(simplex, edges, weights): #filter value is the maximum weight of an edge in the simplex
    oneSimplices = list(itertools.combinations(simplex, 2)) #get set of 1-simplices in the simplex
    max_weight = 0
    for oneSimplex in oneSimplices:
        filter_value = weights[edges.index(set(oneSimplex))]
        if filter_value > max_weight:
            max_weight = filter_value
    return max_weight


def compare(item1, item2):
    #comparison function that will provide the basis for our total order on the simpices
    #each item represents a simplex, bundled as a list [simplex, filter value] e.g. [{0,1}, 4]
    if len(item1[0]) == len(item2[0]):
        if item1[1] == item2[1]: #if both items have same filter value
            if sum(item1[0]) > sum(item2[0]):
                return 1
            else:
                return -1
        else:
            if item1[1] > item2[1]:
                return 1
            else:
                return -1
    else:
        if len(item1[0]) > len(item2[0]):
            return 1
        else:
            return -1

def sort_complex(filter_complex, filter_values): #need simplices in filtration have a total order
    pairedList = zip(filter_complex, filter_values)
    sortedComplex = sorted(pairedList, key=functools.cmp_to_key(compare))
    sortedComplex = [list(t) for t in zip(*sortedComplex)]
    
    return sortedComplex

#return the n-simplices and weights in a complex
def n_simplicies(n, filter_complex):
    nchain = []
    nfilters = []
    for i in range(len(filter_complex[0])):
        simplex = filter_complex[0][i]
        if len(simplex) == (n+1):
            nchain.append(simplex)
            nfilters.append(filter_complex[1][i])
    if (nchain == []):
        nchain = [0]
    return nchain, nfilters

#check if simplex is a face of another simplex
def check_face(face, simplex):
    if simplex == 0:
        return 1
    elif set(face) < set(simplex) and len(face) == len(simplex)-1: #if face is a (n-1) subset of simplex
        return 1
    else:
        return 0

#build boundary matrix for dimension n ---> (n-1) = p
def filter_boundary_matrix(filter_complex):
    bmatrix = np.zeros((len(filter_complex[0]),len(filter_complex[0])), dtype='>i8')
    
    i = 0
    for colSimplex in tqdm(filter_complex[0]):
        j = 0
        for rowSimplex in filter_complex[0]:
            bmatrix[j,i] = check_face(rowSimplex, colSimplex)
            j += 1
        i += 1
    return bmatrix

#returns row index of lowest "1" in a column i in the boundary matrix
def low(i, matrix):
    col = matrix[:, i]
    j = col.shape[0] - 1
    while j > -1:
        if col[j] == 1:
            return j
        j -= 1
    return col.shape[0] - 1

#checks if the boundary matrix is fully reduced
def is_reduced(matrix):
    for j in range(matrix.shape[1]):
        for i in range(j):
            low_j = low(j, matrix)
            low_i = low(i, matrix)
            if low_j == low_i and low_j != -1:
                return i, j
    return [0,0]

#the main function to iteratively reduce the boundary matrix
def reduce_boundary_matrix(matrix):
    #this refers to column index in the boundary matrix
    reduced_matrix = matrix.copy()
    matrix_shape = reduced_matrix.shape
    memory = np.identity(matrix_shape[1], dtype='>i8') #this matrix will store the column additions we make
    r = is_reduced(reduced_matrix)
    while (r != [0,0]):
        i = r[0]
        j = r[1]
        col_j = reduced_matrix[:,j]
        col_i = reduced_matrix[:,i]
        #print("Mod: add col %s to %s \n" % (i+1,j+1)) #Uncomment to see what mods are made
        reduced_matrix[:,j] = np.bitwise_xor(col_i,col_j) #add column i to j
        memory[i,j] = 1
        r = is_reduced(reduced_matrix)
    return reduced_matrix, memory

def read_intervals(reduced_matrix, filter_values): #reduced_matrix includes the reduced boundary matrix AND the memory matrix
    intervals = []
    m = reduced_matrix.shape[1]
    for j in range(m):
        low_j = low(j, reduced_matrix)
        if low_j == (m - 1):
            interval_start = [j, -1]
            intervals.append(interval_start)

        else:
            feature = intervals.index([low_j, -1])
            intervals[feature][1] = j
            epsilon_start = filter_values[intervals[feature][0]]
            epsilon_end = filter_values[j]
            if epsilon_start == epsilon_end:
                intervals.remove(intervals[feature])

    return intervals

def read_persistence(intervals, filter_complex):
    #this converts intervals into epsilon format and figures out which homology group each interval belongs to
    persistence = []
    for interval in intervals:
        start = interval[0]
        end = interval[1]
        homology_group = (len(filter_complex[0][start]) - 1) #filter_complex is a list of lists [complex, filter values]
        epsilon_start = filter_complex[1][start]
        epsilon_end = filter_complex[1][end]
        persistence.append([homology_group, [epsilon_start, epsilon_end]])

    return persistence
