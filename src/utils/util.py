import numpy as np
import os
# from file_read_backwards import FileReadBackwards
from pathlib import Path


def reverse_ef(folder, file):
    """
    Revere Graph's txt
    Load the entire graph into memory for reverse it
    """
    if folder[-1] != '/':
        folder += '/'
    graph_path = folder + file
    output_dir = folder + 'reversed_graphs/'

    create_dir(output_dir)

    output_path = output_dir + file + '-reverse'
    if not check_file_exists(output_path):
        print('Creating reverse ordered graph...')
        edges_stream = np.loadtxt(graph_path, dtype=int, delimiter=' ', ndmin=2)
        # edges_stream = np.loadtxt(graph_path, dtype=str, delimiter=' ', ndmin=2)
        edges_stream_inv = edges_stream[::-1]
        np.savetxt(output_path, edges_stream_inv, fmt='%i', delimiter=' ')
        print('Reverse graph created!\n')
    return output_path


"""
def reverse(folder, file):
    # Revere Graph's txt in efficient way (if reversed file exists return only file path)
    # Does not load the entire graph into memory for reverse it (more efficient!)
    # :param folder: folder path
    # :param file: file name
    
    if folder[-1] != '/':
        folder += '/'
    graph_path = folder + file
    output_dir = folder + 'reversed_graphs/'

    create_dir(output_dir)

    output_path = output_dir + file + '-reverse'
    if not check_file_exists(output_path):
        print('Creating reverse ordered graph of ' + file + '...')
        with open(output_path, mode='x') as f:
            with FileReadBackwards(graph_path) as frb:
                # getting lines by lines starting from the last line up
                for line in frb:
                    f.write(line)
                    f.write('\n')
        print('Reverse graph created!\n')
    return output_path
"""


def opposite_graph(folder, file):
    """
    Transform graph with opposit sign times of temporal edges and with arch orientation reversed
    N.B. t = -(t + 1) (for apply EAT bw and obtain LDT fw)
    """

    if folder[-1] != '/':
        folder += '/'
    graph_path = folder + file
    output_dir = folder + 'opposite_graphs/'

    create_dir(output_dir)

    output_path = output_dir + file + '-opposite'

    if not check_file_exists(output_path):
        print('Creating graph with opposite times of ' + file + '...')
        with open(file=output_path, mode='x') as f:
            with open(file=graph_path) as f1:
                for line in f1:
                    if not line.strip():  # check if line is blank
                        break
                    li = line.split()
                    u = int(li[0])
                    v = int(li[1])
                    t = int(li[2])
                    t += 1
                    f.write(str(v) + " ")
                    f.write(str(u) + " ")
                    f.write(str(-t) + "\n")
        print('Opposite graph created!\n')
    return output_path


def opposite_graph_ascend(folder, file):
    """
    This method is for apply backward BFS Fastest Path to a given Graph in path: folder + file
    This method load all graph in memory
    Transform graph with opposit sign times of temporal edges and with arch orientation reversed.
    The function read the original graph in reversed order in such a way that if the original graph was sorted, the
    generated one will be sorted too
    """
    if folder[-1] != '/':
        folder += '/'
    graph_path = folder + file
    output_dir = folder + 'opposite_graphs_ascend/'

    create_dir(output_dir)

    output_path = output_dir + file + '-opposite_ascend'

    if not check_file_exists(output_path):
        print('Creating graph with opposite times of ' + file + '...')
        edges_stream = np.loadtxt(graph_path, dtype=int, delimiter=' ', ndmin=2)
        edges_stream = edges_stream[::-1]
        with open(file=output_path, mode='x') as f:
            for li in edges_stream:
                u = int(li[0])
                v = int(li[1])
                t = int(li[2])
                f.write(str(v) + " ")
                f.write(str(u) + " ")
                f.write(str(-t))
                # if len(li) > 3:
                #    traversal_time = int(li[3])
                #    f.write(" " + str(traversal_time))
                f.write("\n")
        print('Opposite graph created!\n')
    return output_path


def opposite_graph_ascend_traversal(folder, file):
    """
    This method is for apply backward BFS Shortest Path (can have trav times!=1) to a given Graph in path: folder + file
    This method load all graph in memory
    Transform graph with opposit sign times of temporal edges and with arch orientation reversed.
    The function read the original graph in reversed order in such a way that if the original graph was sorted, the
    generated one will be sorted too
    """
    if folder[-1] != '/':
        folder += '/'
    graph_path = folder + file
    output_dir = folder + 'opposite_graphs_ascend_traversal/'

    create_dir(output_dir)

    output_path = output_dir + file + '-opposite_ascend_traversal'

    if not check_file_exists(output_path):
        print('Creating graph with opposite times of ' + file + '...')
        edges_stream = np.loadtxt(graph_path, dtype=int, delimiter=' ', ndmin=2)
        edges_stream = edges_stream[::-1]
        with open(file=output_path, mode='x') as f:
            for li in edges_stream:
                u = int(li[0])
                v = int(li[1])
                t = int(li[2])
                if len(li) > 3:
                    traversal_time = int(li[3])
                else:
                    traversal_time = 1
                if traversal_time > 1:
                    t = t + traversal_time - 1
                f.write(str(v) + " ")
                f.write(str(u) + " ")
                f.write(str(-t) + " ")
                f.write(str(traversal_time))
                f.write("\n")
        print('Opposite graph created!\n')
        
        print('Loading graph...')
        m = np.loadtxt(output_path, dtype=int, delimiter=' ', ndmin=2)
        print('Sorting graph...')
        m = m[m[:, 2].argsort()]
        print('Saving sorted graph...')
        np.savetxt(output_path, m, fmt='%i', delimiter=' ')
        print('Saved! \n')
    
    return output_path


"""
def opposite_graph_ascend_ef(folder, file):
    # Transform graph with opposit sign times of temporal edges and with arch orientation reversed.
    # The function read the original graph in reversed order in such a way that if the original graph was sorted, the
    # generated one will be sorted too
   
    if folder[-1] != '/':
        folder += '/'
    graph_path = folder + file
    output_dir = folder + 'opposite_graphs_ascend/'

    create_dir(output_dir)

    output_path = output_dir + file + '-opposite_ascend'

    if not check_file_exists(output_path):
        print('Creating graph with opposite times of ' + file + '...')
        with open(file=output_path, mode='x') as f:
            with FileReadBackwards(graph_path) as f1:
                for line in f1:
                    li = line.split()
                    u = int(li[0])
                    v = int(li[1])
                    t = int(li[2])
                    f.write(str(v) + " ")
                    f.write(str(u) + " ")
                    f.write(str(-t) + "\n")
        print('Opposite graph created!\n')
    return output_path
"""


def sort_graph(folder, file):
    if folder[-1] != '/':
        folder += '/'
    graph_path = folder + file

    output_dir = folder + 'sorted_graphs/'
    create_dir(output_dir)

    output_path = output_dir + file + '.sor'
    if not check_file_exists(output_path):
        print('Loading graph ' + file + '...')
        m = np.loadtxt(graph_path, dtype=int, delimiter=' ', ndmin=2)
        print('Sorting graph ' + file + '...')
        m = m[m[:, 2].argsort()]
        print('Saving sorted graph...')
        np.savetxt(output_path, m, fmt='%i', delimiter=' ')
        print('Saved! \n')
    else:
        print('File ' + output_path.rsplit('/', 1)[1] + ' already exist in ' + output_dir + '\n')
    return output_path


def create_dir(path):
    """
    Create directory 'path' if not exist
    """
    if not check_dir_exists(path):
        try:
            os.mkdir(path)
        except OSError:
            raise OSError("Creation of the directory %s failed" % path)


def get_max_ind(dist, t):
    """
    return index of max value (except np.inf values) and its value
    :param dist: 1d numpy array of distances from one node
    :param t: minimum distance
    """
    index = -1
    count = 0
    reachables = 0
    for elem in dist:
        if elem != np.inf:
            reachables += 1
            if elem > t:
                t = elem
                index = count
        count += 1
    return index, t, reachables


def get_min_ind(dist, t):
    """
    return index of max value (except np.inf values) and its value
    :param dist: 1d numpy array of distances from one node
    :param t: minimum distance
    """
    index = -1
    count = 0
    for elem in dist:
        if elem != np.NINF and elem < t:
            t = elem
            index = count
        count += 1
    return index, t


def oly_files(path):
    """
    Iterator of files in 'path' (directories excluded)
    """
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


def check_file_exists(file_path):
    """
    given file path check if file exists
    :param file_path: path of file
    """
    exist = False
    my_file = Path(file_path)
    if my_file.is_file():  # file exist
        exist = True
    return exist


def check_dir_exists(dir_path):
    """
    given directory path check if directory exists
    :param dir_path: directory path
    """
    exist = False
    my_dir = Path(dir_path)
    if my_dir.is_dir():  # directory exist
        exist = True
    return exist


def files_for_dimensions(dire):
    """
    Return a list of files in directory 'dire' ordered by file's size
    :param dire: directory
    """
    g_list = []
    for file in oly_files(dire):
        g_list.append((file, os.stat(dire + file).st_size))
        g_list.sort(key=lambda tup: tup[1])
    return [i[0] for i in g_list]


def convert_all(convert_method, directory, order_by_size=False):
    """
    convert all graphs in directory and save them
    :param convert_method: method to convert the graph (reverse_ef, opposite_graph, opposite_graph_ascend, sort_graph)
    :param directory: directory where all graphs to convert are
    :param order_by_size: if True, sorts all graphs by their size
    :return output_path: directory path of converted graphs

    """
    output_path = ''
    g_list = []
    if order_by_size:
        g_list = files_for_dimensions(directory)
    else:
        for file in oly_files(directory):
            g_list.append(file)
    for g_name in g_list:
        output_path = convert_method(folder=directory, file=g_name)
    return output_path.rsplit('/', 1)[0]
