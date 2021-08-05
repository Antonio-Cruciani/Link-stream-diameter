import numpy as np
import os
from pathlib import Path
import temporal_graph as tg


def check_file_exists(file_path: str):
    """
    Given file path check if file exists

    :param file_path: File path
    :return: True if file exists, False otherwise
    """
    exist = False
    my_file = Path(file_path)
    if my_file.is_file():  # file exist
        exist = True
    return exist


def check_directory_exists(path: str):
    """
    Given directory path check if directory exists

    :param path: Directory path
    :return: True if directory exists, False otherwise
    """
    exist = False
    my_dir = Path(path)
    if my_dir.is_dir():  # directory exist
        exist = True
    return exist


def create_dir(path: str):
    """
    Create directory 'path' if not exist
    """
    if not check_directory_exists(path):
        try:
            os.mkdir(path)
        except OSError:
            raise OSError("Creation of the directory %s failed" % path)


def create_outdir(graph: tg.Graph, directory_name: str, txt_out_name: str):
    """
    Create output directory

    :param graph: A given Graph
    :param directory_name: Out directory name
    :param txt_out_name: Out txt file name
    :returns:
        - Input path of txt
        - Output path of out txt
    """
    txt_path = graph.get_file_path()
    directory, txt_name = txt_path.rsplit('/', 1)
    directory += '/'

    out_directory = directory + directory_name
    if out_directory[-1] != '/':
        out_directory += '/'

    create_dir(out_directory)

    output_path = out_directory + txt_name + '-' + txt_out_name
    return txt_path, output_path


def reverse_edges_sort(graph: tg.Graph, delimiter=" "):
    """
    Reverse the edges sorting of the txt Graph; if reversed file exists return only file path
    (Load the entire graph into memory for reverse it)

    :param graph: A given Graph
    :param delimiter: Delimiter in input txt Graph
    :return: Output path
    """
    txt_path, output_path = create_outdir(graph=graph, directory_name='graphs_reversed_sort',
                                          txt_out_name='reversed_edges')

    if not check_file_exists(output_path):
        print('Creating graph with reversed edges...')
        edges_stream = np.loadtxt(txt_path, dtype=int, delimiter=delimiter, ndmin=2)
        edges_stream_rev = edges_stream[::-1]
        np.savetxt(output_path, edges_stream_rev, fmt='%i', delimiter=' ')
        print('Reverse graph created!\n')
    return output_path


def transform_graph(graph: tg.Graph, delimiter=" "):
    """
    Transform each edge (u, v, t, lambda) in (v, u, -t - lambda, lambda)
    (Load the entire graph into memory for reverse it)
    This method is for apply backward visits to a given Graph, and forward visits for LDT distance (for LDT, after the
    transformation, the graph must be reversed with reverse_edge() method)

    :return: Output path
    """
    txt_path, output_path = create_outdir(graph=graph, directory_name='transformed_graphs',
                                          txt_out_name='transformed')

    if not check_file_exists(output_path):
        print('Creating transformed graph...')
        edges_stream = np.loadtxt(txt_path, dtype=int, delimiter=delimiter, ndmin=2)
        traversal_time = None
        with open(file=output_path, mode='x') as f:
            for li in edges_stream:
                u = int(li[0])
                v = int(li[1])
                t = int(li[2])
                if len(li) > 3:
                    traversal_time = int(li[3])
                    t = t + traversal_time
                else:
                    t += 1
                f.write(str(v) + " ")
                f.write(str(u) + " ")
                f.write(str(-t))
                if traversal_time:
                    f.write(" " + str(traversal_time))
                f.write("\n")
        print('Transformed graph created!\n')
        print('Sorting transformed graph...')
        print('Loading graph...')
        m = np.loadtxt(output_path, dtype=int, delimiter=' ', ndmin=2)
        print('Sorting graph...')
        m = m[m[:, 2].argsort()]
        print('Saving sorted graph...')
        np.savetxt(output_path, m, fmt='%i', delimiter=' ')
        print('Saved! \n')
    return output_path


def sort_graph(graph: tg.Graph):
    """
    Sort input Graph and save it in 'sorted_graphs' directory.

    :return: Path of sorted graph
    """
    txt_path = graph.get_file_path()
    directory, txt_name = txt_path.rsplit('/', 1)
    directory += '/'

    output_dir = directory + 'sorted_graphs/'
    create_dir(output_dir)

    output_path = output_dir + txt_name + '.sor'
    if not check_file_exists(output_path):
        print('Loading graph ' + txt_name + '...')
        m = np.loadtxt(txt_path, dtype=int, delimiter=' ', ndmin=2)
        print('Sorting graph ' + txt_name + '...')
        m = m[m[:, 2].argsort()]
        print('Saving sorted graph...')
        np.savetxt(output_path, m, fmt='%i', delimiter=' ')
        print('Saved! \n')
    else:
        print('File ' + output_path.rsplit('/', 1)[1] + ' already exist in ' + output_dir + '\n')
    return output_path


def only_files(path: str):
    """
    Iterator of files in 'path' (directories excluded)
    """
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


def files_for_dimensions(path):
    """
    :param path: Directory path
    :return: List of files in directory 'dir' ordered by file's size
    """
    file_list = []
    for file in only_files(path):
        file_list.append((file, os.stat(path + file).st_size))
        file_list.sort(key=lambda tup: tup[1])
    return [elem[0] for elem in file_list]
