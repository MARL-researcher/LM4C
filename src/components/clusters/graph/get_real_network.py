"""
    author: wangyuxiang
    date: 2021-2-26

    Data preprocessing:
        Read graph data from a file, construct a Graph object, and return it.
        The graph file uses the following data format:
            96
            1 2 0.43753893752019957
            1 68 0.3533857046475604
            1 66 0.320978468597678
            ...

            The first line gives the number of nodes, 96 in this example.
            Starting on the second line, each edge is stored as src_node_id dst_node_id weight.
            The input graph is directed, with each edge running from source to destination; for simplicity, it is treated as undirected here.
"""
from components.clusters.graph.graph import Graph
from components.clusters.graph.edge import Edge


class GetRealNetwork(object):

    def __init__(self, filename):
        self.filename = filename

    def get_graph(self):
        with open(self.filename, 'rb') as f:
            lines = f.readlines()
            vertices_number = int(lines[0])     # The first line contains the number of nodes.
            graph = Graph(vertices_number)

            for line in lines[1:]:
                line = line.strip().split()
                src_id = int(line[0])
                dst_id = int(line[1])
                weight = float(line[2])
                graph.set_edges_number(graph.get_edges_number() + 2)

                '''
                    To treat the directed input graph as undirected,
                    create two Edge objects for each input edge:
                        - src_id -> dst_id
                        - dst_id -> src_id
                    Add them to the neighbor lists of the corresponding nodes.
                    
                    The directed input may contain both directions of the same edge; add that edge only once.
                '''
                edge1 = Edge(src_id, dst_id, weight)
                edge2 = Edge(dst_id, src_id, weight)
                bool_src = edge1 in graph.get_vertice_connect_edge_list()[src_id]
                bool_dst = edge2 in graph.get_vertice_connect_edge_list()[dst_id]
                graph.get_vertice_connect_edge_list()[src_id].add(edge1)
                graph.get_vertice_connect_edge_list()[dst_id].add(edge2)

                if not bool_src:
                    graph.get_vertice_degree_list()[src_id] += weight
                if not bool_dst:
                    graph.get_vertice_degree_list()[dst_id] += weight
                if not bool_src and not bool_dst:
                    graph.set_degree_sum(graph.get_degree_sum() + 2 * weight)

        return graph


# Test program.
if __name__ == '__main__':
    graph = GetRealNetwork('../data/Lymph6Graph').get_graph()
    print("Vertices number: ", graph.get_vertices_number())
    print("Edges number: ", graph.get_edges_number())
    print("Degree summary: ", graph.get_degree_sum())
    print("Degree of each node:")
    for i in range(1, 6):
        print("    Node %d's degree: %f" % (i, graph.get_vertice_degree_list()[i]))
    print("Neighbors of each node:")
    for i in range(1, 6):
        print("    Node %d's neighbor:" % i)
        for j in graph.get_vertice_connect_edge_list()[i]:
            print("        - %s" % j)
    print("Community number: ", graph.get_community_number())
