"""
    author: wangyuxiang
    date: 2021-2-26

    Graph model class, containing:
        - Number of nodes.
        - Number of edges.
        - Sum of all node degrees.
        - Node degrees stored in a list indexed by node ID.
        - Connected edges for each node, stored in a list indexed by node ID;
          each list element is a set of Edge objects.
        - CommunityOfEachVertice
        - VerticesOfEachCommunity
        - Number of communities in the partition.
"""


class Graph(object):

    def __init__(self, vertices_number, edges_number=0, degree_sum=0.0, vertice_degree_list=None,
                 vertice_connect_edge_list=None, community_number=0):
        # Initialize these parameters with defaults if they were not provided.
        if vertice_degree_list is None:
            vertice_degree_list = []
            for i in range(vertices_number + 1):
                vertice_degree_list.append(0)
        if vertice_connect_edge_list is None:
            vertice_connect_edge_list = []
            for i in range(vertices_number + 1):
                vertice_connect_edge_list.append(set())
        self.vertices_number = vertices_number
        self.edges_number = edges_number
        self.degree_sum = degree_sum
        self.vertice_degree_list = vertice_degree_list
        self.vertice_connect_edge_list = vertice_connect_edge_list  # [set(edge1,..), set(edge1,..), ... , set(edge1,..)]
        self.community_number = community_number

    def get_vertices_number(self):
        return self.vertices_number

    def set_vertices_number(self, vertices_number):
        self.vertices_number = vertices_number

    def get_edges_number(self):
        return self.edges_number

    def set_edges_number(self, edges_number):
        self.edges_number = edges_number

    def get_degree_sum(self):
        return self.degree_sum

    def set_degree_sum(self, degree_sum):
        self.degree_sum = degree_sum

    def get_vertice_degree_list(self):
        return self.vertice_degree_list

    def set_vertice_degree_list(self, vertice_degree_list):
        self.vertice_degree_list = vertice_degree_list

    def get_vertice_connect_edge_list(self):
        return self.vertice_connect_edge_list

    def set_vertice_connect_edge_list(self, vertice_connect_edge_list):
        self.vertice_connect_edge_list = vertice_connect_edge_list

    def get_community_number(self):
        return self.community_number

    def set_community_number(self, community_number):
        self.community_number = community_number
