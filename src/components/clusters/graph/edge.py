"""
    author: wangyuxiang
    date: 2021-2-26

    Graph edge class, containing:
        - Source node ID.
        - Destination node ID.
        - Edge weight.
"""


class Edge(object):

    def __init__(self, src_id, dst_id, weight):
        self.src_id = src_id
        self.dst_id = dst_id
        self.weight = weight

    # Override to return edge information as a string.
    def __str__(self):
        return "Edge [SrcId=]" + str(self.src_id) + ", DstId=" + str(self.dst_id) \
               + ", Weight=" + str(self.weight) + "]"

    def __hash__(self):
        return hash((self.src_id, self.dst_id, self.weight))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if other.__dict__.get("src_id") == self.__dict__.get("src_id") and \
                    other.__dict__.get("dst_id") == self.__dict__.get("dst_id") and \
                    other.__dict__.get("weight") == self.__dict__.get("weight"):
                return True
        return False

    def get_src_id(self):
        return self.src_id

    def set_src_id(self, src_id):
        self.src_id = src_id

    def get_dst_id(self):
        return self.dst_id

    def set_dst_id(self, dst_id):
        self.dst_id = dst_id

    def get_weight(self):
        return self.weight

    def set_weight(self, weight):
        self.weight = weight
