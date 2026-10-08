"""
    author: wangyuxiang
    date: 2021-2-28

    Store the previously computed entropy change from merging two communities.
"""


class TwoID(object):

    def __init__(self, id1, id2):
        self.id1 = id1  # int
        self.id2 = id2  # int

    def __str__(self):
        return "TwoID [ID1=" + str(self.id1) + ", ID2=" + str(self.id2) + "]"

    def __hash__(self):
        return self.id1 * self.id2 + self.id1 + self.id2

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if (other.__dict__.get("id1") == self.__dict__.get("id1") and other.__dict__.get("id2") == self.__dict__.get("id2")) or \
                    (other.__dict__.get("id2") == self.__dict__.get("id1") and other.__dict__.get("id1") == self.__dict__.get("id2")):
                return True
        return False

    def get_id1(self):
        return self.id1

    def get_id2(self):
        return self.id2
