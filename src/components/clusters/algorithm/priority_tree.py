"""
    author: wangyuxiang
    date: 2021-2-27

    Encoding tree class, containing:
        - Root node.
        - Node degrees stored in a list; each degree is the sum of graph vertex degrees in the node's partition.
        - Total degree.
"""
import math
from components.clusters.algorithm.tree_node import TreeNode
from components.clusters.algorithm.two_id import TwoID

'''
    Compute the structural entropy of a node in the tree.
        - gi: cut weight.
        - degree_sum: total degree, corresponding to 2*m in the formula for an undirected graph.
        - vi: current node volume.
        - vifa: parent node volume.
        
    Structural entropy formula:
        H^T(G) = - sigma( (gi/2m) * log2(Vi/Vifa) )
'''


def compute_structural_entropy_of_node(gi, degree_sum, vi, vifa):
    w1 = gi / degree_sum
    w2 = vi / vifa
    w3 = w1 * (math.log2(w2))
    return -w3


class PriorityTree(object):

    def __init__(self, root, vertice_degree_list, degree_sum):
        self.root = root
        self.vertice_degree_list = vertice_degree_list
        self.degree_sum = degree_sum

    '''
        Merge operator.
            - child_one: first node (first community to merge) - TreeNode.
            - child_two: second node (second community to merge) - TreeNode.
            - cutij: cut weight between the two communities.
            - iter_num: current iteration count.
    '''

    def merge(self, child_one, child_two, cutij, iter_num):
        id1 = child_one.get_id()
        id2 = child_two.get_id()

        '''
            Use two new lists, new_com1 and new_com2, to store nodes in the merged community.
            Store all merged nodes in all_leaves.
        '''
        all_leaves = list()
        new_com1 = list()
        new_com2 = list()
        old_child_one_com1 = child_one.get_community_of_leaves1()
        old_child_one_com2 = child_one.get_community_of_leaves2()
        old_child_two_com1 = child_two.get_community_of_leaves1()
        old_child_two_com2 = child_two.get_community_of_leaves2()
        for i in old_child_one_com1:
            new_com1.append(i)
            all_leaves.append(i)
        for i in old_child_one_com2:
            new_com1.append(i)
            all_leaves.append(i)
        for i in old_child_two_com1:
            new_com2.append(i)
            all_leaves.append(i)
        for i in old_child_two_com2:
            new_com2.append(i)
            all_leaves.append(i)

        # Update the parent's information for child_two and clear child_two's information.
        child_one.get_father().get_children()[id2] = None
        child_one.get_father().get_entropy_of_childtree()[id2] = 0.0
        child_one.get_father().get_highest_level_of_childtree()[id2] = 0
        child_two.set_father(None)
        child_two.set_all_leaves(None)
        child_two.set_community_of_leaves1(None)
        child_two.set_community_of_leaves2(None)

        # Update child_one to serve as the parent node after merging.
        child_one.set_own_volumn(child_one.get_own_volumn() + child_two.get_own_volumn())
        child_one.set_cut(child_one.get_cut() + child_two.get_cut() - 2 * cutij)
        se = compute_structural_entropy_of_node(child_one.get_cut(), self.degree_sum, child_one.get_own_volumn(),
                                                child_one.get_father_volumn())
        child_one.set_structural_entropy_of_node(se)
        child_one.set_community_of_leaves1(new_com1)
        child_one.set_community_of_leaves2(new_com2)
        child_one.set_all_leaves(all_leaves)
        child_one.set_iterate_number(iter_num)

        '''
            Update child_one's children, including their structural entropies, to reflect the merged community.
            Rebuild all nodes instead of reusing existing ones; this needs optimization.
        '''
        children = list()
        entropy_of_childtree = list()
        highest_level_of_childtree = list()
        merge_detaH_of_children = dict()  # {TwoID: float}
        combine_detaH_of_children = dict()
        for i in range(len(all_leaves)):
            vertice_id = int(all_leaves[i])
            node_id = i
            tree_level = child_one.get_level() + 1

            # Compute the current node's structural entropy.
            own_volumn = self.vertice_degree_list[vertice_id]
            father_volumn = child_one.get_own_volumn()
            cut = self.vertice_degree_list[vertice_id]
            structural_entropy_of_node = compute_structural_entropy_of_node(cut, self.degree_sum, own_volumn,
                                                                            father_volumn)
            children_of_leaf = list()
            entropy_of_leaf_children = list()
            highest_level_of_leaf_children = list()
            community_of_leaves1 = list()
            community_of_leaves2 = list()
            all_leaves_of_node = list()
            community_of_leaves1.append(str(vertice_id))
            all_leaves_of_node.append(str(vertice_id))

            iter_num = 0
            merge_detaH_of_children = dict()  # {TwoID: Float}
            combine_detaH_of_children = dict()  # {TwoID: Float}

            leaf = TreeNode(node_id, tree_level, child_one, own_volumn, father_volumn, cut, structural_entropy_of_node,
                            children_of_leaf, entropy_of_leaf_children, highest_level_of_leaf_children,
                            community_of_leaves1,
                            community_of_leaves2, all_leaves_of_node, iter_num, merge_detaH_of_children,
                            combine_detaH_of_children)
            children.append(leaf)
            entropy_of_childtree.append(structural_entropy_of_node)
            highest_level_of_childtree.append(tree_level)
        # Update the root node's information.
        child_one.set_children(children)
        child_one.set_entropy_of_childtree(entropy_of_childtree)
        child_one.set_highest_level_of_childtree(highest_level_of_childtree)
        child_one.set_merge_detaH_of_children(merge_detaH_of_children)
        child_one.set_combine_detaH_of_children(combine_detaH_of_children)
        children = None
        entropy_of_childtree = None
        highest_level_of_childtree = None

        '''
            Update ancestor nodes' child subtree entropy and maximum-level arrays.
        '''
        curr = child_one
        father = curr.get_father()
        while True:
            curr_id = curr.get_id()
            curr_tree_entropy = 0.0
            curr_tree_highest_level = 0
            for i in range(len(curr.get_entropy_of_childtree())):
                curr_tree_entropy += curr.get_entropy_of_childtree()[i]
                if curr.get_highest_level_of_childtree()[i] > curr_tree_highest_level:
                    curr_tree_highest_level = curr.get_highest_level_of_childtree()[i]
            curr_tree_entropy += curr.get_structural_entropy_of_node()  # Also include the current root node's structural entropy.
            father.get_entropy_of_childtree()[curr_id] = curr_tree_entropy
            father.get_highest_level_of_childtree()[curr_id] = curr_tree_highest_level

            # Continue updating ancestor nodes.
            curr = father
            father = father.get_father()
            if not father:
                break

        # Remove all detaHofChildren entries involving id1 or id2 from child_one's parent.
        merge_detaH_of_childone_father = child_one.get_father().get_merge_detaH_of_children()
        merge_detaH_of_childone_father.pop(TwoID(id1, id2))  # Remove the TwoID entry for the child_one and child_two communities.
        # Remove all TwoID entries involving child_one or child_two.
        for i in range(len(child_one.get_father().get_children())):
            if merge_detaH_of_childone_father.get(TwoID(i, id1)):
                merge_detaH_of_childone_father.pop(TwoID(i, id1))
            if merge_detaH_of_childone_father.get(TwoID(i, id2)):
                merge_detaH_of_childone_father.pop(TwoID(i, id2))

        # Continue updating ancestors and remove the corresponding delta-H entries.
        curr = child_one.get_father()
        father = curr.get_father()
        while father:
            curr_id = curr.get_id()
            merge_detaH_of_childone_father = father.get_merge_detaH_of_children()
            for i in range(len(father.get_children())):
                if merge_detaH_of_childone_father.get(TwoID(i, curr_id)):
                    merge_detaH_of_childone_father.pop(TwoID(i, curr_id))
            curr = father
            father = father.get_father()

        return child_one

    '''
        Combine operator.
    '''
    def combine(self, child_one, child_two, cutij, iter_num):
        id1 = child_one.get_id()
        id2 = child_two.get_id()

        '''
            To join two nodes at the same level, create a new parent node and compute its attributes below.
        '''
        father = child_one.get_father()
        level = child_one.get_level()
        id = child_one.get_id()
        vi = child_one.get_own_volumn() + child_two.get_own_volumn()
        v_fa = child_one.get_father_volumn()
        gi = child_one.get_cut() + child_two.get_cut() - 2 * cutij
        se = compute_structural_entropy_of_node(gi, self.degree_sum, vi, v_fa)
        children = list([child_one, child_two])

        # new community and all_leaves
        all_leaves = list()
        new_com1 = list()
        new_com2 = list()
        old_child_one_com1 = child_one.get_community_of_leaves1()
        old_child_one_com2 = child_one.get_community_of_leaves2()
        old_child_two_com1 = child_two.get_community_of_leaves1()
        old_child_two_com2 = child_two.get_community_of_leaves2()
        for i in old_child_one_com1:
            new_com1.append(i)
            all_leaves.append(i)
        for i in old_child_one_com2:
            new_com1.append(i)
            all_leaves.append(i)
        for i in old_child_two_com1:
            new_com2.append(i)
            all_leaves.append(i)
        for i in old_child_two_com2:
            new_com2.append(i)
            all_leaves.append(i)

        # Sum the entropies of child_one and its subtree, do the same for child_two, and store them in the new parent's entropy_of_childtree.
        Hone = compute_structural_entropy_of_node(child_one.get_cut(), self.degree_sum,
                                                                     child_one.get_own_volumn(), vi)
        Htwo = compute_structural_entropy_of_node(child_two.get_cut(), self.degree_sum,
                                                                     child_two.get_own_volumn(), vi)
        entropy_child_tree_one = Hone
        entropy_child_tree_two = Htwo
        for e in child_one.entropy_of_childtree():
            entropy_child_tree_one += e
        for e in child_two.entropy_of_childtree():
            entropy_child_tree_two += e
        entropy_of_childree = list([entropy_child_tree_one, entropy_child_tree_two])

        # Compute the levels of the current node's child subtrees.
        level_of_children = list()
        highest_level_of_child_one = 0
        for i in child_one.get_highest_level_of_childtree():
            if highest_level_of_child_one < i:
                highest_level_of_child_one = i
        if highest_level_of_child_one != 0:
            level_of_children.append(highest_level_of_child_one + 1)  # Is the missing increment of 1 intentional here?
        else:
            level_of_children.append(child_one.get_level() + 1)
        highest_level_of_child_two = 0
        for i in child_two.get_highest_level_of_childtree():
            if highest_level_of_child_two < i:
                highest_level_of_child_two = i
        if highest_level_of_child_two != 0:
            level_of_children.append(highest_level_of_child_two + 1)
        else:
            level_of_children.append(child_two.get_level() + 1)

        merge_detaH_of_two_children = dict()
        combine_detaH_of_two_children = dict()

        tnode = TreeNode(id, level, father, vi, v_fa, gi, se,
                         children, entropy_of_childree, level_of_children,
                         new_com1,
                         new_com2, all_leaves, iter_num, merge_detaH_of_two_children,
                         combine_detaH_of_two_children)

        father.get_children[child_one.get_id()] = tnode
        father.get_children[child_two.get_id()] = None
        father.get_entropy_of_childtree[child_two.get_id()] = 0.0
        father.get_highest_level_of_childtree[child_two.get_id()] = 0

        # Update ancestor nodes.
        curr = tnode
        while True:
            curr_id = curr.get_id()
            curr_tree_entropy = 0.0
            curr_tree_highest_level = 0
            for i in range(len(curr.get_entropy_of_childtree())):
                curr_tree_entropy += curr.get_entropy_of_childtree()[i]
                if curr.get_highest_level_of_childtree()[i] > curr_tree_highest_level:
                    curr_tree_highest_level = curr.get_highest_level_of_childtree()[i]
            curr_tree_entropy += curr.get_structural_entropy_of_node()  # Also include the current root node's structural entropy.
            father.get_entropy_of_childtree()[curr_id] = curr_tree_entropy
            father.get_highest_level_of_childtree()[curr_id] = curr_tree_highest_level

            # Continue updating ancestor nodes.
            curr = father
            father = father.get_father()
            if not father:
                break

        # ???
        tnode.set_level(tnode.get_level() - 1)
        for i in range(len(tnode.get_highest_level_of_childtree())):
            tnode.get_highest_level_of_childtree()[i] = tnode.get_highest_level_of_childtree()[i] - 1
        # Update child_one and child_two.
        child_one.set_father(tnode)
        child_two.set_father(tnode)
        child_one.set_id(0)
        child_two.set_id(1)
        child_one.set_father_volumn(vi)
        child_two.set_father_volumn(vi)
        child_one.set_structural_entropy_of_node(Hone)
        child_two.set_structural_entropy_of_node(Htwo)
        self.update_level(tnode)

        # Remove all detaHofChildren entries involving id1 or id2 from tnode's parent.
        combine_detaH_of_tnode_father = tnode.get_father().get_combine_detaH_of_children()
        combine_detaH_of_tnode_father.pop(TwoID(id1, id2))  # Remove the TwoID entry for the child_one and child_two communities.
        # Remove all TwoID entries involving child_one or child_two.
        for i in range(len(child_one.get_father().get_children())):
            if combine_detaH_of_tnode_father.get(TwoID(i, id1)):
                combine_detaH_of_tnode_father.pop(TwoID(i, id1))
            if combine_detaH_of_tnode_father.get(TwoID(i, id2)):
                combine_detaH_of_tnode_father.pop(TwoID(i, id2))

        # Continue updating ancestors and remove the corresponding delta-H entries.
        curr = tnode.get_father()
        father = curr.get_father()
        while father:
            curr_id = curr.get_id()
            combine_detaH_of_childone_father = father.get_combine_detaH_of_children()
            for i in range(len(father.get_children())):
                if combine_detaH_of_childone_father.get(TwoID(i, curr_id)):
                    combine_detaH_of_childone_father.pop(TwoID(i, curr_id))
            curr = father
            father = father.get_father()

        return tnode

    '''
        Update the levels of all children of the newly created parent tnode.
    '''
    def update_level(self, node):
        node.set_level(node.get_level() + 1)
        if node.get_iterate_number() != 0:
            for i in range(len(node.get_highest_level_of_childtree())):
                level = node.get_highest_level_of_childtree()[i]
                if level != 0:
                    node.get_highest_level_of_childtree()[i] = level + 1
            for n in node.get_children():
                if n is not None:
                    self.update_level(n)

    def get_root(self):
        return self.root

    def set_root(self, root):
        self.root = root
