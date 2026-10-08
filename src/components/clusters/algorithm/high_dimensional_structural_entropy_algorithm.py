"""
    author: wangyuxiang
    date: 2021-2-27

    Structural entropy minimization algorithms, including:
        - One-dimensional structural entropy minimization.
        - Two-dimensional structural entropy minimization.
        - Three-dimensional structural entropy minimization.
"""
from components.clusters.algorithm.priority_tree import compute_structural_entropy_of_node, PriorityTree
from components.clusters.algorithm.tree_node import TreeNode
from components.clusters.algorithm.two_id import TwoID


class HighDimensionalStructureEntropyAlgorithm(object):

    def __init__(self, graph):
        self.vertice_degree_list = graph.get_vertice_degree_list()
        self.degree_sum = graph.get_degree_sum()
        self.vertice_number = graph.get_vertices_number()
        self.cut_set = dict()  # {TwoCommunity: Float}
        self.vertice_connect_edge_list = list()  # [{string: Float}, {string: Float}]

        for i in range(self.vertice_number + 1):
            self.vertice_connect_edge_list.append(dict())

        for i in range(1, self.vertice_number + 1):
            edge_of_node_i = graph.get_vertice_connect_edge_list()[i]  # Set of edges connected to node i.
            '''
            Each node i has a dictionary of neighboring nodes and the corresponding edge weights.
                i: {
                    "k_1": w1,
                    "k_2": w2,
                    ...
                    "k_n": wn
                } 
            k_1 is a neighbor of node i, and w1 is the weight of edge_i_to_k_1.
            '''
            d = dict()
            for e in edge_of_node_i:
                d[str(e.get_dst_id())] = e.get_weight()
            self.vertice_connect_edge_list[i] = d

        '''
        Initialize cut_set with pairs of connected singleton communities and their edge weights.
        '''
        for i in range(1, self.vertice_number):
            edge_dict_of_nodei = self.vertice_connect_edge_list[i]
            for j in range(i + 1, self.vertice_number + 1):
                if edge_dict_of_nodei.get(str(j)):
                    comi = set()
                    comi.add(str(i))
                    comj = set()
                    comj.add(str(j))
                    two_community = self.TwoCommunity(comi, comj)
                    gij = edge_dict_of_nodei[str(j)]  # Cut weight between the two communities; here, the weight of the edge connecting the two nodes.
                    self.cut_set.setdefault(two_community, gij)

    '''
        One-dimensional structural entropy minimization:
            - An encoding tree of maximum height 1 has one leaf level, representing the graph without a community partition.
            - One-dimensional structural entropy equals the Shannon entropy of the degree distribution.
    '''

    def one_dimension(self):
        SE = 0.0
        for i in range(1, self.vertice_number + 1):
            SE += compute_structural_entropy_of_node(self.vertice_degree_list[i],
                                                     self.degree_sum,
                                                     self.vertice_degree_list[i],
                                                     self.degree_sum)
        return SE

    '''
        Two-dimensional structural entropy minimization.
    '''

    def two_dimension(self):
        id = 0
        level = 0
        vi = self.degree_sum  # Volume of the current node.
        vifa = 0  # Volume of the parent node.
        gi = 0.0  # Cut weight.
        se = 0.0  # The root node's structural entropy is zero.
        children = list()  # List of the current node's children: [TreeNode, TreeNode, ...].
        entropy_of_childtree = list()  # Structural entropy of each child subtree.
        highest_level_of_childtree = list()  # Maximum height of each child subtree.
        iter_num = -1
        merge_detaH_of_children = dict()  # {TwoID: Float}
        combine_detaH_of_children = dict()  # {TwoID: Float}
        '''
            Create the root node of the encoding tree.
                - The root has no parent (None).
                - For two-dimensional structural entropy, leaves belong to the second level; the root's three leaf-storage parameters are None.
        '''
        root = TreeNode(id, level, None, vi, vifa, gi, se, children, entropy_of_childtree, highest_level_of_childtree,
                        None, None, None, iter_num, merge_detaH_of_children, combine_detaH_of_children)
        big_tree = PriorityTree(root, self.vertice_degree_list, self.degree_sum)

        '''
            Create a TreeNode for each graph vertex and add it as a leaf under the root before merge and combine operations.
            Initialize or compute all parameters for each TreeNode and pass them to its constructor.
        '''
        for i in range(self.vertice_number):
            vertice_id = i + 1
            node_id = i
            tree_level = root.get_level() + 1

            # Compute the current node's structural entropy.
            own_volumn = self.vertice_degree_list[vertice_id]
            father_volumn = root.get_own_volumn()
            cut = self.vertice_degree_list[vertice_id]
            structural_entropy_of_node = compute_structural_entropy_of_node(cut, self.degree_sum, own_volumn,
                                                                            father_volumn)

            children_of_leaf = list()
            entropy_of_leaf_children = list()
            highest_level_of_leaf_children = list()
            community_of_leaves1 = list()
            community_of_leaves2 = list()
            all_leaves = list()
            community_of_leaves1.append(str(vertice_id))
            all_leaves.append(str(vertice_id))

            iter_num = 0
            merge_detaH_of_children = dict()  # {TwoID: Float}
            combine_detaH_of_children = dict()  # {TwoID: Float}

            leaf = TreeNode(node_id, tree_level, root, own_volumn, father_volumn, cut, structural_entropy_of_node,
                            children_of_leaf, entropy_of_leaf_children, highest_level_of_leaf_children,
                            community_of_leaves1,
                            community_of_leaves2, all_leaves, iter_num, merge_detaH_of_children,
                            combine_detaH_of_children)
            children.append(leaf)
            entropy_of_childtree.append(structural_entropy_of_node)
            highest_level_of_childtree.append(tree_level)
        # Update the root node's information.
        root.set_children(children)
        root.set_entropy_of_childtree(entropy_of_childtree)
        root.set_highest_level_of_childtree(highest_level_of_childtree)

        iter_num = 1
        while True:
            # Record the largest structural entropy change and the corresponding partition nodes.
            max1 = None  # TreeNode
            max2 = None  # TreeNode
            g12 = 0.0
            maxdetaH = 0.0

            # The root's children currently contain a TreeNode for every graph vertex.
            children_of_root = root.get_children()
            merge_detaH_of_root_children = root.get_merge_detaH_of_children()
            for i in range(self.vertice_number - 1):
                for j in range(i + 1, self.vertice_number):
                    if children_of_root[i] and children_of_root[j]:  # ???
                        rooti = children_of_root[i]
                        rootj = children_of_root[j]
                        cutij = self.get_cut(rooti.get_all_leaves(), rootj.get_all_leaves())
                        if cutij != 0:
                            two_id = TwoID(rooti.get_id(), rootj.get_id())
                            if merge_detaH_of_root_children.get(two_id):
                                detaH = merge_detaH_of_root_children.get(two_id)
                            else:
                                detaH = self.merge_same_level(rooti, rootj, root, cutij)
                                merge_detaH_of_root_children.setdefault(two_id, detaH)
                            if detaH > maxdetaH:
                                max1 = rooti
                                max2 = rootj
                                maxdetaH = detaH
                                g12 = cutij
            if maxdetaH > 0:
                new_node = big_tree.merge(max1, max2, g12, iter_num)
            else:
                break
            iter_num += 1

            # Update cut weights between the merged node (new community) and other nodes (communities) at the same level.
            update_cut_of_newnode_and_othernode = new_node.get_father().get_children()
            for node in update_cut_of_newnode_and_othernode:
                if node and node.get_id() != new_node.get_id():
                    cut = self.get_cut(node.get_all_leaves(), new_node.get_community_of_leaves1()) + \
                          self.get_cut(node.get_all_leaves(), new_node.get_community_of_leaves2())
                    if cut != 0.0:
                        self.set_cut(node.get_all_leaves(), new_node.get_all_leaves(), cut)
        # Output the partition and normalize the tree.
        self.output_twod_result(big_tree.get_root())
        self.adjust_tree(big_tree)
        return big_tree

    '''
        Three-dimensional structural entropy minimization.
    '''

    def three_dimension(self):
        id = 0
        level = 0
        vi = self.degree_sum  # Volume of the current node.
        vifa = 0  # Volume of the parent node.
        gi = 0.0  # Cut weight.
        se = 0.0  # The root node's structural entropy is zero.
        children = list()  # List of the current node's children: [TreeNode, TreeNode, ...].
        entropy_of_childtree = list()  # Structural entropy of each child subtree.
        highest_level_of_childtree = list()  # Maximum height of each child subtree.
        iter_num = -1
        merge_detaH_of_children = dict()  # {TwoID: Float}
        combine_detaH_of_children = dict()  # {TwoID: Float}
        '''
            Create the root node of the encoding tree.
                - The root has no parent (None).
                - For two-dimensional structural entropy, leaves belong to the second level; the root's three leaf-storage parameters are None.
        '''
        root = TreeNode(id, level, None, vi, vifa, gi, se, children, entropy_of_childtree, highest_level_of_childtree,
                        None, None, None, iter_num, merge_detaH_of_children, combine_detaH_of_children)
        big_tree = PriorityTree(root, self.vertice_degree_list, self.degree_sum)

        '''
            Create a TreeNode for each graph vertex and add it as a leaf under the root before merge and combine operations.
            Initialize or compute all parameters for each TreeNode and pass them to its constructor.
        '''
        for i in range(self.vertice_number):
            vertice_id = i + 1
            node_id = i
            tree_level = root.get_level() + 1

            # Compute the current node's structural entropy.
            own_volumn = self.vertice_degree_list[vertice_id]
            father_volumn = root.get_own_volumn()
            cut = self.vertice_degree_list[vertice_id]
            structural_entropy_of_node = compute_structural_entropy_of_node(cut, self.degree_sum, own_volumn,
                                                                            father_volumn)

            children_of_leaf = list()
            entropy_of_leaf_children = list()
            highest_level_of_leaf_children = list()
            community_of_leaves1 = list()
            community_of_leaves2 = list()
            all_leaves = list()
            community_of_leaves1.append(str(vertice_id))
            all_leaves.append(str(vertice_id))

            iter_num = 0
            merge_detaH_of_children = dict()  # {TwoID: Float}
            combine_detaH_of_children = dict()  # {TwoID: Float}

            leaf = TreeNode(node_id, tree_level, root, own_volumn, father_volumn, cut, structural_entropy_of_node,
                            children_of_leaf, entropy_of_leaf_children, highest_level_of_leaf_children,
                            community_of_leaves1,
                            community_of_leaves2, all_leaves, iter_num, merge_detaH_of_children,
                            combine_detaH_of_children)
            children.append(leaf)
            entropy_of_childtree.append(structural_entropy_of_node)
            highest_level_of_childtree.append(tree_level)
        # Update the root node's information.
        root.set_children(children)
        root.set_entropy_of_childtree(entropy_of_childtree)
        root.set_highest_level_of_childtree(highest_level_of_childtree)

        iter_num = 1
        while True:
            while True:
                # Record the largest structural entropy change and the corresponding partition nodes.
                max1 = None  # TreeNode
                max2 = None  # TreeNode
                g12 = 0.0
                maxdetaH = 0.0

                # The root's children currently contain a TreeNode for every graph vertex.
                children_of_root = root.get_children()
                merge_detaH_of_root_children = root.get_merge_detaH_of_children()

                # Compute merge results for the first level of the encoding tree.
                for i in range(self.vertice_number - 1):
                    for j in range(i + 1, self.vertice_number):
                        if children_of_root[i] and children_of_root[j]:  # ???
                            rooti = children_of_root[i]
                            rootj = children_of_root[j]
                            cutij = self.get_cut(rooti.get_all_leaves(), rootj.get_all_leaves())
                            if cutij != 0:
                                two_id = TwoID(rooti.get_id(), rootj.get_id())
                                if merge_detaH_of_root_children.get(two_id):
                                    detaH = merge_detaH_of_root_children.get(two_id)
                                else:
                                    detaH = self.merge_same_level(rooti, rootj, root, cutij)
                                    merge_detaH_of_root_children.setdefault(two_id, detaH)
                                if detaH > maxdetaH:
                                    max1 = rooti
                                    max2 = rootj
                                    maxdetaH = detaH
                                    g12 = cutij

                # Compute merge results for the second level, whose parent nodes are children of the root.
                for level_one_father_k in children_of_root:
                    # Merge second-level children only when their first-level parent exists and is not a leaf.
                    if level_one_father_k is not None and level_one_father_k.get_iterate_number() != 0:
                        level_two_of_k = level_one_father_k.get_children()
                        merge_detaH_of_k_children = level_one_father_k.get_merge_detaH_of_children()
                        for i in range(len(level_two_of_k) - 1):
                            for j in range(i + 1, len(level_two_of_k)):
                                if level_two_of_k[i] and level_two_of_k[j]:
                                    level_two_nodei = level_two_of_k[i]
                                    level_two_nodej = level_two_of_k[j]
                                    cutij = self.get_cut(level_two_nodei.get_all_leaves(),
                                                         level_two_nodej.get_all_leaves())
                                    if cutij != 0:
                                        two_id = TwoID(level_two_nodei.get_id(), level_two_nodej.get_id())
                                        if merge_detaH_of_k_children.get(two_id):
                                            detaH = merge_detaH_of_k_children.get(two_id)
                                        else:
                                            detaH = self.merge_same_level(level_two_nodei, level_two_nodej,
                                                                          level_one_father_k, cutij)
                                            merge_detaH_of_k_children.setdefault(two_id, detaH)
                                        if detaH > maxdetaH:
                                            max1 = level_two_nodei
                                            max2 = level_two_nodej
                                            maxdetaH = detaH
                                            g12 = cutij

                if maxdetaH > 0:
                    new_node = big_tree.merge(max1, max2, g12, iter_num)
                else:
                    break
                iter_num += 1

                # Update cut weights between the merged node (new community) and other nodes (communities) at the same level.
                update_cut_of_newnode_and_othernode = new_node.get_father().get_children()
                for node in update_cut_of_newnode_and_othernode:
                    if node and node.get_id() != new_node.get_id():
                        cut = self.get_cut(node.get_all_leaves(), new_node.get_community_of_leaves1()) + \
                              self.get_cut(node.get_all_leaves(), new_node.get_community_of_leaves2())
                        if cut != 0.0:
                            self.set_cut(node.get_all_leaves(), new_node.get_all_leaves(), cut)

            # When merging yields no further changes, begin combine operations.
            iter_num += 1
            max1 = None
            max2 = None
            g12 = 0.0
            maxdetaH = 0.0
            children_of_root = root.get_children()
            combine_detaH_of_root_children = root.get_combine_detaH_of_children()
            highest_level_of_one = root.get_highest_level_of_childtree()

            # Compute combine results for the first level of the encoding tree.
            for i in range(len(children_of_root) - 1):
                for j in range(i + 1, len(children_of_root)):
                    # The subtree selected for combining must have a maximum height below 3; if its height is 3,
                    # combining would increase its height to 4, exceeding three-dimensional structural entropy minimization.
                    if children_of_root[i] and children_of_root[j] and \
                            highest_level_of_one[i] < 3 and highest_level_of_one[j] < 3:
                        nodei = children_of_root[i]
                        nodej = children_of_root[j]
                        cutij = self.get_cut(nodei.get_all_leaves(), nodej.get_all_leaves())
                        if cutij != 0:
                            two_id = TwoID(nodei.get_id(), nodej.get_id())
                            if combine_detaH_of_root_children.get(two_id):
                                detaH = combine_detaH_of_root_children.get(two_id)
                            else:
                                detaH = self.combine_same_level(nodei, nodej, root, cutij)
                                combine_detaH_of_root_children.setdefault(two_id, detaH)
                            if detaH > maxdetaH:
                                max1 = nodei
                                max2 = nodej
                                maxdetaH = detaH
                                g12 = cutij

            # Compute combine results for the second level of the encoding tree.
            for level_one_father_k in children_of_root:
                # Operate on second-level children only when their first-level parent exists and is not a leaf.
                if level_one_father_k is not None and level_one_father_k.get_iterate_number() != 0:
                    level_two_of_k = level_one_father_k.get_children()
                    highest_level_of_two = level_one_father_k.get_highest_level_of_childtree()
                    combine_detaH_of_k_children = level_one_father_k.get_combine_detaH_of_children()
                    for i in range(len(level_two_of_k) - 1):
                        for j in range(i + 1, len(level_two_of_k)):
                            if level_two_of_k[i] and level_two_of_k[j] and \
                                    highest_level_of_two[i] < 3 and highest_level_of_two[j] < 3:
                                level_two_nodei = level_two_of_k[i]
                                level_two_nodej = level_two_of_k[j]
                                cutij = self.get_cut(level_two_nodei.get_all_leaves(), level_two_nodej.get_all_leaves())
                                if cutij != 0:
                                    two_id = TwoID(level_two_nodei.get_id(), level_two_nodej.get_id())
                                    if combine_detaH_of_k_children.get(two_id):
                                        detaH = combine_detaH_of_k_children.get(two_id)
                                    else:
                                        detaH = self.combine_same_level(level_two_nodei, level_two_nodej,
                                                                        level_one_father_k, cutij)
                                        combine_detaH_of_k_children.setdefault(two_id, detaH)
                                    if detaH > maxdetaH:
                                        max1 = level_two_nodei
                                        max2 = level_two_nodej
                                        maxdetaH = detaH
                                        g12 = cutij

            if maxdetaH > 0:
                new_node = big_tree.combine(max1, max2, g12, iter_num)
            else:
                break

            # Update cut weights between the combined node (new community) and other nodes (communities) at the same level.
            update_cut_of_newnode_and_othernode = new_node.get_father().get_children()
            for node in update_cut_of_newnode_and_othernode:
                if node and node.get_id() != new_node.get_id():
                    cut = self.get_cut(node.get_all_leaves(), new_node.get_community_of_leaves1()) + \
                          self.get_cut(node.get_all_leaves(), new_node.get_community_of_leaves2())
                    if cut != 0.0:
                        self.set_cut(node.get_all_leaves(), new_node.get_all_leaves(), cut)

        # Output the partition and normalize the tree.
        self.output_threed_result(big_tree.get_root())
        self.adjust_tree(big_tree)
        return big_tree

    '''
        Retrieve the cut weight between two communities from self.cut_set.
        comi: [string, string, ...]
    '''

    def get_cut(self, comi, comj):
        comi = set(comi)
        comj = set(comj)
        two_community = self.TwoCommunity(comi, comj)
        if self.cut_set.get(two_community):
            return self.cut_set[two_community]
        return 0.0

    '''
        Update or create the cut weight between two communities.
        comi: [string, string, ...]
    '''

    def set_cut(self, comi, comj, gij):
        comi = set(comi)
        comj = set(comj)
        two_community = self.TwoCommunity(comi, comj)
        if not self.cut_set.get(two_community):
            self.cut_set.setdefault(two_community, gij)

    '''
        Compute the structural entropy change from merging two nodes at the same level.
    '''

    def merge_same_level(self, nodei, nodej, father, cutij):
        entropy_before = father.get_entropy_of_childtree()[nodei.get_id()] + \
                         father.get_entropy_of_childtree()[nodej.get_id()]

        entropy_after = 0.0

        # Compute the structural entropy of the node formed by merging the two current nodes.
        v_new = nodei.get_own_volumn() + nodej.get_own_volumn()
        cut_new = nodei.get_cut() + nodej.get_cut() - 2 * cutij
        v_fa = nodei.get_father_volumn()
        entropy_after += compute_structural_entropy_of_node(cut_new, self.degree_sum, v_new, v_fa)

        # Recompute the entropy of both nodes' children because their community volumes have changed.
        # leaf_id identifies a leaf of node i and matches the corresponding graph vertex ID.
        for leaf_id in nodei.get_all_leaves():
            entropy_after += compute_structural_entropy_of_node((self.vertice_degree_list[int(leaf_id)]),
                                                                self.degree_sum,
                                                                self.vertice_degree_list[int(leaf_id)],
                                                                v_new)
        for leaf_id in nodej.get_all_leaves():
            entropy_after += compute_structural_entropy_of_node((self.vertice_degree_list[int(leaf_id)]),
                                                                self.degree_sum,
                                                                self.vertice_degree_list[int(leaf_id)],
                                                                v_new)
        return entropy_before - entropy_after

    '''
        Compute the structural entropy change from combining two nodes at the same level.
    '''

    def combine_same_level(self, nodei, nodej, father, cutij):
        entropy_before = father.get_entropy_of_childtree()[nodei.get_id()] + \
                         father.get_entropy_of_childtree()[nodej.get_id()]

        entropy_after = 0.0
        v_new = nodei.get_own_volumn() + nodej.get_own_volumn()
        cut_new = nodei.get_cut() + nodej.get_cut() - 2 * cutij
        v_fa = nodei.get_father_volumn()
        for e in nodei.get_entropy_of_childtree():
            entropy_after += e
        for e in nodej.get_entropy_of_childtree():
            entropy_after += e
        entropy_after += compute_structural_entropy_of_node(nodei.get_cut(), self.degree_sum, nodei.get_own_volumn(),
                                                            v_new)
        entropy_after += compute_structural_entropy_of_node(nodej.get_cut(), self.degree_sum, nodej.get_own_volumn(),
                                                            v_new)
        entropy_after += compute_structural_entropy_of_node(cut_new, self.degree_sum, v_new,
                                                            v_fa)
        return entropy_before - entropy_after

    '''
        Print the two-dimensional encoding tree.
    '''
    def output_twod_result(self, node):
        if node.get_iterate_number() == 0:
            return node.get_all_leaves()[0] + ' '
        else:
            s = ""
            for n in node.get_children():
                if n:
                    s += self.output_twod_result(n)
                    if n.get_level() == 1:
                        print(s)
                        s = ""
            return s

    '''
        Print the three-dimensional encoding tree.
    '''
    def output_threed_result(self, node):
        if node.get_iterate_number() == 0:
            return node.get_all_leaves()[0] + ' '
        else:
            s = ""
            for n in node.get_children():
                if n:
                    s += self.output_threed_result(n)
                    if n.get_level() == 2:
                        print(s)
                        s = ""
                    if n.get_level() == 1:
                        print()
                        s = ""
            return s

    # Remove empty tree nodes and normalize node IDs.
    def adjust_tree(self, tree):
        root = tree.get_root()
        self.delete_none(root)
        self.correct_id(root)

    def delete_none(self, root):
        if root.get_iterate_number() != 0:
            children = root.get_children()
            # Remove the specified element (None here) from a list.
            j = 0
            for i in range(len(children)):
                if not children[j]:
                    children.pop(j)
                else:
                    j += 1
            for i in children:
                self.delete_none(i)

    def correct_id(self, root):
        if root.get_iterate_number() != 0:
            children = root.get_children()
            for i in range(len(children)):
                children[i].set_id(i)
            for i in children:
                self.correct_id(i)

    '''
        Store information about two communities for cut-weight calculations.
    '''

    class TwoCommunity(object):

        def __init__(self, comi, comj):
            self.comi = comi  # set(string, string, ... , string)
            self.comj = comj

        def __str__(self):
            return "TwoCommunity [comi=" + str(self.comi) + ", comj=" + str(self.comj) + "]"

        # Override hashing to support object equality comparisons.
        def __hash__(self):
            a1 = 0
            a2 = 0
            min1 = 10000000
            min2 = 10000000
            for i in self.comi:
                a1 += int(i)
                if min1 < int(i):
                    min1 = int(i)
            for i in self.comj:
                a2 += int(i)
                if min2 < int(i):
                    min2 = int(i)
            return a1 * min1 + a2 * min2

        def __eq__(self, other):
            if self is other:
                return True
            if other is None:
                return False
            if not isinstance(other, self.__class__):
                return False

            other_comi = other.__dict__.get("comi")
            other_comj = other.__dict__.get("comj")
            my_comi = self.__dict__.get("comi")
            my_comj = self.__dict__.get("comj")

            if my_comi is None:
                if other_comi is not None:
                    return False
            if my_comj is None:
                if other_comj is not None:
                    return False
            a1 = len(my_comi)
            b1 = len(my_comj)
            a2 = len(other_comi)
            b2 = len(other_comj)

            if my_comi == other_comi and my_comj == other_comj and a1 == a2 and b1 == b2 or \
                    my_comj == other_comi and my_comi == other_comj and a1 == b2 and b1 == a2:
                return True
            return False

    '''
        Test program.
    '''

    # Print cut_set, which stores cut weights between the corresponding communities.
    def print_cut_set(self):
        print("----------Test cut set-----------")
        print("Size of cut set: %d" % len(self.cut_set))
        for k, v in self.cut_set.items():
            print("Community = %s, weight = %f" % (k, v))

    # Print the entire encoding tree.
    def print_tree(self, root):
        print(root)
        print('  - ', root.get_merge_detaH_of_children())
        print('  - ', root.get_combine_detaH_of_children())
        for i in root.get_children():
            self.print_tree(i)
