"""GroupTree: walks of the group graph."""
from vescan_thalweg.group_tree import GroupTree


def tree():
    #  0 -> 1 -> 2, 1 -> 3   (main tree, 60 mm)
    #  9 -> 1                 (stub pointing to the main tree's hub)
    #  5                      (isolated fragment)
    nodes = {g: {"length": l, "generation": 0} for g, l in [(0, 30.0), (1, 10.0), (2, 10.0), (3, 10.0),
                                                              (9, 0.5), (5, 2.0)]}
    edges = [{"parent": 0, "child": 1}, {"parent": 1, "child": 2}, {"parent": 1, "child": 3},
             {"parent": 9, "child": 1}]
    return GroupTree(nodes, edges, [0, 9, 5])


def test_post_order_children_first():
    order = tree().post_order([0])
    assert order.index(2) < order.index(1) < order.index(0)
    assert order.index(3) < order.index(1)
    assert set(order) == {0, 1, 2, 3}


def test_subtree_with_exclusions():
    t = tree()
    assert t.subtree(0) == {0, 1, 2, 3}
    assert t.subtree(0, excluded={1}) == {0}


def test_fragments_keep_what_the_main_tree_reaches():
    """Stub 9 reaches hub 1, but 1 (and its subtree) stay in the main tree."""
    assert tree().fragments() == {9, 5}


def test_parent_and_length():
    t = tree()
    assert t.parent[2] == 1 and t.length({0, 1}) == 40.0


def test_bfs_edges_and_first_below():
    t = tree()
    assert list(t.bfs_edges([0])) == [(0, 1), (1, 2), (1, 3)]
    assert t.first_below(0, lambda g: g in {2, 3}) == 2          # breadth-first, first child
    assert t.first_below(0, lambda g: False) is None


def test_from_groups_same_as_json_view():
    groups = {0: {"length": 5.0}, 1: {"length": 3.0}}
    adjacency = {0: {"upstream": [], "downstream": [1]}, 1: {"upstream": [0], "downstream": []}}
    t = GroupTree.from_groups(groups, adjacency, [0])
    assert t.children[0] == [1] and t.parent[1] == 0 and t.length(t.subtree(0)) == 8.0
