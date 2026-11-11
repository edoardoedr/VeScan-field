"""The group graph of a tree (branches and bifurcations) read from the topology JSON
(05_branch_tree_topology.json: {"roots", "nodes", "edges"}), with the walks the stages use.

One place for the walks VeScan rewrites in every stage (post-order, subtrees, components per
root). The visit orders are VeScan's (stack, children in edge order), so the results match
even where the order matters.
"""
from __future__ import annotations

import json
from collections import deque


class GroupTree:
    def __init__(self, nodes: dict, edges: list, roots: list):
        self.nodes = nodes                                   # {group id (int): node dict}
        self.edges = edges                                   # [{"parent", "child"}]
        self.roots = list(roots)
        self.children = {g: [] for g in nodes}
        self.parent = {}
        for e in edges:
            self.children.setdefault(e["parent"], []).append(e["child"])
            self.parent[e["child"]] = e["parent"]

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls({int(g): n for g, n in data["nodes"].items()}, data["edges"], data["roots"])

    @classmethod
    def from_groups(cls, groups, adjacency, roots):
        """From the groups in memory (build_graph's format: groups[g] with "length",
        adjacency[g] = {"upstream", "downstream"}), before the topology JSON exists."""
        edges = [{"parent": g, "child": c} for g, links in adjacency.items() for c in links["downstream"]]
        return cls(groups, edges, roots)

    def length(self, groups) -> float:
        return sum(self.nodes[g].get("length", 0.0) for g in groups if g in self.nodes)

    def post_order(self, roots=None) -> list:
        """Groups in post-order (children before parents) from the roots; groups that are not
        reached (fragments not attached to the tree) do not appear."""
        order, visited = [], set()
        stack = [(r, False) for r in (self.roots if roots is None else roots)]
        while stack:
            g, expanded = stack.pop()
            if expanded:
                order.append(g)
                continue
            if g in visited:
                continue
            visited.add(g)
            stack.append((g, True))
            stack.extend((c, False) for c in self.children.get(g, []) if c not in visited)
        return order

    def bfs_edges(self, roots=None):
        """(parent, child) pairs breadth-first from the roots, each group once."""
        roots = self.roots if roots is None else roots
        visited, queue = set(roots), deque(roots)
        while queue:
            g = queue.popleft()
            for c in self.children.get(g, []):
                if c not in visited:
                    visited.add(c)
                    yield g, c
                    queue.append(c)

    def first_below(self, root, predicate):
        """The first group, breadth-first from root (included), satisfying predicate; None if none."""
        seen, queue = set(), deque([root])
        while queue:
            g = queue.popleft()
            if g in seen:
                continue
            seen.add(g)
            if predicate(g):
                return g
            queue.extend(c for c in self.children.get(g, []) if c not in seen)
        return None

    def subtree(self, root, excluded=()) -> set:
        """Every group downstream of root (root included), without entering `excluded`."""
        members, stack = set(), [root]
        while stack:
            g = stack.pop()
            if g in members or g in excluded:
                continue
            members.add(g)
            stack.extend(c for c in self.children.get(g, []) if c not in members)
        return members

    def fragments(self) -> set:
        """Groups of every component (one per root) except the longest: fragments detached from
        the real tree. Components overlap (a stub can point to a hub that the main root also
        reaches): whatever the main component reaches is real tree and never a fragment (on a
        real airway tree, without this subtraction a 0 mm stub removed 20% of the tree)."""
        components = sorted(((self.length(s), s) for s in (self.subtree(r) for r in self.roots)),
                            key=lambda c: c[0], reverse=True)
        if len(components) <= 1:
            return set()
        main = components[0][1]
        return set().union(*(s for _, s in components[1:])) - main
