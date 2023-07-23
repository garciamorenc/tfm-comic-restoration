from collections import defaultdict
from itertools import chain
from banyan import SortedDict, OverlappingIntervalsUpdator

def closed_regions(rects):

    # Sweep Line Algorithm to set up adjacency sets:
    neighbors = defaultdict(set)
    status = SortedDict(updator=OverlappingIntervalsUpdator)
    events = sorted(chain.from_iterable(
            ((r.left, False, r), (r.right, True, r)) for r in set(rects)))
    for _, is_right, rect in events:
        for interval in status.overlap(rect.vertical):
            neighbors[rect].update(status[interval])
        if is_right:
            status.get(rect.vertical, set()).discard(rect)
        else:
            status.setdefault(rect.vertical, set()).add(rect)

    # Connected Components Algorithm for graphs:
    seen = set()
    def component(node, neighbors=neighbors, seen=seen, see=seen.add):
        todo = set([node])
        next_todo = todo.pop
        while todo:
            node = next_todo()
            see(node)
            todo |= neighbors[node] - seen
            yield node
    for node in neighbors:
        if node not in seen:
            yield component(node)