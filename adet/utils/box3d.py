import torch
def box_overlap(b1, b2):
    ''' b1: 6
    b2: 6'''
    seg = [line_seg(b1[[i,i+3]], b2[[i,i+3]]) for i in [0,1,2]]
    if None in seg:
        return None
    return torch.tensor(seg).transpose(-1,-2).flatten()

def line_seg(l1, l2):
    seg =  [max(l1[0],l2[0]), min(l1[1], l2[1])]
    if seg[1]>seg[0]:
        return seg


import numpy as np
def polygon_intersection(poly1, poly2):
    """
    Use the Sutherland-Hodgman algorithm to compute the intersection of 2 convex polygons.
    """
    def line_intersection(e1, e2, s, e):
        dc = e1 - e2
        dp = s - e
        n1 = np.cross(e1, e2)
        n2 = np.cross(s, e)
        n3 = 1.0 / (np.cross(dc, dp))
        return np.array([(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3])

    def is_inside_edge(p, e1, e2):
        """Return True if e is inside edge (e1, e2)"""
        return np.cross(e2-e1, p-e1) >= 0

    output_list = poly1
    # e1 and e2 are the edge vertices for each edge in the clipping polygon
    e1 = poly2[-1]

    for e2 in poly2:
        input_list = output_list
        output_list = []
        s = input_list[-1]

        for e in input_list:
            if is_inside_edge(e, e1, e2):
                # if s in not inside edge (e1, e2)
                if not is_inside_edge(s, e1, e2):
                    # line intersects edge hence we compute intersection point
                    output_list.append(line_intersection(e1, e2, s, e))
                output_list.append(e)
            # is s inside edge (e1, e2)
            elif is_inside_edge(s, e1, e2):
                output_list.append(line_intersection(e1, e2, s, e))

            s = e
        e1 = e2
    return np.array(output_list)