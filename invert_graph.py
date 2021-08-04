from domaintools import API
from configparser import ConfigParser

import networkx as nx
import matplotlib.pyplot as plt

import itertools


def get_pivots(data_obj, name, return_data=None, count=0, pivot_threshold=500):
    """
    Does a deep dive through a data object to check count vs pivot threshold.
    Args:
        data_obj: Either a list or dict that needs to check pivot count
        name: pivot category name
        return_data: Holds data to return once we reach the end of the data_obj
        count: Lets us track to know when we are finished with the data_obj
        pivot_threshold: Threshold to include as a pivot.
    """
    if return_data is None:
        return_data = []
    count += 1
    if isinstance(data_obj, dict) and len(data_obj):
        temp_name = name
        for k, v in data_obj.items():
            if isinstance(data_obj[k], (dict, list)):
                name = "{}_{}".format(name, k)
                temp_data = get_pivots(
                    data_obj[k], name, return_data, count, pivot_threshold
                )
                if temp_data:
                    return_data.append([name[1:].upper().replace("_", " "), temp_data])
            name = temp_name
        if "count" in data_obj and (1 < data_obj["count"] < pivot_threshold):
            return data_obj["value"], data_obj["count"]
    elif isinstance(data_obj, list) and len(data_obj):
        for index, item in enumerate(data_obj):
            temp_data = get_pivots(item, name, return_data, count, pivot_threshold)
            if temp_data:
                if isinstance(temp_data, list):
                    for x in temp_data:
                        return_data.append(x)
                elif isinstance(temp_data, tuple):
                    return_data.append([name[1:].upper().replace("_", " "), temp_data])
    count -= 1
    if count:
        return
    else:
        return return_data


def main(pivot_threshold, edge_threshold, exclude_list):
    config = ConfigParser()
    config.read("config.ini")

    api = API(config["DOMAINTOOLS"]["API_USER"], config["DOMAINTOOLS"]["API_KEY"])
    # x = {
    #     "query": {
    #         "search_hash": "U2FsdGVkX1+fY1VmVaTbmaLBFbP23gMPxMulxfv//YCYKGNmovQdCSmSGt23b/4qBGV6ZoeVTlS7/HQBNgc3CTr9iy1/WLFFixhEf0u1TznFFCvFo0lggJMuZN24GF5Wfq83+kMhjCCMmOVhWeHyFf2qKXg2hXVkV8tWySfBH17Fyq3XZKYtJHOb3YkSs+e/QqTjqNH8NcP5q4aSPn5qYvs872tmQp1F3CPyRaY7FupxNqaSbrKgiIklBq9gt87wTZ2NyjwApxwGoSg+adxEhE6btSNvv8WObUKvvkXpipQmdEzZH3UfXyAneZyPhJXm6L+qnpEv0otiaz00FI2JSmp0VAJJEZGucItHy7dqxaz77+CDiTLtS4nIagPeFqVhs3BHoAuMK0H+vPKANiYETDd7sDerrPwcC6JPj4Z7q5GBawK8NuDu4lacTjE948L6cvrvkn/cgo17g8F6RbBzf8/rF7HaMaxzyAuXpLVxegUcMCPsOLPtEgVsAkbO6KFokjIuFX6GJto98G7L3ufg8Otkc2UyjINmJu1rTblnzoPyQZHAf4eerrp8S3AXYbVgXSJyyVtIY5LOlEjwP3TeOJTvmpCnw/4g6qxBjZDwjQtuM6ajwRVm65lfAsxYbLXdaoC36e4Mt/K9AQ0fHHmYDPmo3FtW0F5NmXWZSjXluLpxoXH8XsxJ7IyDnZfKk0621QEcpxhHx8F7gjtEF7DvPqUXbM86EXsu3y+xfU0f4kQkmGTEVV4pKDmLyntycp32hfP3qrcE9bXzkYkGPOLHQUH3jjM6IWFPRb0mFyEB5/vOc+EnSs7eLWNnoNP8PVYx1T70ttPr8KntI88aAT5BrUu10VJD0meeEUg+yzlK5eE="
    #     }
    # }
    x = {'query': {'ip': "107.152.46.105"}}

    results = api.iris_investigate(**x["query"]).data()
    graph = nx.Graph()
    pv_dict = {}

    for domain in results.get("response").get("results"):
        # GET PIVOTS
        nps = get_pivots(domain, "", pivot_threshold=pivot_threshold)
        pv_list = []
        for p in nps:
            if p[0] not in exclude_list:
                pv_list.append("{}_{}".format(p[0], p[1][0]))
        # CREATE POSSIBLE NODES AND POSSIBLE EDGES
        x = itertools.combinations(pv_list, 2)
        for g in x:
            if "{}:::{}".format(g[0], g[1]) in pv_dict:
                if domain["domain"] not in pv_dict["{}:::{}".format(g[0], g[1])]:
                    pv_dict["{}:::{}".format(g[0], g[1])].append(domain["domain"])
            else:
                pv_dict["{}:::{}".format(g[0], g[1])] = [domain["domain"]]

    b_pv_list = []
    my_set = set()

    # FILTER OUT EDGES THAT DON'T MEET THRESHOLD
    for k, v in pv_dict.items():
        if len(v) > edge_threshold:
            a = k.split(":::")
            b_pv_list.append([a[0], a[1], len(v)])
            my_set.add(a[0])
            my_set.add(a[1])
            print(k, v, len(v))

    # CREATE NODES
    for m in my_set:
        graph.add_node(m)

    # CREATE EDGES
    for m in b_pv_list:
        graph.add_edge(m[0], m[1])

    # edge_labels = dict([((n1, n2), f'1')
    #                     for n1, n2 in graph.edges])
    nx.draw(graph, pos=nx.spring_layout(graph, scale=2), with_labels=True)
    plt.margins(x=0.4)
    plt.show()


if __name__ == "__main__":
    EDGE_THRESHOLD = 1
    PIVOT_THRESHOLD = 500
    EXCLUSION_LIST = []
    main(PIVOT_THRESHOLD, EDGE_THRESHOLD, EXCLUSION_LIST)

