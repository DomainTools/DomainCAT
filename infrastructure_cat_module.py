import os
import sys
import re
import json
import math
from difflib import SequenceMatcher
import plotly.graph_objects as go
import requests
import networkx as nx
import pandas as pd
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from ipywidgets import interactive, HBox, VBox
import ipywidgets as widgets
from IPython.display import HTML, display
import tabulate
from dotenv import dotenv_values
from domaintools import API
from configparser import ConfigParser

import networkx as nx
import matplotlib.pyplot as plt

import itertools

# load REST API creds from .env file
dcat_config = dotenv_values(".env")


def show_iris_query_ui(domain_list_ui, search_hash_ui):
    lookup_ui = widgets.VBox([
        widgets.Label(value="Enter a return delimited list of domains to lookup (no commas, no quotes)"),
        domain_list_ui,
        widgets.Label(value="Or..."),
        widgets.Label(value="Enter an Iris search hassh to lookup"),
        search_hash_ui,
    ])
    return lookup_ui


def clean_domain_list(domain_list_ui):
    # remove any quotes, spaces, or defanging square brackets
    full_domain_list = domain_list_ui.value.strip().replace(' ', '').replace('"', '').replace("'", "").replace('[',
                                                                                                               '').replace(
        ']', '')
    # replace commas with new lines
    full_domain_list = full_domain_list.replace(",", "\n")
    # update the widget
    domain_list_ui.value = full_domain_list
    # split into array
    return full_domain_list.split("\n")


def get_rest_api_creds(api_username_ui, api_pw_ui):
    api_username = api_username_ui.value
    if len(api_username) == 0:
        api_username = dcat_config["IRIS_API_USERNAME"]
    api_key = api_pw_ui.value
    if len(api_key) == 0:
        api_key = dcat_config["IRIS_API_KEY"]
    return api_username, api_key


def query_iris_rest_api(api_username_ui, api_pw_ui, domain_list_ui, search_hash_ui):
    api_username, api_key = get_rest_api_creds(api_username_ui, api_pw_ui)
    api = API(api_username, api_key)
    if len(domain_list_ui.value) > 0:
        # split list of domains into groups of 100 because of API restrictions
        results = []
        full_domain_list = clean_domain_list(domain_list_ui)
        max_domains = 100
        start = 0
        end = max_domains
        for _ in range(math.ceil(len(full_domain_list) / max_domains)):
            # slice out max domains to query
            partial_domain_list = full_domain_list[start:end]
            # build query string
            domain_list = ",".join(partial_domain_list)
            iris_query = {"domain": domain_list}
            # query rest api
            print(f"...querying Iris REST API for {len(partial_domain_list)} domains")
            iris_results = api.iris_investigate(**iris_query)
            # build up the set of return domain objects
            results += iris_results.response().get('results', {})
            # update slice indexes
            start = end
            end += max_domains
        return results
    elif len(search_hash_ui.value) > 0:
        iris_query = {"search_hash": search_hash_ui.value}
        iris_results = api.iris_investigate(**iris_query)
        # print(iris_results.status)
        iris_results = iris_results.response().get('results', {})
        return iris_results
    else:
        print(
            "Domain List and Search Hash text boxes are empty. Please enter either a list of domains or search hash to lookup")
        raise Exception("Domain List and Search Hash text boxes are empty")


class Config(object):
    """ Little helper class to hold all the config values"""


class Domain(object):
    """ Little helper class to hold the domain name and risk score
    """

    def __init__(self, domain_json):
        self.json = domain_json
        self.name = domain_json["domain"]
        self.risk_score = domain_json["domain_risk"]['risk_score']
        self.pivots = {}
        self.label = f"{self.name} ({self.risk_score})"

    def __str__(self):
        return f"name: {self.name}, risk: {self.risk_score}"

    def __repr__(self):
        return str(self)


class DomainRelationship(object):
    def __init__(self, weight: float, category: str):
        # this is the maximum weight that an edge can have.
        # Adjust this if you want to play around with stronger edge weights
        self.max_weight = 5.0
        self.weight = weight
        self.categories = [category]

    def __str__(self):
        return f"weight: {self.weight}, categories: {self.categories}"

    def __repr__(self):
        return str(self)

    def add(self, weight: float, category: str):
        """ Note: certain pivot categories can be added more than once for 2 domains;
        things like IP and name server. For example, two domains could be on the same set of 5
        IP addreese. For now the weights are just summed if there are more than one pivots of
        the same category, but maybe we need a different strategy. Since IPs have multiple pivots
        (ip address, country code, asn, isp) this means if there were 5 shared IPs between two
        domains, the weight would be: 4 * 5 * pivot_weight.
        This might over amplify the edge strength
        """
        if category not in self.categories:
            # this helps by not overly boosting the edge weight if two domains share
            # multipel IP addresses
            self.weight += weight
        self.weight = min(self.weight, self.max_weight)
        self.categories.append(category)

    def get_description(self):
        return "<br>".join(sorted(self.categories))


class Pivot(object):
    def __init__(self, category, value, global_count):
        self.category = category
        self.value = value
        self.global_count = global_count
        self.domains = set()

    #     def union(self, other: "Pivot"):
    #         self.domains.union(other.domains)

    def label(self):
        #         return f"category: {self.category}: value: {self.value} ({self.global_count})"
        return f"{self.category}: {self.value} ({self.global_count})"

    def __str__(self):
        return f"category: {self.category}, " \
               f"value: {self.value}, " \
               f"global_count: {self.global_count}, " \
               f"domains: {self.domains}"

    def __repr__(self):
        return str(self)


# build graph
def get_edge_count(n: int):
    # for a complete graph, the edge count is: n(n-1)/2
    return n * (n - 1) / 2


# def pivot_on_matching_substrings(graph: "Graph", domains: dict, config: "Config"):
#     """Create pivots between domains that share a common substring of
#     `config.longest_common_substring` chars long.
#
#     Note: SequenceMatcher has some known issues with not finding the longest match in very long
#     strings, but does a pretty good job with shorter strings such as domain names.
#     https://stackoverflow.com/questions/18715688/find-common-substring-between-two-strings
#     """
#     domain_names = list(domains.keys())
#     for x in range(len(domain_names)):
#         domain1 = domain_names[x]
#         string1 = domain1.split('.')[0]
#         # pull out substrings to ignore
#         if config.ignore_substrings and len(config.ignore_substrings) > 0:
#             for ignore in config.ignore_substrings:
#                 string1 = string1.replace(ignore, "")
#         for y in range(x + 1, len(domain_names)):
#             domain2 = domain_names[y]
#             string2 = domain2.split('.')[0]
#             # pull out substrings to ignore
#             if config.ignore_substrings and len(config.ignore_substrings) > 0:
#                 for ignore in config.ignore_substrings:
#                     string2 = string2.replace(ignore, "")
#             # find the longest common substring between the two domains
#             matcher = SequenceMatcher(None, string1, string2, False)
#             match = matcher.find_longest_match(0, len(string1), 0, len(string2))
#             longest_match = string1[match.a: match.a + match.size]
#             # check if the matching substring is long enough
#             if len(longest_match) >= config.longest_common_substring:
#                 # add pivots
#                 _append_value_to_pivot(
#                     graph,
#                     "longest_common_substring",
#                     longest_match, None,
#                     domains[domain1], config)
#                 _append_value_to_pivot(
#                     graph,
#                     "longest_common_substring",
#                     longest_match, None,
#                     domains[domain2], config)


def build_pivot_graph(iris_results: list, config: "Config"):
    """ Main workflow function that takes the results from an Iris Investigate query and
    builds the graph object of how each of the domains in the query are connected to each other"""

    # parse the Iris API Result to build the pivot data structure
    graph, domains = init_local_pivot_graph(iris_results, config)
    print(len(graph.nodes))
    print()

    # normalize registrar pivots (see note in function comments)
    # if "registrar" in pivot_categories and config.normalize_registrars:
    #    normalize_similar_registrars(pivot_categories["registrar"])

    # create pivots for longest common substrings
    # pivot_on_matching_substrings(graph, domains, config)
    # print(len(graph.nodes))
    # print()

    # trim pivots from graph that have less than the set count threshold or contain all domains
    # graph = trim_pivots(graph, len(domains), config)
    # print(len(graph.nodes))
    # print()

    # trim unconnected domains and domains with only a create date pivot
    # TURBO: I'm not sure yet how to do this
    #     trimmed_unconnected_domains = trim_unconnected_domains(graph, domains, config)
    #     print(len(graph.nodes))
    #     print()

    #     trimmed_create_date_domains = trim_domains_with_only_create_date_pivot(graph, pivot_categories)
    #     print(len(graph.nodes))
    #     print()

    #     print(f"{len(trimmed_unconnected_domains)} "
    #           f"domains trimmed because they were not connected to other domains")
    #     print(f"{len(trimmed_create_date_domains)} "
    #           f"domains trimmed because create_date was the only pivot")
    print(f"{len(graph.nodes)} nodes in graph structure \n")

    # build the graph structure based on the domain pivots
    graph = build_local_pivot_graph(graph, domains, config)
    return (graph, domains,
            {
                #                 "unconnected": trimmed_unconnected_domains,
                #                 "create_date": trimmed_create_date_domains
            }
            )


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


def build_infra_graph(iris_results: list, config: "Config"):
    graph = nx.Graph()
    pv_dict = {}
    config.domain_risk_dict = {}
    for domain in iris_results:
        if domain["domain"] not in config.domain_risk_dict:
            config.domain_risk_dict[domain["domain"]] = domain.get("domain_risk", {}).get("risk_score", 0)
        # GET PIVOTS
        nps = get_pivots(domain, "", pivot_threshold=config.pivot_threshold)
        pv_list = []
        for p in nps:
            if p[0] not in config.exclude_list:
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
        if len(v) > config.edge_threshold:
            a = k.split(":::")
            b_pv_list.append([a[0], a[1], v, len(v)])
            my_set.add(a[0])
            my_set.add(a[1])
            # print(k, v, len(v))

    # CREATE NODES
    for m in my_set:
        graph.add_node(m, color='blue', size=0)

    # CREATE EDGES
    for m in b_pv_list:
        graph.add_edge(m[0], m[1], domains=m[2], length=m[3])
    return graph, config


def build_pair_infra_graph(iris_results: list, config: "Config"):
    graph = nx.Graph()
    pv_dict = {}
    config.domain_risk_dict = {}
    for domain in iris_results:
        if domain["domain"] not in config.domain_risk_dict:
            config.domain_risk_dict[domain["domain"]] = domain.get("domain_risk", {}).get("risk_score", 0)
        # GET PIVOTS
        nps = get_pivots(domain, "", pivot_threshold=config.pivot_threshold)
        pv_list = [
            "{}_{}".format(p[0], p[1][0])
            for p in nps
            if p[0] not in config.exclude_list
        ]

        # CREATE POSSIBLE NODES AND POSSIBLE EDGES
        x = itertools.combinations(pv_list, 2)
        # print(x)
        i_list = []
        for g in x:
            # print("{}:::{}".format(g[0], g[1]))
            if "{}:::{}".format(g[0], g[1]) not in i_list and g[0] != g[1]:
                i_list.append("{}:::{}".format(g[0], g[1]))
        y = itertools.combinations(i_list, 2)
        for g in y:

            if "{}|||{}".format(g[0], g[1]) in pv_dict:
                if domain["domain"] not in pv_dict["{}|||{}".format(g[0], g[1])]:
                    pv_dict["{}|||{}".format(g[0], g[1])].append(domain["domain"])
            else:
                pv_dict["{}|||{}".format(g[0], g[1])] = [domain["domain"]]
    # print(pv_dict)
    b_pv_list = []
    my_set = set()

    # FILTER OUT EDGES THAT DON'T MEET THRESHOLD
    for k, v in pv_dict.items():
        if len(v) > config.edge_threshold:
            a = k.split("|||")
            if a[0] != a[1]:
                b_pv_list.append([a[0], a[1], v, len(v)])
                my_set.add(a[0])
                my_set.add(a[1])
                # print(k, v, len(v))

        # CREATE NODES
    for m in my_set:
        graph.add_node(m, color='blue', size=0)

        # CREATE EDGES
    for m in b_pv_list:
        graph.add_edge(m[0], m[1], domains=m[2], length=m[3])
    return graph, config


def calc_viz_layout(layout: str, graph: "Graph", dimension: int):
    # KK layout only
    if layout == "kk":
        return nx.layout.kamada_kawai_layout(graph, dim=dimension)

    # spring layout only
    if layout == "fr":
        return nx.layout.spring_layout(graph, dim=dimension)

    # kk layout as initialization for spring layout
    if layout == "kk_to_fr":
        pos = nx.layout.kamada_kawai_layout(graph, dim=dimension, weight=None)
        return nx.layout.spring_layout(graph, pos=pos, dim=dimension)

    # spring layout as initialization for kk layout
    if layout == "fr_to_kk":
        pos = nx.layout.spring_layout(graph, dim=dimension)
        return nx.layout.kamada_kawai_layout(graph, pos=pos, dim=dimension)
    raise Exception("invalid layout choice")


def average_risk_score(domain_list, domain_dict):
    total = sum(domain_dict[d] for d in domain_list)
    avg_risk_score = int(total / len(domain_list))
    # print(avg_risk_score)
    if avg_risk_score >= 90:
        color = 'red'
    elif avg_risk_score >= 75:
        color = 'orange'
    elif avg_risk_score >= 55:
        color = 'yellow'
    else:
        color = 'green'
    return color, avg_risk_score


def build_3d_graph_layout(graph: "Graph", config):
    """ Build the graph layout based on the specified algorithm and get the node positions
    in xyz dimensions"""

    pos = calc_viz_layout("kk_to_fr", graph, 3)

    node_labels, node_risk_scores, node_size, names, Xn, Yn, Zn = [], [], [], [], [], [], []
    i = 0
    for node in graph.nodes(data=True):
        # build x,y,z coordinates data structure for nodes
        Xn.append(pos[node[0]][0])
        Yn.append(pos[node[0]][1])
        Zn.append(pos[node[0]][2])
        domain_set = set()
        for e in graph.edges(node[0], data=True):
            domain_set.update(e[2]['domains'])
        domain_list = list(domain_set)
        color, avg_risk_score = average_risk_score(domain_list, config.domain_risk_dict)
        node_labels.append(
            "{}<br>Avg Risk Score: {}<br>Number of unique domains on edges: {}".format(node[0], avg_risk_score,
                                                                                       len(domain_list)))
        node_risk_scores.append(color)
        node_size.append(len(domain_list))
        names.append(domain_list)

    if not config.node_size:
        node_size = 6

    # build x,y,z coordinates data structure for edges
    Xe, Ye, Ze = [], [], []
    for e in graph.edges:
        u = pos[e[0]]
        v = pos[e[1]]
        Xe += [u[0], v[0], None]
        Ye += [u[1], v[1], None]
        Ze += [u[2], v[2], None]

    # Create the 3d Plotly graph and render it
    # build line objects for our edges
    trace1 = go.Scatter3d(x=Xe, y=Ye, z=Ze,
                          mode='lines',
                          name='domains',
                          line=dict(color='rgb(125,125,125)', width=0.5),
                          opacity=0.9,
                          hoverinfo='none')

    trace2 = go.Scatter3d(
        x=Xn, y=Yn, z=Zn,
        mode='markers',
        name='pivots',
        marker=dict(
            symbol='circle',
            size=node_size,
            color=node_risk_scores,
            line=dict(color='rgb(50,50,50)', width=0.5),
        ),
        text=node_labels,
        hoverinfo='text')

    # background definition, but everything is turned off
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='')

    layout = go.Layout(
        title=f"Graph of interconnected infrastructure ({len(node_labels)} infra nodes)",
        width=1000, height=1000,
        showlegend=False,
        scene=dict(xaxis=dict(axis), yaxis=dict(axis), zaxis=dict(axis)),
        margin=dict(t=100), hovermode='closest')

    data = [trace1, trace2]
    fig = go.FigureWidget(data=data, layout=layout)

    # handle selection of domains
    # def node_selection_fn(trace, points, selector):
    #     selected_domains = [names[idx] for idx in points.point_inds]
    #     update_selected_domains(selected_domains)

    # handle node click events
    def node_click_fn(trace, points, selector):
        if len(points.point_inds) > 1:
            print(f"node_click passed in more than 1 point: {points.point_inds}")

        # clear the old selected points
        # trace.selectedpoints = []
        # if len(points.point_inds) == 0:
        # return

        # get the list of selected domain names
        selected_domains = [names[idx] for idx in points.point_inds]
        # for id in points.point_inds:
        # selected_domains = selected_domains + trace.customdata[id]

        # set the new selected points
        # don't like having to loop in a loop to get the domain index, but I don't know a better way
        # trace.selectedpoints = points.point_inds + [names.index(name) for name in trace.customdata[id]]

        update_selected_domains(selected_domains)

    def update_selected_domains(selected_domains):
        if len(selected_domains) == 0:
            return

        # sort domains by length, then alpha
        selected_domains.sort(key=len, reverse=True)
        with out:
            # write selected domains to the output widget
            print(f"Selected Infra: ({len(selected_domains)})\n")
            for selected_domain in selected_domains:
                print(selected_domain)
        out.clear_output(wait=True)

        # calc pivots selected domains have in common
        # get_2d_shared_pivots(graph, selected_domains)

    # event handler for node selection
    # fig.data[1].on_selection(node_selection_fn)
    # event handle for node click
    fig.data[1].on_click(node_click_fn)

    # Create a table FigureWidget that updates the list of selected domains
    out = widgets.Output(layout={'border': '1px solid black'})
    domain_ui = widgets.VBox((fig, out))
    return domain_ui


def build_2d_graph_layout(graph: "Graph", config):
    """ build the graph layout based on the specified algorithm and get the node positions
    in xy dimensions"""
    pos = calc_viz_layout("kk_to_fr", graph, 2)
    # pos = calc_viz_layout("fr_to_kk", g, 2)

    # build edge data
    edge_x, edge_y = [], []
    for e in graph.edges():
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    # create edge scatter plot
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        opacity=0.6
    )

    # build node data
    node_adjacencies, node_risk_scores, node_text, node_labels, node_size, node_x, node_y = [], [], [], [], [], [], []
    names = list(graph.nodes)
    for name in graph.nodes(data=True):
        domain = graph.nodes[name[0]]
        x, y = pos[name[0]]
        node_x.append(x)
        node_y.append(y)
        # get the domain's connected nodes
        neighbors = list(graph.neighbors(name[0]))
        node_adjacencies.append(neighbors)
        domain_set = set()
        for e in graph.edges(name[0], data=True):
            domain_set.update(e[2]['domains'])
        domain_list = list(domain_set)
        color, avg_risk_score = average_risk_score(domain_list, config.domain_risk_dict)
        node_labels.append(
            "{}<br>Avg Risk Score: {}<br>Number of unique domains on edges: {}".format(name[0], avg_risk_score,
                                                                                       len(domain_list)))
        node_risk_scores.append(color)
        node_size.append(len(domain_list))
        names.append(domain_list)

    if not config.node_size:
        node_size = 6

    # build node scatter plot
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_labels,
        customdata=node_adjacencies,
        marker=dict(
            showscale=True,
            reversescale=True,
            color=node_risk_scores,
            colorscale=[[0.0, 'red'], [0.3, 'orange'], [0.5, 'yellow'], [1.0, 'green']],
            # cmin/cmax needed so plotly doesn't normalize the scores to calculate the color
            cmin=0, cmax=100,
            size=node_size,
            colorbar=dict(
                thickness=15,
                title='Risk Score',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    # create the jup widget holder for plotly
    fig = go.FigureWidget(
        [edge_trace, node_trace],
        layout=go.Layout(
            title=f'Graph of interconnected infrastructure ({len(node_labels)} infra nodes)',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=5, l=5, r=5, t=30),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    )

    # handle selection of domains
    def node_selection_fn(trace, points, selector):
        selected_domains = [names[idx] for idx in points.point_inds]
        update_selected_domains(selected_domains)

    # handle node click events
    def node_click_fn(trace, points, selector):
        if len(points.point_inds) > 1:
            print(f"node_click passed in more than 1 point: {points.point_inds}")

        # clear the old selected points
        trace.selectedpoints = []
        if len(points.point_inds) == 0:
            return

        # get the list of selected domain names
        selected_domains = [names[idx] for idx in points.point_inds]
        for id in points.point_inds:
            selected_domains = selected_domains + trace.customdata[id]

        # set the new selected points
        # don't like having to loop in a loop to get the domain index, but I don't know a better way
        trace.selectedpoints = points.point_inds + [names.index(name) for name in trace.customdata[id]]

        update_selected_domains(selected_domains)

    def update_selected_domains(selected_domains):
        if len(selected_domains):
            return

        # sort domains by length, then alpha
        selected_domains.sort(key=len, reverse=True)
        with out:
            # write selected domains to the output widget
            print(f"Selected Infra: ({len(selected_domains)})\n")
            for selected_domain in selected_domains:
                print(selected_domain)
        out.clear_output(wait=True)


    # event handler for node selection
    fig.data[1].on_selection(node_selection_fn)
    # event handle for node click
    fig.data[1].on_click(node_click_fn)

    # Create a table FigureWidget that updates the list of selected domains
    out = widgets.Output(layout={'border': '1px solid black'})
    domain_ui = widgets.VBox((fig, out))
    return domain_ui


def get_shared_pivots(graph: "Graph", selected_domains: list):
    shared_pivots = {}
    for name in selected_domains:
        domain = graph.nodes[name]["domain"]
        for cat in domain.pivot_categories:
            for cat_value in domain.pivot_categories[cat]:
                key = f"{cat}: {cat_value}"
                if key not in shared_pivots:
                    shared_pivots[key] = []
                shared_pivots[key].append(domain)

    # filter by pivots that have >= n domains
    shared_pivots = {k: v for k, v in shared_pivots.items() if len(v) >= 3}
    return shared_pivots