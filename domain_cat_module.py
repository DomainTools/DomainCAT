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


def query_domain_data(api_username: str, api_key: str, search_hash: str):
    root_api_url = "https://api.domaintools.com/v1/iris-investigate/"
    iris_query = {"api_username": api_username, "api_key": api_key, "search_hash": search_hash}

    resp = requests.post(root_api_url, data=iris_query)
    if resp.status_code != 200:
        raise Exception('POST /iris-investigate/ {}'.format(resp.status_code))
    iris_results = resp.json()
    return iris_results


class Config(object):
    """ Little helper class to hold all the config values"""

    
class Domain(object):
    """ Little helper class to hold the domain name and risk score
    """
    def __init__(self, domain_json):
        self.index = None
        self.json = domain_json
        self.name = domain_json["domain"]
        self.risk_score = domain_json["domain_risk"]['risk_score']
        self.pivot_categories = {}
    
    def __str__(self):
        return f"index: {self.index}, name: {self.name}, risk: {self.risk_score}"
    
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
        if self.weight > self.max_weight:
            self.weight = self.max_weight
        self.categories.append(category)
    
    def get_description(self):
        return "<br>".join(sorted(self.categories))


class PivotValue(object):
    def __init__(self, pivot_value, pivot_count):
        self.pivot_value = pivot_value
        self.pivot_count = pivot_count
        self.domains = set()
        
    def union(self, other: "PivotValue"):
        self.domains.union(other.domains)
        
    def __str__(self):
        return f"pivot_value: {self.pivot_value}, " \
               f"pivot_count: {self.pivot_count}, " \
               f"domains: {self.domains}"
    
    def __repr__(self):
        return str(self)

        
def get_edge_count(n: int):
    # for a complete graph, the edge count is: n(n-1)/2
    return n * (n - 1) / 2


def build_domain_pivot_graph(iris_results: list, config: "Config"):
    """ Main workflow function that takes the results from an Iris Investigate query and
    builds the graph object of how each of the domains in the query are connected to each other"""
    
    # parse the Iris API Result to build the pivot data structure
    pivot_categories, domain_list = build_pivot_categories(iris_results, config)

    # normalize registrar pivots (see note in function comments)
    #if "registrar" in pivot_categories and config.normalize_registrars:
    #    normalize_similar_registrars(pivot_categories["registrar"])

    # create pivots for longest common substrings
    pivot_on_matching_substrings(pivot_categories, domain_list, config)

    # trim pivots from graph that have less than the set count threshold or contain all domains
    trim_pivots(pivot_categories, len(domain_list), config)

    # trim unconnected domains and domains with only a create date pivot
    domain_list, trimmed_unconnected_domains = trim_unconnected_domains(
        pivot_categories, domain_list, config)
    domain_list, trimmed_create_date_domains = trim_domains_with_only_create_date_pivot(
        pivot_categories, domain_list)

    print(f"{len(trimmed_unconnected_domains)} "
          f"domains trimmed because they were not connected to other domains")
    print(f"{len(trimmed_create_date_domains)} "
          f"domains trimmed because create_date was the only pivit")
    print(f"{len(domain_list)} domains in pivot structure \n")

    # build the graph structure based on the domain pivots
    graph = build_domain_graph(domain_list, pivot_categories, config)
    return (pivot_categories,
            domain_list,
            graph,
            {"unconnected": trimmed_unconnected_domains,
             "create_date": trimmed_create_date_domains})


def build_pivot_categories(iris_results: list, config: "Config"):
    """ Collect pivot categories found in result set ("ssl_hash" for example)"""
    pivot_categories = {}
    # collect a list of all domains
    domain_list = {}

    for domain_json in iris_results:
        # create a domain object
        domain = Domain(domain_json)
        domain_list[domain.name] = domain

        append_value_with_count(pivot_categories, 'adsense', domain_json, domain, config)
        append_value_with_count(pivot_categories, 'google_analytics', domain_json, domain, config)
        append_value_with_count(pivot_categories, 'create_date', domain_json, domain, config)
        append_value_with_count(pivot_categories, 'redirect_domain', domain_json, domain, config)
        append_value_with_count(pivot_categories, 'registrar', domain_json, domain, config)

        # haven't seen "ssl_email" in the wild yet, so not sure if it is a value/count or just value 
        append_values_with_counts(pivot_categories, 'ssl_email', domain_json, domain, config)

        # IPs are composite objects, so pull out each value for each IP
        for ip_json in domain_json["ip"]:
            # at some point add logic to add /24 in here
            append_value_with_count(pivot_categories, 'address', ip_json, domain, config, 'ip_address')
            append_value_with_count(pivot_categories, 'country_code', ip_json, domain, config, 'ip_country_code')
            append_value_with_count(pivot_categories, 'isp', ip_json, domain, config, 'ip_isp')
            append_values_with_counts(pivot_categories, 'asn', ip_json, domain, config, 'ip_asn')

        # name servers are composite objects, so pull out each value for each name server
        for ns_json in domain_json["name_server"]:
            append_value_with_count(pivot_categories, 'host', ns_json, domain, config, 'ns_host')
            append_value_with_count(pivot_categories, 'domain', ns_json, domain, config, 'ns_domain')
            append_values_with_counts(pivot_categories, 'ip', ns_json, domain, config, 'ns_ip')

        append_value(pivot_categories, 'tld', domain_json, domain, config)

        # ssl certs are composite objects, so pull out each value for each ssl cert
        for ssl_json in domain_json['ssl_info']:
            append_value_with_count(pivot_categories, 'hash', ssl_json, domain, config, "ssl_hash")
            append_value_with_count(pivot_categories, 'subject', ssl_json, domain, config, "ssl_subject")
            append_value_with_count(pivot_categories, 'organization', ssl_json, domain, config, "ssl_org")

        # mx servers are composite objects, so pull out each value for each mx server
        for mx_json in domain_json['mx']:
            append_value_with_count(pivot_categories, 'host', mx_json, domain, config, "mx_host")
            append_value_with_count(pivot_categories, 'domain', mx_json, domain, config, "mx_domain")
            append_values_with_counts(pivot_categories, 'ip', mx_json, domain, config, "mx_ip")
            # mx priority might be interesting at some point to node stringth  
    return pivot_categories, domain_list
    

def append_value(pivot_categories: dict,
                 pivot_category: str,
                 json_data: dict,
                 domain: "Domain",
                 config: "Config",
                 new_pivot_category: str = None):
    # check if pivot is in dictionary
    if pivot_category in json_data:
        pivot_value = str(json_data[pivot_category]).strip()

        # check we have a value to add
        if len(pivot_value) > 0:
            _append_value_to_pivot(pivot_categories, pivot_category, pivot_value, None,
                                   domain, config, new_pivot_category)

            
def append_value_with_count(pivot_categories: dict,
                            pivot_category: str,
                            json_data: dict,
                            domain: "Domain",
                            config: "Config",
                            new_pivot_category: str = None):
    # check if pivot is in dictionary
    if pivot_category in json_data:
        if isinstance(json_data[pivot_category], dict): 
            pivot_value = str(json_data[pivot_category]["value"]).strip()
            pivot_count = json_data[pivot_category]["count"]

            # trim pivots that are above the threshold (except create_date)
            if pivot_count < config.count_threshold or pivot_category == "create_date":
                # check we have a value to add
                if len(pivot_value) > 0 and pivot_count > 0:
                    _append_value_to_pivot(pivot_categories, pivot_category, pivot_value,
                                           pivot_count, domain, config, new_pivot_category)
            
            
def append_values_with_counts(pivot_categories: dict,
                              pivot_category: str,
                              json_data: dict,
                              domain: "Domain",
                              config: "Config",
                              new_pivot_category: str = None):
    # check if pivot is in dictionary
    if pivot_category in json_data:
        for pivot in json_data[pivot_category]:
            pivot_value = str(pivot["value"]).strip()
            pivot_count = pivot["count"]
                        
            # check if we want to add this value
            if len(pivot_value) > 0 and pivot_count > 0 and pivot_count < config.count_threshold:
                _append_value_to_pivot(pivot_categories, pivot_category, pivot_value,
                                       pivot_count, domain, config, new_pivot_category)


def _append_value_to_pivot(pivot_categories: dict,
                           pivot_category: str,
                           pivot_value: str,
                           pivot_count: int,
                           domain: "Domain",
                           config: "Config",
                           new_pivot_category: str = None):
    # if we pass in a new_pivot_category, replace pivot_category with new_pivot_category
    if new_pivot_category:
        pivot_category = new_pivot_category

    # check if we're capturing data for this pivot category 
    if pivot_category not in config.pivot_category_config:
        return
    
    # make sure we have the pivot dictionary 
    if pivot_category not in pivot_categories:
        pivot_categories[pivot_category] = {}

    # make sure we have the pivot value set
    if pivot_value not in pivot_categories[pivot_category]:
        pivot_categories[pivot_category][pivot_value] = PivotValue(pivot_value, pivot_count)

    # add domain to the pivot domain array
    pivot_categories[pivot_category][pivot_value].domains.add(domain.name)
    
    # add pivot category and value to the domain
    if pivot_category not in domain.pivot_categories:
        domain.pivot_categories[pivot_category] = []
    domain.pivot_categories[pivot_category].append(pivot_value)


def normalize_similar_registrars(registrar_pivots: dict):
    """ The same registrar can often show up in WHOIS records with different string values.
    For example:
        NAMECHEAP
        NAMECHEAP INC
        NAMECHEAP. INC
        NAMECHEAP, INC
        NAMECHEAP, INC.
    
    This function splits the registrar string by any non character value, and selects the longest
    word as the normalized registar value. If any two registrars share the same normalized value,
    then the domains from those two registrars will be merged. The end goal is all the domains
    from the 5 different namecheap registrars string values shown above would be merged into one.
    
    Note: this isn't a very good solution. There are cases where this will create invalid connections 
    between domains. For example, two different registars that shared a common longest word in 
    their name, link "NAMECHEAP, INC" and "NOT NAMECHEAP, INC". 
    
    It looks like this happens a lot so turning off the feature for now.
    
    TODO: this algorithm needs work. it allows things such as
      good
        PDR LTD. D/B/A PUBLICDOMAINREGISTRY.COM == PDR Ltd. d/b/a PublicDomainRegistry.com
        GODADDY.COM, == LLC GODADDY.COM, INC
        NAMECHEAP, INC == NameCheap, Inc.
      bad
        TUCOWS DOMAINS INC == WILD WEST DOMAINS, INC
        NETWORK SOLUTIONS, == LLC Network Solutions, LLC
        NETWORK SOLUTIONS, == LLC BIGROCK SOLUTIONS LTD
    """
    return
#     registrars = [registrar for registrar in registrar_pivots]
#     for x in range(len(registrars)):
#         reg1 = registrars[x]
#         if reg1 in registrar_pivots:
#             # normalize registrar string
#             reg1_norm = sorted(
#                 list(set(re.findall(r"[\w']+", reg1.lower()))), key=len, reverse=True)[0]
#             for y in range(x+1, len(registrars)):
#                 reg2 = registrars[y]
#                 # normalize registrar string
#                 reg2_norm = sorted(
#                     list(set(re.findall(r"[\w']+", reg2.lower()))), key=len, reverse=True)[0]
#                 if reg1_norm == reg2_norm:
#                     # pick the registrar with the most domains
#                     if registrar_pivots[reg1].pivot_count > registrar_pivots[reg2].pivot_count:
#                         reg_keep = reg1
#                         reg_pop = reg2
#                     else:
#                         reg_keep = reg2
#                         reg_pop = reg1
#                     # combine domains for matching registrars
#                     registrar_pivots[reg_keep].union(registrar_pivots[reg_pop])
#                     # remove reg_pop from dictionary of all registrar pivots
#                     registrar_pivots.pop(reg_pop)
#                     print(f"Merged registrar {reg_pop} into {reg_keep}")


def pivot_on_matching_substrings(pivot_categories: dict, domain_list: dict, config: "Config"):
    """Create pivots between domains that share a common substring of
    `config.longest_common_substring` chars long.
    
    Note: SequenceMatcher has some known issues with not finding the longest match in very long
    strings, but does a pretty good job with shorter strings such as domain names.
    https://stackoverflow.com/questions/18715688/find-common-substring-between-two-strings
    """    
    matches = {}
    domains = list(domain_list.values())
    for x in range(len(domains)):
        domain1 = domains[x]
        string1 = domain1.name.split('.')[0]
        # pull out substrings to ignore
        if config.ignore_substrings and len(config.ignore_substrings) > 0:
            for ignore in config.ignore_substrings:
                string1 = string1.replace(ignore, "")
        for y in range(x+1, len(domains)):
            domain2 = domains[y]
            string2 = domain2.name.split('.')[0]
            # pull out substrings to ignore
            if config.ignore_substrings and len(config.ignore_substrings) > 0:
                for ignore in config.ignore_substrings:
                    string2 = string2.replace(ignore, "")
            # find the longest common substring between the two domains
            matcher = SequenceMatcher(None, string1, string2, False)
            match = matcher.find_longest_match(0, len(string1), 0, len(string2))
            longest_match = string1[match.a: match.a + match.size]
            # check if the matching substring is long enough
            if len(longest_match) >= config.longest_common_substring:
                if longest_match not in matches:
                    matches[longest_match] = set()
                matches[longest_match].add(domain1)
                matches[longest_match].add(domain2)
                # add pivots
                _append_value_to_pivot(
                    pivot_categories, 
                    "longest_common_substring",
                    longest_match, None,
                    domain1, config)
                _append_value_to_pivot(
                    pivot_categories, 
                    "longest_common_substring",
                    longest_match, None,
                    domain2, config)
    return matches     


def trim_pivots(pivot_categories: dict, domain_count: int, config: "Config"):
    """ Remove two types of pivots. Pivots that contain all the domains from the Iris result set, 
    and pivots that have less than the set threshold of domains in them from this Iris result set.
    By defualt, pivots that only have one domain are removed, but this can be configured by 
    setting the min_pivot_size variable to a different value. For example, if you only wanted 
    to use pivots that had 10 or more domains connected to them
    """    
    for pivot_category_key in pivot_categories:
        pivot_category = pivot_categories[pivot_category_key]
        total_pivots = 0
        del_count = 0
        for pivot_value in list(pivot_category.keys()):
            total_pivots += 1
            if len(pivot_category[pivot_value].domains) < config.min_pivot_size:
                # check for pivots with less than the threshold value
                del pivot_category[pivot_value]
                del_count += 1
            elif len(pivot_category[pivot_value].domains) == domain_count:
                # check for pivots with all domains in them
                del pivot_category[pivot_value]
                if config.print_debug_output:
                    print(f"deleted {pivot_category_key}:{pivot_value}. Contained all domains")
        if config.print_debug_output:
            print(f"deleted {del_count} "
                  f"singleton pivots out of {total_pivots} "
                  f"pivots from {pivot_category_key}")

            
def trim_unconnected_domains(pivot_categories: dict, domain_list: dict, config: "Config"):
    """ Remove any domains that have no shared connection to any othe domain
    """
    if config.print_debug_output: print(f"{len(domain_list)} domains in Iris result set")
    connected_domains = set()
    for pivot_category_key in pivot_categories:
        pivot_category = pivot_categories[pivot_category_key]
        for pivot_value in list(pivot_category.keys()):
            pivot_domains = pivot_category[pivot_value].domains
            connected_domains = connected_domains.union(pivot_domains)

    # collect the set of all domains that are connected in some manner
    domain_list_tmp = {}
    for domain_name in connected_domains:
        domain_list_tmp[domain_name] = domain_list[domain_name]
        # remove domain. domain_list becomes the set of all disconnected domains
        domain_list.pop(domain_name)

    if config.print_debug_output: 
        print(f"{len(domain_list_tmp)} domains are interconnected")
        print(f"{len(domain_list)} domains are unconnected")        
        print("Unconnected domains removed from graph:")
        for domain in domain_list:
            print(f"  {domain_list[domain].name}: risk score {domain_list[domain].risk_score}")
    
    return domain_list_tmp, list(domain_list.values())


def trim_domains_with_only_create_date_pivot(pivot_categories: dict, domain_list: dict):
    """ if a domain ONLY has a create_date pivot, then that isn't a very good indicator of
    connectedness."""
    # identify domains to trim
    trimmed_domains = set()
    for domain_name in domain_list:
        domain = domain_list[domain_name]
        if len(domain.pivot_categories) == 1 and "create_date" in domain.pivot_categories:
            trimmed_domains.add(domain)
            
    # remove trimmed domains from domain_list
    for trimmed_domain in trimmed_domains:
        domain_list.pop(trimmed_domain.name)
        
        # remove trimmed domain from the main pivot_categories data structure
        domain_create_date = trimmed_domain.pivot_categories["create_date"][0]
        pivot_categories["create_date"][domain_create_date].remove(trimmed_domain.name)
        if len(pivot_categories["create_date"][domain_create_date]) == 0:
            pivot_categories["create_date"].pop(domain_create_date)
        if len(pivot_categories["create_date"]) == 0:
            pivot_categories.pop("create_date")
        
    return domain_list, list(trimmed_domains)


def get_pivot_connection_weight(pivot_category: str,
                                global_pivot_count: int,
                                local_pivot_count: int,
                                config: "Config"):
    """ If we aren't using the pivot count to set the edge weight, just return a constant value of
    1 for every pivot. If we do want to use the pivot count, use the function:
        1 - (log(pivot count) / (log(max possible pivot count)))
    This creates an inverse log ratio where small pivots have a high edge weight,
    and very large pivots have a low edge weight.
        
    Note: also experimenting with raising this log ratio to different exponents to get greater
    separation between large and small pivots: math.pow(1.0 + inverse_log_ratio, 3) - 1
    """
    if pivot_category not in config.pivot_category_config:
        raise Exception(f"Unexpected Pivot Category: {pivot_category}")
    
    # scale the edge strength based on the ratio of the global pivot count vs the max domains
    if config.scale_edge_strength_by_pivot_count:
        if global_pivot_count is None:
            # Some pivots don't have a count. For example, tld or longest common substring.
            # if global pivot count is None, for now set to 1 (?)
            # But we probably need to then normalize this weigth against the max weight calculated.
            # Also, TLD doesn't have a pivot count because it's often huge. Is that the same.
            #    importance as common substrings? Probably not.
            return 0.5
        inv_ratio = 1.0 - math.log(1.0 + global_pivot_count) / math.log(1.0 + config.max_domains)
        return inv_ratio
#         return math.pow(1.0 + inverse_log_ratio, 3) - 1
    return 1


def build_domain_graph(domain_list: dict, pivot_categories: dict, config: "Config"):
    # init empty graph
    graph = nx.Graph()

    # add nodes
    for index, domain in enumerate(domain_list.values()):
        domain.index = index
        graph.add_node(domain.index,
                       risk_score=domain.risk_score,
                       alpha=1.0,
                       label=f"{domain.name} ({domain.risk_score})")

    # add edges
    edge_count = 0
    for category in pivot_categories:        
        for pivot_value in pivot_categories[category]:
            pv_obj = pivot_categories[category][pivot_value]
            domains = list(pv_obj.domains)

            # get the edge weight and create edges 
            weight = get_pivot_connection_weight(
                category, pv_obj.pivot_count, len(pv_obj.domains), config)
            if weight > 0:
                for x in range(len(domains)):
                    for y in range(x+1, len(domains)):
                        u = domain_list[domains[x]].index
                        v = domain_list[domains[y]].index
                        edge_count += 1
                        if graph.has_edge(u, v):
                            graph[u][v]['relationship'].add(weight, category)
                        else:
                            graph.add_edge(u, v, relationship=DomainRelationship(weight, category))

    # now that all edges are added, set the weight attribute with the adjusted weight
    for edge in graph.edges:
        graph[edge[0]][edge[1]]['weight'] = graph[edge[0]][edge[1]]['relationship'].weight

    print(f"Total Graph Connections: {edge_count}")
    print(f"Distinct Graph Connections: {len(graph.edges)}")
    return graph


def calc_pivot_stats(pivot_categories: dict, domain_list: dict):
    from IPython.display import HTML, display
    import tabulate

    # calc the max number of edges possible for this set of domains
    max_edge_count = get_edge_count(len(domain_list))

    # collect counts for each pivot category
    category_domain_counts = {}
    category_edge_counts = {}
    for category_key in pivot_categories:
        category_domain_counts[category_key] = 0
        category_edge_counts[category_key] = 0
        category = pivot_categories[category_key]
        for pivot_value in category:
            category_domain_counts[category_key] += len(category[pivot_value].domains)

            # if all domains share a pivot value, it would be considered a "connected graph" 
            #   so get the edge count for a connected graph
            edge_count = get_edge_count(len(category[pivot_value].domains))
            category_edge_counts[category_key] += round(edge_count)

    total_connections = 0

    headers = ["Pivot Category",
               "# of Domains",
               "# of Pivots",
               "avg domains per pivot",
               "# of connections"]
    table = []
    total_domains = len(domain_list)
    for category_key in category_domain_counts:
        cat_pivot_count = len(pivot_categories[category_key])
        if cat_pivot_count > 0:
            domain_count = category_domain_counts[category_key]
            edge_count = category_edge_counts[category_key]

            total_connections += edge_count

            avg_domains = domain_count / cat_pivot_count
            percent_of_total_domains = round(100 * (domain_count / total_domains), 2)                
            percent_of_total_edges = round(100 * (edge_count / max_edge_count), 2)            
            table.append([category_key,
                          f"{domain_count} ({percent_of_total_domains}%)",
                          cat_pivot_count,
                          round(avg_domains, 2),
                          f"{edge_count} ({percent_of_total_edges}%)"])

    print(f"{len(domain_list)} Domains in Pivot Structure")
    display(HTML(tabulate.tabulate(table, headers=headers, tablefmt='html')))    


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
    
    
def build_3d_graph_layout(graph: "Graph", domain_list: dict):
    """ Build the graph layout based on the specified algorithm and get the node positions
    in xyz dimensions"""
    pos = calc_viz_layout("kk_to_fr", graph, 3)

    node_labels, node_risk_scores, Xn, Yn, Zn = [], [], [], [], []
    domain_indexes = sorted([domain.index for domain in domain_list.values()])
    for k in domain_indexes:
        # build x,y,z coordinates data structure for nodes
        Xn.append(pos[k][0])
        Yn.append(pos[k][1])
        Zn.append(pos[k][2])

        # get domain colors by risk score
        node = graph.nodes[k]
        node_labels.append(node['label'])
        # get the domain risk score
        node_risk_scores.append(node['risk_score'])

    # build x,y,z coordinates data structure for edges
    Xe, Ye, Ze = [], [], []
    for e in graph.edges:
        u = pos[e[0]]
        v = pos[e[1]]
        Xe+=[u[0], v[0], None]
        Ye+=[u[1], v[1], None]
        Ze+=[u[2], v[2], None]

    # Create the 3d Plotly graph and render it
    # build line objects for our edges
    trace1=go.Scatter3d(x=Xe, y=Ye, z=Ze,
                   mode='lines', 
                   name='edges',
                   line=dict(color='rgb(125,125,125)', width=0.5),
                   opacity=0.9, 
                   hoverinfo='none')

    trace2=go.Scatter3d(
                   x=Xn, y=Yn, z=Zn,
                   mode='markers', 
                   name='domains',
                   marker=dict(
                       symbol='circle', 
                       size=6,
                       showscale=True,
                       color=node_risk_scores, 
                       colorscale=[[0.0, 'red'], [0.3, 'orange'], [0.5, 'yellow'], [1.0, 'green']],
                       reversescale=True,
                       line=dict(color='rgb(50,50,50)', width=0.5),
                       colorbar=dict(
                           thickness=15,
                           title='Risk Score',
                           xanchor='left',
                           titleside='right'
                       ),
                   ),
                   text=node_labels, 
                   hoverinfo='text')    
    
    # background definition, but everything is turned off
    axis=dict(showbackground=False,
              showline=False,
              zeroline=False,
              showgrid=False,
              showticklabels=False,
              title='')

    layout = go.Layout(
             title=f"Graph of interconnected domains ({len(node_labels)} domains)",
             width=1000, height=1000,
             showlegend=False,
             scene=dict(xaxis=dict(axis), yaxis=dict(axis), zaxis=dict(axis)),
             margin=dict(t=100), hovermode='closest')

    data=[trace1, trace2]
    fig=go.Figure(data=data, layout=layout)
    return fig


def build_2d_graph_layout(graph: "Graph", domain_list: dict, get_2d_shared_pivots: "function"):
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
    node_adjacencies, node_risk_scores, node_text, node_x, node_y = [], [], [], [], []
    domains = sorted(domain_list.values(), key=lambda domain: domain.index)
    for domain in domains:
        x, y = pos[domain.index]
        node_x.append(x)
        node_y.append(y)
        # get the domain's connected nodes
        neighbors = list(graph.neighbors(domain.index))
        node_adjacencies.append(neighbors)
        # get the domain text
        domain_name = domain.name    
        risk_score = domain.risk_score
        node_text.append(f'{domain_name}: risk {risk_score}, connections {len(neighbors)}')
        # get the domain risk score
        node_risk_scores.append(domain.risk_score)

    # build node scatter plot
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        customdata=node_adjacencies,
        marker=dict(
            showscale=True,
            colorscale=[[0.0, 'red'], [0.3, 'orange'], [0.5, 'yellow'], [1.0, 'green']],
            reversescale=True,
            color=node_risk_scores,        
            size=10,
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
        title=f'Graph of interconnected domains ({len(node_text)} domains)',
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=5,l=5,r=5,t=30),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    )
    
    # handle selection of domains
    def node_selection_fn(trace, points, selector):
        update_selected_nodes(points.point_inds)
    
    # handle node click events
    def node_click_fn(trace, points, selector):        
        if len(points.point_inds) > 1:
            print(f"node_click passed in more than 1 point: {points.point_inds}")
        trace.selectedpoints = []
        if len(points.point_inds) == 0:
            return
        new_points = []
        for id in points.point_inds:
            new_selected = [id] + trace.customdata[id]            
            new_points = new_points + new_selected
        trace.selectedpoints = new_points
        update_selected_nodes(trace.selectedpoints)
        
    def update_selected_nodes(selected_nodes):
        if len(selected_nodes) == 0:
            return

        # write selected domains to the output widget
        selected_domains = [domains[id].name for id in selected_nodes]
        with out:
            print("Selected Domains:\n")
            for selected_domain in selected_domains:
                print(selected_domain)
        out.clear_output(wait=True)
        
        # calc pivots selected domains have in common
        get_2d_shared_pivots(domain_list, selected_domains)
        
    # event handler for node selection
    fig.data[1].on_selection(node_selection_fn)        
    # event handle for node click
    fig.data[1].on_click(node_click_fn)

    # Create a table FigureWidget that updates the list of selected domains
    cols = [
        [domain.name for domain in domain_list.values()],
        [domain.risk_score for domain in domain_list.values()]
    ]

    out = widgets.Output(layout={'border': '1px solid black'})

    domain_ui = widgets.VBox((fig, out))
    return domain_ui
    

def get_shared_pivots(domain_list: dict, selected_domains: list):
    shared_pivots = {}
    for id in selected_domains:
        domain = domain_list[id]
        for cat in domain.pivot_categories:
            for cat_value in domain.pivot_categories[cat]:
                key = f"{cat}: {cat_value}"
                if key not in shared_pivots:
                    shared_pivots[key] = []
                shared_pivots[key].append(domain)

    # filter by pivots that have >= n domains
    shared_pivots = {k: v for k, v in shared_pivots.items() if len(v) >= 3}
    return shared_pivots
    
    
def create_pivot_heatmaps(shared_pivots: dict):
    print("\n Heatmap of which pivots connect the most domains together: by pivot category")
    pivot_cat_crosstab, pivot_value_crosstab = create_pivot_tables(shared_pivots)
    fig, ax = plt.subplots(figsize=(10, 10))
    im = heatmap(
        pivot_cat_crosstab,
        pivot_cat_crosstab.index,
        pivot_cat_crosstab.columns,
        ax=ax,
        cmap="Blues")
    texts = annotate_heatmap(im, valfmt="{x}")
    fig.tight_layout()
    plt.show()

    print("\n Heatmap of which pivots connect the most domains together: by pivot value")
    fig, ax = plt.subplots(figsize=(10, 10))
    im = heatmap(
        pivot_value_crosstab,
        pivot_value_crosstab.index,
        pivot_value_crosstab.columns,
        ax=ax,
        cmap="Blues")
    texts = annotate_heatmap(im, valfmt="{x}")
    fig.tight_layout()
    plt.show()

    print("\n List of the most frequent pivot values")
    create_pivot_summary(pivot_value_crosstab)
    
    
def create_pivot_tables(shared_pivots: dict):
    # Create the pandas DataFrame 
    data = []
    for pivot_value in shared_pivots:
        for d in shared_pivots[pivot_value]:
            pivot_cat = pivot_value.split(": ")[0]
            data.append([d.name, pivot_cat, pivot_value])
    df = pd.DataFrame(data, columns = ['domain', 'pivot_cat', 'pivot']) 

    # Build contingency table of domains to pivot
    pivot_cat_crosstab = pd.crosstab(df['pivot_cat'], df['domain'])
    pivot_value_crosstab = pd.crosstab(df['pivot'], df['domain'])

    # sort rows by total # of pivots
    pivot_cat_crosstab['sum'] = pivot_cat_crosstab[list(pivot_cat_crosstab.columns)].sum(axis=1)
    pivot_cat_crosstab.sort_values("sum", 0, ascending=False, inplace=True)
    pivot_cat_crosstab.drop("sum", 1, inplace=True)

    # sort rows by total # of pivots
    pivot_value_crosstab['sum'] = pivot_value_crosstab[list(pivot_value_crosstab.columns)].sum(axis=1)
    pivot_value_crosstab.sort_values("sum", 0, ascending=False, inplace=True)
    pivot_value_crosstab.drop("sum", 1, inplace=True)

    return pivot_cat_crosstab, pivot_value_crosstab


def create_pivot_summary(pivot_value_crosstab: "Pandas_CrossTab"):
    # show just an output view of pivot name and count for selection
    summary = pivot_value_crosstab.copy()
    summary['count'] = summary[list(summary.columns)].sum(axis=1)
    summary.sort_values("count", 0, ascending=False, inplace=True)
    summary = summary[["count"]]

    headers = ["Pivot Category", "Pivot Values", "Count"]
    table = []
    for index, row in summary.iterrows():
        cat, pivot = index.split(": ")
        table.append([cat, pivot, row["count"]])
    display(HTML(tabulate.tabulate(table, headers=headers, tablefmt='html')))



def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts