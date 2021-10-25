
## DomainCAT to do Tasks
- refactor the code to just use the graph data structure as much as possible and less of the domain_list map
- figure out how to make the create_date pivot a window over n days vs just 1 day
- prune connections that are below some weight threshold
- refactor append_values_with_count(s) functions to share logic
- figure out a better way to normalize registrars
- create a way to type a domain name select that domain
- create a way to type a pivot (category or value?) and select all domains that are connected
- add every pivot possible. I mostly skipped the whois pivots because they aren't that useful anymore
- address the comment in DomainRelationship.add. Essentially domains that share 2 or more IP addresses could potentially have their edge strength artificially boosted
- maybe play around with normalizing edge weights once the graph is created, but before rendering

## Bugs to Fix

## Wish List

when looking at domains that are probably realted and created over a short period of time, it would be useful to have some viz that shows / groups the pivots per create date. That way you could see stuff like on day 1 TLD1 and regustrar1 were used, then day two TLD1 and registrar2 were used, then day 3 TLD2 and registrar2 were used. That kind of thing

Given a selection of domains, show what attributes they are NOT connected on

date range of domains
timeline view that shows how tight or loosely connected the domains are for each day or week

auto identify the clusters and show the pivot table for each cluster

auto discovery substrings