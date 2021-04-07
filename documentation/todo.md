
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