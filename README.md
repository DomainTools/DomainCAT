# DomainCAT (Domain Connectivity Analysis Tool)

## "See Connections Between Domains Right Meow"

**The Domain Connectivity Analysis Tool is used to analyze aggregate connectivity patterns across a set of domains during security investigations**

This project was a collaborative effort between [myself](https://www.linkedin.com/in/jconwell/) and [Matthew Pahl](https://www.linkedin.com/in/security-sme/)

## Introduction

When analyzing pivots during threat hunting, most people approach it from the perspective of “what can a single 
pivot tell you?” But often actors will set their domains up to use commodity hosting infrastructure, so the number of 
entities associated with a given pivot are so big they don’t really give you any useful information. 

This is where DomainCAT can help. Actors make decisions around domain registration and hosting options when setting 
up their malicious infrastructure. These can be considered behavioral choices.
- What registrar(s) do they use?
- What TLDs do they prefer?
- What hosting provider(s) do they like?
- What TLS cert authority do they use?

All of these decisions, together, makeup part of that actor’s infrastructure tools, tactics and procedures (TTPs), 
and we can analyze them as a whole to look for patterns across a set of domains. 

DomainCAT is a tool written in Jupyter Notebooks, a web-based interactive environment that lets you combine text, 
code, data, and interactive visualizations into your threat hunting toolbelt. The tool analyzes aggregate 
connectivity patterns across a set of domains looking at every pivot for every domain, asking; what are the shared 
pivots across these domains, how many shared pivots between each domain, do they have a small pivot count or a really 
large one? All of these aspects are taken into consideration as it builds out a connectivity graph that models how 
connected all the domains in an Iris search are to each other.

### Example Visualizations:

#### 3D visualization of domain to domain connections based on shared infrastructure, registration and naming patterns
![SegmentLocal](images/intro_3d.gif "segment")

#### 2D visualization of domain to domain connection
![domain_graph2d.png](images/2d_zoom.gif "segment")

## DomainCat Tutorial

#### Click here for the [DomainCAT Tutorial](documentation/tutorial.md) documentation

## Installation Steps: Docker (recommended)

_Note: building the container takes a bit of RAM to compile the resources for the jupyterlab-plotly extension. Bump up your RAM in Docker preferences to around 4Gb while building the container. Then afterwards you can drop it back down to your normal level to run the container_

### Steps:

Clone the git repository locally

`$ git clone https://github.com/DomainTools/DomainCAT.git` 

Change directory to the domaincat folder

`$ cd domaincat`

Build the jupyter notebook container

`$ docker build --tag domaincat .`

Run the jupyter notebook

`$ docker run -p 9999:9999 --name domaincat domaincat`

## Installation Steps: Manual (cross your fingers)

_Note: this project uses JupyterLab Widgets, which requires nodejs >= 12.0.0 to be installed...which is on you_

### Steps:

Clone the git repository locally

`$ git clone https://github.com/DomainTools/DomainCAT.git` 

Change directory to the domaincat folder

`$ cd domaincat`

Install python libraries

`$ pip install -r requirements.txt`

JupyterLab widgets extension

```
$ jupyter labextension install jupyterlab-plotly@4.14.3 --no-build
$ jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build
$ jupyter labextension install plotlywidget@4.14.3 --no-build
$ jupyter lab build
```

Run the jupyter notebook

`$ jupyter lab`

___

# Release Notes:

August 24, 2021:
- Adding a way to remove domains in the graph that you aren't interested in (look at the bottom of the notebook)
- Refactor of the backend data structures to be a bit more efficient

April 27, 2021:
- Added support for `dotenv` to store REST API credentials in a `.env` file
- Added logic to support
  - comma delimited list of domains
  - domains defanged with square brackets

April 23, 2021:
- Added config flag to only analyze active domains
- Show count of selected domains

April 19: 2021:
- Bug fix to not normalize risk scores values when calculating node color
- Mo'better sorting of selected domains

April 15, 2021: 
- Bug fix: wrong json element returned when querying search hash

April 14, 2021: 
- Added UI to search either a list of domain names or an Iris search hash
- Added UI to enter Iris REST API username and password 

April 7, 2021: 
- Initial commit

___

_Plotly Bug: in the 2D visualization of the domain graph there is a weird bug in `Plotly Visualization library` where 
if your cursor is directly over the center of a node, the node's tool tip with the domain's name will disappear and 
if you click the node, it unselects all nodes. So only click on a node if you see it's tool tip_
