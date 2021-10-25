FROM ubuntu:latest
# install the basics
RUN apt-get update && apt-get -y update
RUN apt-get install -y build-essential python3.6 python3-pip python3-dev 
RUN pip3 -q install pip --upgrade

# install nodejs v12
RUN apt-get install -y curl dirmngr apt-transport-https lsb-release ca-certificates
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash -
RUN apt-get install -y nodejs
RUN apt-get install -y gcc g++ make
RUN node --version
RUN npm --version

# copy dependency files
RUN mkdir src
WORKDIR src/
COPY requirements.txt .

# instsall Jupyter, domaincat requirements, and widget extensions
RUN pip3 install -r requirements.txt
RUN export NODE_OPTIONS=--max-old-space-size=4096
RUN jupyter labextension install jupyterlab-plotly@4.14.3 --no-build
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build
RUN jupyter labextension install plotlywidget@4.14.3 --no-build
RUN jupyter lab build
RUN npm cache clean --force
RUN unset NODE_OPTIONS

# Rest of Files copied
COPY . .

# Run jupyter lab
CMD ["jupyter", "lab", "--port=9999", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
