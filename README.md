# Hindsight Logging Across Space and Time



## Installation

```
# setup a conda environment
conda create -n 262a python=3.7 ipython
conda activate 262a

# install pythonparser for gumtree
git clone git@github.com:GumTreeDiff/pythonparser.git
cd pythonparser
pip install -r requirements.txt
ln -f -s $PWD/pythonparser  SOMEWHERE_IN_YOUR_PATH


# install java (you may have a different setup on your system)
conda install -c anaconda openjdk 

# get gumtree binary
wget https://github.com/GumTreeDiff/gumtree/releases/download/v2.1.2/gumtree.zip
unzip gumtree.zip
ln -s $PWD/gumtree-2.1.2/bin/gumtree SOMEWHERE_IN_YOUR_PATH

# test
gumtree diff file1 file2
```
