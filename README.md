# Hindsight Logging Across Space and Time

## Run diff-match-patch
Here's an example command that should Just Work™
```
python diff-patch-match.py --in-dir tests/toy-simple --log-version 2 --out-dir tests/toy-simple-diff-patch-match --log=DEBUG
```


## Testing
Pass in two different directories with the same filenames, and the test harness will compare their log outputs:
```
VERBOSE=1 ./test-harness.sh tests/toy-simple/v1-gt tests/toy-simple-diff-patch-match/v1
```

(you can turn off VERBOSE if you want less output)

## GumTree Installation (old: this gumtree apparently does not work for python)

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
