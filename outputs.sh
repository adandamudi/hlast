#!/bin/bash

bash propagate.sh dmp '' '' '' --log WARN

bash propagate.sh gt toy-simple '' '' --gumtree '{"min_height":1,"min_dice":.3}'
bash propagate.sh gt autogen-versions
bash propagate.sh gt real-fast-neural-style
bash propagate.sh gt real-imagenet '' '' --minor 6
bash propagate.sh gt real-reinforcement-learning
bash propagate.sh gt real-fairseq