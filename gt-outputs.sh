#!/bin/bash

bash gt-propagate.sh toy-simple '' '' --gumtree '{"min_height":1,"min_dice":.3}'

bash gt-propagate.sh real-fast-neural-style

bash gt-propagate.sh real-imagenet '' '' --minor 6

bash gt-propagate.sh real-reinforcement-learning

bash gt-propagate.sh real-fairseq

bash gt-propagate.sh autogen-versions