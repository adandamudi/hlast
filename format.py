#!/usr/bin/env python

from argparse import ArgumentParser, FileType
from ast import parse, unparse
import sys

parser = ArgumentParser()
parser.add_argument('file', type=FileType('r+'))
parser.add_argument('--minor', type=int, default=sys.version_info[1])
(args, _) = parser.parse_known_args()

tree = parse(args.file.read(), feature_version=(3, args.minor))
formatted = unparse(tree)

with args.file as f:
    f.seek(0)
    print(unparse(tree), file=f)
    f.truncate()
