#!/usr/bin/env bash

program1_dir=$1
program2_dir=$2

# loop through program1_dir
agreement_ct=0
tests=0
agreements=""
disagreements=""
for program1 in $program1_dir/*py; do
    program2=${program2_dir}/$(basename $program1)
    diffval=$(diff -w $program1 $program2)
    if [ -n "$VERBOSE" ]; then
	echo "diff: $diffval"
    fi
    if [[ "$diffval" == "" ]]; then
        ((agreement_ct++))
        agreements="$agreements${program1}\n"
    else
      disagreements="$disagreements${program1}\n"
    fi
    ((tests++))
done

echo "================================"
echo "Output Agreements: $agreement_ct/$tests"
printf "$agreements\n"

printf "================================\n"
echo "Disagreements:"
printf "$disagreements\n"

