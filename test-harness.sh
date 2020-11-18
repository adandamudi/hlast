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
    output1=$(python $program1)
    result=$?
    if [ "$result" != "0" ]; then
        output2="PROGRAM FAILURE for first input (see stacktrace above)"
    fi
    output2=$(python $program2)
    result=$?
    if [ "$result" != "0" ]; then
        output2="PROGRAM FAILURE for second input (see stacktrace above)"
    fi
    
    if [ -n "$VERBOSE" ]; then
        echo "$program1 output: ${output1}"
        echo "$program2 output: ${output2}"
        echo
    fi

    if [[ "$output1" == "$output2" ]]; then
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

