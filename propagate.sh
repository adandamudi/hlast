#!/bin/bash

usage() { echo "Usage: $0 (dmp|gt) [codebase] [version] [file]"; exit; }

selected_approach=$1 && [[ "$selected_approach" =~ (dmp|gt) ]] || usage
selected_codebase=$2 && [[ -z "$selected_codebase" || -d "tests/$selected_codebase" ]] || usage
selected_version=$3  && [[ -z "$selected_version" || -d "tests/$selected_codebase/v$selected_version" ]] || usage
selected_file=$4     && [[ -z "$selected_file" || -f "tests/$selected_codebase/$selected_version/$selected_file" ]] || usage
extra_args=( "${@:5}" )

approach="$selected_approach"

for dir in tests/*; do
codebase="$(basename "$dir")"
if [[ -z "$selected_codebase" || "$codebase" == "$selected_codebase" ]]; then
    last="$(echo tests/$codebase/v*-log)"
    version=$(basename "$last" | egrep -o \\d+)

    if [[ $approach == dmp ]]; then
        mkdir -p out/$approach; rm -rf out/$approach/$codebase

        echo "[$codebase/$version] dmp-propagate.py $codebase $version"
        python dmp-propagate.py "${extra_args[@]}" --in-dir "tests/$codebase" --log-version $version --out-dir "out/$approach/$codebase"

        if [[ -d "$result" ]]; then
            echo "[$codebase/$version] diff -q result ground-truth"
            diff -q "$result" "$ground_truth" || sdiff "$result" "$ground_truth"
        fi
    fi

    log="$last"
    nolog="${log%-log}"
    # Start with sanity check
    ground_truth="$log"

    while [[ $version > 0 ]]; do
    target="tests/$codebase/v$version"
    result="out/$approach/$codebase/v$version"
    if [[ -z "$selected_version" || $version == "$selected_version" ]]; then

        #   target ----------> nolog
        #     |                  |
        #     |-- ground_truth   '-- log
        #     |
        #     '-- *result*

        for path in $target/*; do
        file="$(basename "$path")"
        if diff "$log/$file" "$nolog/$file" > /dev/null; then
            echo "[$codebase/$version/$file] no log statement!"

        elif [[ -z "$selected_file" || "$file" == "$selected_file" ]]; then

            if [[ $approach == gt ]]; then
                mkdir -p $result; rm -f $result/$file

                lineno=$(diff $log/$file $nolog/$file | grep -B1 ^\< | egrep -o ^\\d+)

                echo "[$codebase/$version/$file] gt-propagate.py lineno=$lineno"
                ./gt-propagate.py "${extra_args[@]}" "$lineno" "$log/$file" "$target/$file" --out "$result/$file"


                if [[ -f "$result/$file" ]]; then
                    ./format.py "${extra_args[@]}" "$ground_truth/$file"
                    echo "[$codebase/$version/$file] diff -q result ground-truth"
                    diff -q "$result/$file" "$ground_truth/$file" || sdiff "$result/$file" "$ground_truth/$file"
                    git restore "$ground_truth/$file"  # undo formatting
                fi
            fi

        fi
        done
    fi
    # Move to next version
    log="$ground_truth"; nolog="$target"
    version=$((version-1)); ground_truth="tests/$codebase/v$version-gt"
    done
fi
done