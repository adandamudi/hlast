#!/bin/bash

selected_codebase=$1; selected_filename=$2; selected_version=$3
extra_args=( "${@:4}" )

message(){ echo "[$codebase/$filename@v$version]" "$@"; }

for dir in tests/*; do
    codebase="${dir##*/}"

    if [[ -z "$selected_codebase" || "$codebase" == "$selected_codebase" ]]; then

        for path in tests/$codebase/v*-log/*.py; do
            filename="${path##*/}"
            
            version="${path#*/v}"
            version="${version%-log/*}"

            if [[ -z "$selected_filename" || "$filename" == "$selected_filename" ]]; then

                log="tests/$codebase/v$version-log/$filename"
                nolog="tests/$codebase/v$version/$filename"

                # Start with sanity check by propagating to nolog
                ground_truth="$log"

                while [[ $version > 0 ]]; do
                    target="tests/$codebase/v$version/$filename"
                    result="out/gt/$codebase/v$version/$filename"

                    if diff "$log" "$nolog" > /dev/null; then
                        message "no log statement!"

                    elif [[ -z "$selected_version" || $version == "$selected_version" ]]; then
                        mkdir -p "${result%/*}"
                        rm -rf "$result"

                        # vvvvv Modify to use a different log propagation
                        git restore "$log"
                        lineno=$(diff $log $nolog | grep -B1 ^\< | egrep -o ^\\d+)

                        message "propagate.py lineno=$lineno"
                        ./gt-propagate.py "${extra_args[@]}" "$lineno" "$log" "$target" --out "$result"

                        ./format.py "$ground_truth" "${extra_args[@]}"
                        # ^^^^^

                        if [[ -f "$result" ]]; then
                            message "diff -q result ground-truth"
                            diff -q "$result" "$ground_truth" || sdiff "$result" "$ground_truth"
                        fi

                    fi

                    log="$ground_truth"
                    nolog="$target"

                    version=$((version-1))
                    ground_truth="tests/$codebase/v$version-gt/$filename"
                done
            fi
        done
    fi
done