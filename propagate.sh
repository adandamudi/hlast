#!/bin/bash

selected_suite=$1; selected_test=$2; selected_version=$3
extra_args=( "${@:4}" )

format(){ cat $1; }

for dir in tests/*; do
    suite="${dir##*/}"

    #FIXME: Store replicable results in a separate directory
    if [[ "$suite" == *-diff-patch-match || "$suite" == *-gumtree ]]; then continue; fi

    if [[ -n "$selected_suite" && "$suite" != "$selected_suite" ]]; then continue; fi

    for file in tests/$suite/v*-log/*.py; do
        test="${file##*/}"; test="${test%.py}"
        v="${file#*/v}"; v="${v%-log/*}"

        if [[ -n "$selected_test" && "$test" != "$selected_test" ]]; then continue; fi

        # Variables:
        #   target ---> base
        #    |           |
        #    '---> gt    '---> log

        base="tests/$suite/v$v/$test.py"
        log="$file"; gt="$log"

        while [[ $v > 0 ]]; do
            target="tests/$suite/v$v/$test.py"
            result="tests/$suite-gumtree/v$v/$test.py"

            if [[ -z "$selected_version" || $v == "$selected_version" ]]; then
                mkdir -p "${result%/*}"; rm -rf "$result";

                # vvvvv Modify to use a different log propagation
                lineno=$(diff $log $base | grep -B1 ^\< | egrep -o ^\\d+)
                if [[ -z "$lineno" ]]; then continue; fi

                echo "[$suite/$test/$v] propagate.py lineno=$lineno"
                ./propagate.py "${extra_args[@]}" "$lineno" "$log" "$target" --out "$result"

                format(){ ./format.py "$1" "${extra_args[@]}"; }
                # ^^^^^

                if [[ -f "$result" ]]; then
                    echo "[$suite/$test/$v] diff -q result ground-truth"
                    diff -q "$result" <(format "$gt") || sdiff "$result" <(format "$gt")
                fi
            fi

            v=$((v-1))
            base="$target"
            log="$gt"

            gt="tests/$suite/v$v-gt/$test.py"
        done
    done
done