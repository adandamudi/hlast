#!/bin/bash

format(){ python -c "import sys, ast; print(ast.unparse(ast.parse(sys.stdin.read())))"; }

for dir in tests/*; do
    suite="${dir##*/}"

    #FIXME: Store replicable results in a separate directory
    if [[ "$suite" == *-diff-patch-match || "$suite" == *-gumtree ]]; then continue; fi

    if [[ -n $1 && $suite != $1 ]]; then continue; fi

    for file in tests/$suite/v*-log/*.py; do
        test="${file##*/}"; test="${test%.py}"
        v="${file#*/v}"; v="${v%-log/*}"

        if [[ -n $2 && $test != $2 ]]; then continue; fi

        # Variables:
        #   target ---> base
        #    |           |
        #    '---> gt    '---> log

        base="tests/$suite/v$v/$test.py"
        log="$file"; gt="$log"

        while [[ $v > 0 ]]; do
            target="tests/$suite/v$v/$test.py"
            result="tests/$suite-gumtree/v$v/$test.py"

            if [[ -z $3 || $v == $3 ]]; then
                mkdir -p "${result%/*}"; rm -rf "$result"

                # Modify to use a different log propagation
                lineno=$(diff $log $base | grep -B1 ^\< | egrep -o ^\\d+)
                echo "[$suite/$test/$v] propagate.py lineno=$lineno"
                python propagate.py $lineno "$log" "$target" "$result"

                if [[ -f "$result" ]]; then
                    echo "[$suite/$test/$v] diff -q result ground-truth"
                    diff -q "$result" <(format < "$gt") || sdiff "$result" <(format < "$gt")
                fi
            fi

            v=$((v-1))
            base="$target"
            log="$gt"

            gt="tests/$suite/v$v-gt/$test.py"
        done
    done
done