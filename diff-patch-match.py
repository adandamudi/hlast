import diff_patch_match.python3 as dmp_module
import argparse
import sys
import os
import glob
import logging as log

def parse_args():
    parser = argparse.ArgumentParser(description='Diff patch match based log propagator')
    parser.add_argument(
        '--out-dir',
        required=True)
    parser.add_argument(
        '--in-dir',
        required=True)
    parser.add_argument(
        '--log-version',
        type=int,
        required=True)
    parser.add_argument(
        '--log',
        default="INFO")

    args = parser.parse_args()
    return args


def diff_lineMode(text1, text2):
    dmp = dmp_module.diff_match_patch()
    a = dmp.diff_linesToChars(text1, text2)
    lineText1 = a[0]
    lineText2 = a[1]
    lineArray = a[2]
    diffs = dmp.diff_main(lineText1, lineText2, False)
    dmp.diff_charsToLines(diffs, lineArray)
    return diffs

def dmp(args):
    log.basicConfig(level=getattr(log, args.log.upper(), None))
    os.makedirs(args.out_dir, exist_ok=True)

    starting_log_dir = os.path.join(args.in_dir, "v{}-log".format(args.log_version))
    starting_dir = os.path.join(args.in_dir, "v{}".format(args.log_version))
    log.debug("Starting directory: {}".format(starting_dir))
    log.debug("Starting log directory: {}".format(starting_log_dir))

    dmp = dmp_module.diff_match_patch()
    log.info("Starting at latest version: v{}".format(args.log_version))
    for logfile in glob.glob(os.path.join(starting_log_dir, "*.py")):
        infile = logfile.replace("-log", "")
        base_filename = os.path.basename(logfile)
        log.debug("Parsing version {}".format(infile))
        log.debug("Parsing log version {}".format(logfile))
        with open(infile, 'r') as f:
            start_version=f.read()
        with open(logfile, 'r') as f:
            start_log_version=f.read()
        log.debug("Starting version: \n{}".format(start_version))
        log.debug("Starting log version: \n{}".format(start_log_version))
        
        # diff = diff_lineMode(start_version, start_log_version)
        diff = dmp.diff_main(start_version, start_log_version)
        dmp.diff_cleanupSemantic(diff)
        # find the log entry and the line number
        linenum = 0
        for entry in diff:
            log.debug(entry)
            if entry[0] <= 0:
                linenum += entry[1].count('\n')
            elif 'print' in entry[1]:
                if entry[1][0] == '\n':
                    linenum += 1
                logline = entry[1].strip()
                break
        log.debug("log line should be inserted after line {}: {}".format(linenum, logline))
        log.debug(diff)

        # now for each previous version, figure out the log linenum (and indentation level)
        last_version = start_version
        v = args.log_version - 1
        while True:
            vdir = os.path.join(args.in_dir, "v{}".format(v))
            outdir = os.path.join(args.out_dir, "v{}".format(v))

            if not os.path.exists(vdir):
                log.debug("Last version processed: {}".format(v + 1))
                break
            os.makedirs(outdir, exist_ok=True)
            with open(os.path.join(vdir, base_filename)) as f:
                log.debug("Reading {}".format(f.name))
                update_version = f.read()
                log.debug("read the following file:\n{}".format(update_version))

            # always diff from the last version
            diff = dmp.diff_main(update_version, last_version)
            dmp.diff_cleanupSemantic(diff)
            log.debug("Computed diff: {}".format(diff))

            # the linenumber in the old version
            new_linenum = 0
            # the linumber in the new version
            lines_so_far = 0

            update_version_lines = update_version.split('\n')
            for entry in diff:
                num_newlines = entry[1].count('\n')

                # update the linecount for the previous version (no change 0 or removal -1)
                if entry[0] <= 0:
                    new_linenum += num_newlines
                # update the linecount for the last version (no change 0 or addition +1)
                if entry[0] >= 0:
                    lines_so_far += num_newlines

                # we've met or surpassed the linecount for the last version
                if lines_so_far >= linenum:
                    if lines_so_far > linenum and entry[0] == 0:
                        # if it's a common unchanged entry (change val 0), 
                        # and we're passed the desired line count
                        # then subtract the appropriate number of lines
                        # so the log shows up at the right spot
                        new_linenum = new_linenum - (lines_so_far - linenum)

                    log.debug("aligned line number in previous version: ".format(new_linenum))

                    #########################################################################
                    # find the appropriate indentation before and after the given line_number
                    #########################################################################
                    # first, search for an earlier line
                    prev_linenum = new_linenum - 1
                    prev_indent = ""
                    notfound = True
                    while prev_linenum > 0 and notfound:
                        lval = update_version_lines[prev_linenum]
                        if len(lval.strip()) == 0:
                            prev_linenum -= 1
                            continue
                        # we have a previous line with content
                        indent_len = len(lval) - len(lval.lstrip())
                        if indent_len:
                            prev_indent = lval[:indent_len]
                        notfound = False

                    notfound = True
                    next_linenum = new_linenum
                    next_indent = ""
                    while next_linenum < len(update_version_lines) and notfound:
                        lval = update_version_lines[next_linenum]
                        if len(lval.strip()) == 0:
                            next_linenum += 1
                            continue
                        # we have a next line with content
                        indent_len = len(lval) - len(lval.lstrip())
                        if indent_len:
                            next_indent = lval[:indent_len]
                        notfound = False

                    # next and prev indent:
                    indentation = prev_indent
                    if len(next_indent) > len(prev_indent):
                        indentation = next_indent

                    # write to file
                    update_version_lines.insert(new_linenum, indentation + logline)
                    with open(os.path.join(outdir, base_filename), "w") as f:
                        f.write("\n".join(update_version_lines))
                    
                    # prep for the next version
                    last_version = update_version
                    linenum = new_linenum
                    break



            dmp.diff_cleanupSemantic(diff)
            v -= 1

    log.info("Done. See results in {}".format(args.out_dir))

        


if __name__ == "__main__":
    dmp(parse_args())
