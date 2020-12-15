import string 
import argparse
import random
import logging as log
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Diff patch match based log propagator')
    parser.add_argument(
        '--out-dir',
        required=True)
    parser.add_argument(
        '--template',
        required=True)
    parser.add_argument(
        '--num-versions',
        type=int,
        default=2)
    parser.add_argument(
        '--noise-prob',
        type=float,
        default=0.5)
    parser.add_argument(
        '--no-rename',
        default=False,
        type=bool)
    parser.add_argument(
        '--log',
        default="INFO")

    args = parser.parse_args()
    log.basicConfig(level=getattr(log, args.log.upper(), None))
    return args


RENAME_KEY="#RENAME"
PERMUTE_KEY="#PERMUTE"
NOISE_KEY="#ADD-NOISE"
END_PERMUTE_KEY="#END-PERMUTE"
END_NOISE_KEY="#END-ADD-NOISE"
LOG_STR_INDICATOR = "HLAST TEST LOG"
def main(args):
    log.debug("Parsing template {}".format(args.template))
    with open(args.template, 'r') as f:
        template_lines = f.readlines()
    
    last_version = template_lines.copy()

    # the first version is the template
    outfname, gt_outfname = make_version_dir_return_outfiles(args, 1)
    with open(gt_outfname, 'w') as outf:
        for l in template_lines:
            outf.write("%s\n" % l.rstrip())
    with open(outfname, 'w') as outf:
        for l in template_lines:
            if LOG_STR_INDICATOR not in l:
                outf.write("%s\n" % l.rstrip())

    # Now do all subsequent versions
    # use name map to keep track of renames
    var_name_map = {}

    for vnum in range(args.num_versions - 1):
        
        # if last version -- make the "gt" the log dir
        if vnum == args.num_versions - 2:
            outfname, gt_outfname = make_version_dir_return_outfiles(args, vnum + 2, "log")
        else:
            outfname, gt_outfname = make_version_dir_return_outfiles(args, vnum + 2)
            
        this_version = []
        test_version = []
        add_noise = False

        # do a pass to perform the permutations
        perm_ranges=[]
        current_set=[0]*2
        for i, line in enumerate(last_version):
            keyline = line.strip()
            if keyline[:len(PERMUTE_KEY)] == PERMUTE_KEY:
                log.debug("permute key activated for {}".format(line))
                current_set[0] = i + 1
            elif keyline[:len(END_PERMUTE_KEY)] == END_PERMUTE_KEY:
                log.debug("end permute activated for {}".format(line))
                current_set[1] = i
                perm_ranges.append(current_set)
                current_set = [0] * 2

        copied_version = last_version.copy()
        for perm_set in perm_ranges:
            porder = list(range(*perm_set))
            random.shuffle(porder)
            for i, ri in zip(range(*perm_set), porder):
                last_version[i] = copied_version[ri]

        # do a pass to perform renames and noise
        for i, line in enumerate(last_version):
            line = line.rstrip()
            keyline = line.strip()
            if keyline[:len(RENAME_KEY)] == RENAME_KEY and not args.no_rename:
                log.debug("rename key activated for {}".format(line))
                # for all of the following lines, rename the provided keys randomly
                rename_vars = line[len(RENAME_KEY):].strip().split()
                for rv in rename_vars:
                    current_name = var_name_map.get(rv, rv)
                    new_name = rand_var()
                    var_name_map[rv] = new_name
                    for j, forward_line in enumerate(last_version[i+1:]):
                        last_version[i + 1 + j] = forward_line.replace(current_name, new_name)

            elif keyline[:len(NOISE_KEY)] == NOISE_KEY:
                log.debug("noise key activated for {}".format(line))
                add_noise = True

            elif keyline[:len(END_NOISE_KEY)] == END_NOISE_KEY:
                log.debug("end noise activated for {}".format(line))
                add_noise = False

            this_version.append(line)
            if LOG_STR_INDICATOR not in line:
                test_version.append(line)

            if add_noise and args.noise_prob > random.random() and len(keyline) > 0 and keyline[-1] != ":":
                indent_len = len(line) - len(line.lstrip())
                noise_line = line[:indent_len] + "{} = {} # rand noise".format(rand_var(), rand_assign())
                this_version.append(noise_line)
                test_version.append(noise_line)

        # Write the version to file and gt file
        with open(gt_outfname, 'w') as outf:
            for l in this_version:
                outf.write("%s\n" % l)
        with open(outfname, 'w') as outf:
            for l in test_version:
                outf.write("%s\n" % l)
        last_version = this_version

#
# HELPERS
#
def make_version_dir_return_outfiles(args, vnum, gt_str="gt"):
    # TODO could make this a little function...
    vdir = os.path.join(args.out_dir, "v{}".format(vnum))
    vdir_gt = os.path.join(args.out_dir, "v{}-{}".format(vnum, gt_str))
    os.makedirs(vdir, exist_ok=True)
    os.makedirs(vdir_gt, exist_ok=True)
    out_fname = os.path.basename(args.template)
    outfile = os.path.join(vdir, out_fname)
    gt_outfile = os.path.join(vdir_gt, out_fname)
    return outfile, gt_outfile

# https://pynative.com/python-generate-random-string/
def rand_var(length=6):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return "rn_{}".format(result_str)

# lazy, but doesn't matter
def rand_assign():
    return '"{}"'.format(rand_var())

if __name__ == "__main__":
    main(parse_args())