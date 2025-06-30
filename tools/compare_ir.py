import difflib
def diff_strings_colored(lines1, lines2):
    # Split the strings into lines for comparison
    #lines1 = str1.splitlines()
    #lines2 = str2.splitlines()

    # Create a Differ object
    differ = difflib.Differ()

    # Compute the difference between the two lists of lines
    diff = differ.compare(lines1, lines2)

    # Print the differences with colors
    for line in diff:
        #if line.startswith('+'):
        #    print(f"\033[32m{line}\033[0m")  # Green for additions
        #elif line.startswith('-'):
        #    print(f"\033[31m{line}\033[0m")  # Red for deletions
        #elif line.startswith('?'):
        #    print(f"\033[34m{line}\033[0m")  # Blue for change indicators
        #else:
        print(line)

def split_lines_by_sep(filename, s, sep_start=None, sep_end=None):
    #lines = s.split('\n')
    lines = s.splitlines()
    blocks = []
    if sep_start is None:
        return [lines]
    new_block = []
    within_new_block = False
    cur_block_key = ''
    for lineno, line in enumerate(lines):
        if sep_start in line:
            if within_new_block:
                blocks.append((cur_block_key, new_block))
                new_block = []
            within_new_block = True
            cur_block_key = line
        elif sep_end is not None and sep_end in line:
            blocks.append((cur_block_key, new_block))
            new_block = []
            within_new_block = False
        elif within_new_block:
            new_block.append(line)
        else:
            #print(f"<{filename}:{lineno}>: {line}")
            pass #TODO: print other lines also?
    if within_new_block:
        blocks.append((cur_block_key, new_block))

    return blocks

def diff_within_file(filename1, str1, sep_start=None, sep_end=None):
    key_block_pairs = split_lines_by_sep(filename1, str1, sep_start, sep_end)
    assert len(key_block_pairs) >= 1
    prev_key, prev_block = key_block_pairs[0]
    for key, block in key_block_pairs[1:]:
        print(f"{prev_key} --> {key}")
        diff_strings_colored(prev_block, block)
        prev_key, prev_block = key, block

def diff_across_files(filename1, filename2, str1, str2, sep_start=None, sep_end=None):
    key_block_pairs1 = split_lines_by_sep(filename1, str1, sep_start, sep_end)
    key_block_pairs2 = split_lines_by_sep(filename2, str2, sep_start, sep_end)

    key_block_map2 = {key: block for key, block in key_block_pairs2}

    #diff_pair = []

    num1 = len(key_block_pairs1)

    cnt1 = 0
    while cnt1 < num1:
        key1, block1 = key_block_pairs1[cnt1]
        if key1 in key_block_map2:
            print(f"{key1}")
            block2 = key_block_map2[key1]
            #diff_pair.append((ke1, block1, block2))
            diff_strings_colored(block1, block2)
        cnt1 += 1

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f1', required=True)
    parser.add_argument('-f2', required=False, default=None, type=str)
    parser.add_argument('-s1', required=False, default=None, type=str)
    parser.add_argument('-s2', required=False, default=None, type=str)

    args = parser.parse_args()

    with open(args.f1) as f:
        str1 = f.read()
    if args.f2 is not None:
        with open(args.f2) as f:
            str2 = f.read()
        diff_across_files(args.f1, args.f2, str1, str2, args.s1, args.s2)
    else:
        diff_within_file(args.f1, str1, args.s1, args.s2)

