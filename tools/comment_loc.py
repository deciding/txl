import os
import sys

#filename = 'dump/bn128/consumer2buffer1stage2/_attn_fwd_tma.ttgir'
filename = sys.argv[1]

fileprefix, ext = os.path.splitext(filename)

old_filename = f"{fileprefix}{ext}"

new_filename = f"{fileprefix}_loc{ext}"

with open(filename) as f:
    lines = f.readlines()
with open(old_filename, 'w') as f:
    for line in lines:
        f.write(line)

import re

def parse_loc_case1(loc_str):
    pattern = r'loc\("(.+?)":(\d+):(\d+)\)'
    match = re.fullmatch(pattern, loc_str)
    if match:
        return {
            'type': 'known',
            'file': match.group(1),
            'line': int(match.group(2)),
            'column': int(match.group(3))
        }
    return None

def parse_loc_case2(loc_str):
    if loc_str == 'loc(unknown)':
        return {'type': 'unknown'}
    return None

def parse_loc_case3(loc_str):
    pattern = r'loc\(callsite\(#loc(\d+) at #loc(\d+)\)\)'
    match = re.fullmatch(pattern, loc_str)
    if match:
        return {
            'type': 'callsite',
            'loc1': '#loc'+match.group(1).strip(),
            'loc2': '#loc'+match.group(2).strip()
        }
    return None

def parse_loc(loc_str):
    # Remove any whitespace
    loc_str = loc_str.strip()

    # Try each case in order
    result = parse_loc_case2(loc_str)
    if result:
        return result

    result = parse_loc_case1(loc_str)
    if result:
        return result

    result = parse_loc_case3(loc_str)
    if result:
        return result

    return {'type': 'unparsed'}

loc_map = {}
loc_callsite_map = {}
for line in lines:
    if line.startswith("#loc"):
        locl, locr = line.strip().split('=')
        locl = locl.strip()

        locr = parse_loc(locr)
        if locr['type'] == 'known':
            loc_map[locl] = locr
        elif locr['type'] == 'callsite':
            loc_callsite_map[locl] = locr

for locl, locr in loc_callsite_map.items():
    loc1, loc2 = locr['loc1'], locr['loc2']
    while loc1 in loc_callsite_map:
        locr = loc_callsite_map[loc1]
        loc1, loc2 = locr['loc1'], locr['loc2']
    if loc1 in loc_map:
        loc_map[locl] = loc_map[loc1]

def parse_line_with_loc(line):
    # Pattern to match lines ending with loc(#locXXX)
    pattern = r'loc\(#loc(\d+)\)$'
    match = re.search(pattern, line.strip())

    if match:
        return '#loc' + match.group(1).strip()
    return None

def count_prefix_spaces(s):
    return len(s) - len(s.lstrip(' '))

def insert_char(s, char, n):
    return s[:n] + char + s[n:]

file_to_lines = {}
new_lines = []
# Start to add comment
for line in lines:
    #line = line.strip()
    loc = parse_line_with_loc(line)
    if loc and loc in loc_map:
        locr = loc_map[loc]
        assert locr['type'] == 'known'
        f1, lno1, cno1 = locr['file'], locr['line'], locr['column']
        if not f1 in file_to_lines:
            with open(f1) as f:
                lines1 = f.readlines()
            file_to_lines[f1] = lines1
        else:
            lines1 = file_to_lines[f1]
        l1 = lines1[int(lno1)-1]
        l1 = insert_char(l1, '|', cno1)
        l1 = l1.strip()
        l1 = ' ' * count_prefix_spaces(line) + '// ' + l1
        new_lines.append(l1)

    new_lines.append(line)

print('\n'.join(new_lines))

with open(new_filename, 'w') as f:
    for line in new_lines:
        f.write(line+'\n')
