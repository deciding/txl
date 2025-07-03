import os
import re
from collections import deque

# python tools/trace_include_headers.py thirdparty/triton/python/src/ir.cc triton/Dialect/TritonGPU/IR/Dialect.h -I/ssd2/zhangzn/txl/thirdparty/triton/python/build/cmake.linux-x86_64-cpython-3.12 -I/ssd2/zhangzn/txl/thirdparty/triton -I/ssd2/zhangzn/txl/thirdparty/triton/include -I/ssd2/zhangzn/txl/thirdparty/triton/. -I/home/zhangzn/.triton/llvm/llvm-a66376b0-ubuntu-x64/include -I/ssd2/zhangzn/txl/thirdparty/triton/python/build/cmake.linux-x86_64-cpython-3.12/include -I/ssd2/zhangzn/txl/thirdparty/triton/third_party -I/ssd2/zhangzn/txl/thirdparty/triton/python/build/cmake.linux-x86_64-cpython-3.12/third_party -I/ssd2/zhangzn/txl/thirdparty/triton/python/src -I/ssd2/zhangzn/txl/thirdparty/triton/third_party/amd/lib/TritonAMDGPUTransforms/../../include -I/ssd2/zhangzn/txl/thirdparty/triton/python/build/cmake.linux-x86_64-cpython-3.12/third_party/amd/lib/TritonAMDGPUTransforms/../../include

# dest_header must be in include format
# include_paths in abs paths
def find_include_paths(src_file, include_paths, dest_header):
    # Normalize paths
    dest_header = os.path.normpath(dest_header)
    abs_header = ''
    
    # Build a dictionary to map headers to their full paths
    header_arr = []
    for include_path in include_paths:
        include_path = os.path.normpath(include_path)
        for root, _, files in os.walk(include_path):
            for file in files:
                if file.endswith(('.h', '.hpp')):
                    header = os.path.normpath(os.path.join(root, file))
                    if header.endswith(dest_header):
                        abs_header = header
                    header_arr.append(header)
    
    # Check if destination header exists
    if not abs_header:
        print(f"Error: Destination header {dest_header} not found in include paths")
        return []
    
    # Read includes from a file
    def get_includes(file_path):
        # this file path is in include format, need to first find it in include_paths (header_arr)
        abs_file_path = ''
        for header in header_arr:
            if header.endswith(file_path):
                abs_file_path = header

        if current_file != src_file and not abs_file_path:
            #import pdb;pdb.set_trace()
            return []
        if current_file == src_file:
            abs_file_path = src_file

        includes = []
        try:
            with open(abs_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                # Match #include statements
                matches = re.findall(r'#include\s+[<"](.+?)[>"]', content)
                for match in matches:
                    abs_match_path = ''
                    for header in header_arr:
                        if header.endswith(match):
                            abs_match_path = header
                        includes.append(match)
                        break
            print(file_path)
            print(includes)
            #import pdb;pdb.set_trace()
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {str(e)}")
        return includes
    
    # BFS to find all paths
    all_paths = []
    queue = deque()
    queue.append([src_file])
    
    visited = set()
    max_depth = 20  # Prevent infinite loops
    
    while queue:
        path = queue.popleft()
        current_file = path[-1]
        
        if len(path) > max_depth:
            continue
        
        # all in include format
        # finish
        if os.path.normpath(current_file) == os.path.normpath(dest_header):
            all_paths.append(path)
            continue
        
        if current_file in visited:
            continue
        
        visited.add(current_file)
        
        # Only process header files (not .cpp/.cc files in the middle)
        if current_file != src_file and not current_file.endswith(('.h', '.hpp')):
            continue
        
        includes = get_includes(current_file)
        for include in includes:
            if include not in path:  # Avoid cycles
                new_path = path.copy()
                new_path.append(include)
                queue.append(new_path)
    
    return all_paths

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Find include paths between source and destination header.')
    parser.add_argument('src_file', help='Source file path (e.g., ir.cc)')
    parser.add_argument('dest_header', help='Destination header path (e.g., Dialect.h)')
    parser.add_argument('-I', '--include', action='append', default=[], 
                       help='Include paths to search for headers')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.src_file):
        print(f"Error: Source file {args.src_file} does not exist")
        return
    
    include_paths = args.include
    if not include_paths:
        print("Warning: No include paths specified")
    
    paths = find_include_paths(args.src_file, include_paths, args.dest_header)
    
    if not paths:
        print(f"No paths found from {args.src_file} to {args.dest_header}")
    else:
        print(f"Found {len(paths)} path(s) from {args.src_file} to {args.dest_header}:")
        for i, path in enumerate(paths, 1):
            print(f"\nPath {i}:")
            print(" -> ".join(path))

if __name__ == '__main__':
    main()
