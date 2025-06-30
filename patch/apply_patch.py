def apply_patch(patch_file, target_file):
    # Read the patch file
    with open(patch_file, 'r') as f:
        patch_lines = f.readlines()

    # Extract headers (between // HEADERS and // PATCH:)
    headers = []

    patch_dict = {}

    in_headers = False
    in_patch = False

    for line in patch_lines:
        if '// HEADERS' in line:
            in_headers = True
            continue
        elif '// PATCH:' in line:
            in_headers = False
            in_patch = True
            pybind_prefix = line.split('// PATCH:')[-1].strip()
            patch_dict[pybind_prefix] = []
            continue

        if in_headers:
            headers.append(line)
        elif in_patch:
            patch_dict[pybind_prefix].append(line)

    headers = ''.join(headers).strip()
    for k, v in patch_dict.items():
        patch_dict[k] = ''.join(patch_dict[k]).strip()

    # Read the target file
    with open(target_file, 'r') as f:
        target_lines = f.readlines()
    
    # 1. Add headers to first include block
    in_includes = False
    modified_lines = []
    headers_added = False
    
    for line in target_lines:
        modified_lines.append(line)
        
        if not headers_added and line.startswith('#include'):
            in_includes = True
        elif in_includes and not line.startswith('#include'):
            modified_lines.insert(-1, headers + '\n')
            headers_added = True
            in_includes = False
    
    # 2. Find the pybind definition and append patch
    target_content = ''.join(modified_lines)
    pybind_pos = target_content.find(pybind_prefix)
    
    if pybind_pos != -1:
        # Find the closing }); after the prefix
        close_pos = target_content.find('});', pybind_pos)
        
        if close_pos != -1:
            # Remove the });
            new_content = (
                target_content[:close_pos] + 
                '\n' + patch_content + '\n' + 
                target_content[close_pos:]
            )
            
            # Write the modified content back
            with open(target_file, 'w') as f:
                f.write(new_content)
            
            print(f"Successfully applied patch to {target_file}")
            return
    
    print("Could not find the target location in the file")

# Usage
apply_patch('patch.cpp', 'ir.cc')
