#!/usr/bin/env python3
"""
Builds an HTML file to compare TTIR/TTGIR/PTX and Python source with interactive line binding.
"""

import re
import sys
from pathlib import Path


def detect_file_type(filepath):
    """Detect file type by extension"""
    ext = filepath.lower().split(".")[-1]
    if ext in ("ttir", "ttgir", "ptx"):
        return ext
    return "unknown"


def parse_ttir_locations(ttir_path):
    """
    Parse TTIR/TTGIR file to extract:
    - Line number to location mapping (for IR lines)
    - Location variable definitions (#locN -> (file, line, col))
    """
    with open(ttir_path, "r") as f:
        lines = f.readlines()

    loc_defs = {}
    ir_line_to_loc = {}

    for i, line in enumerate(lines):
        line_num = i + 1

        # Match location definitions:
        # #loc = loc("/path/to/file":line:col)
        match = re.match(r'^(#loc\d*)\s*=\s*loc\("([^"]+)":(\d+):(\d+)\)', line)
        if match:
            loc_var, filepath, line, col = match.groups()
            loc_defs[loc_var] = (filepath, int(line), int(col))
            continue

        # Match: #loc19 = loc("x_ptr"(#loc))
        match = re.match(r'^(#loc\d+)\s*=\s*loc\("([^"]+)"\(#loc\)\)', line)
        if match:
            loc_var, varname = match.groups()
            # Reference to #loc, which is the true location
            if "#loc" in loc_defs:
                loc_defs[loc_var] = loc_defs["#loc"]
            continue

        # Match: #loc23 = loc("bid"(#loc2))
        match = re.match(r'^(#loc\d+)\s*=\s*loc\("([^"]+)"\(#loc\d+\)\)', line)
        if match:
            loc_var, varname = match.groups()
            # Need to find the referenced loc
            ref_match = re.search(r"\(#loc(\d+)\)", line)
            if ref_match:
                ref_loc = f"#loc{ref_match.group(1)}"
                if ref_loc in loc_defs:
                    loc_defs[loc_var] = loc_defs[ref_loc]
            continue

        # Match: #loc36 = loc(callsite(#loc1 at #loc33))
        match = re.match(
            r"^(#loc\d+)\s*=\s*loc\(callsite\(#loc\d+ at #loc(\d+)\)\)", line
        )
        if match:
            loc_var, ref_loc_num = match.groups()
            ref_loc = f"#loc{ref_loc_num}"
            if ref_loc in loc_defs:
                loc_defs[loc_var] = loc_defs[ref_loc]
            continue

        # Match: #loc1 = loc(unknown) - ignore
        if re.match(r"^#loc\d+\s*=\s*loc\(unknown\)", line):
            continue

        # This is an IR line - check if it has a location reference
        loc_match = re.search(r"loc\(#loc(\d+)\)", line)
        if loc_match:
            loc_num = int(loc_match.group(1))
            ir_line_to_loc[line_num] = f"#loc{loc_num}"

    return ir_line_to_loc, loc_defs


def parse_ptx_locations(ptx_path):
    """
    Parse PTX file to extract:
    - Line number to location mapping
    - PTX .loc directives have format: .loc [file_id] [line] [col] // filename:line:col

    Patterns:
    1. .loc 1 69 16 // vector_add.py:69:16 - direct location
    2. .loc 2 261 15 // standard.py:261:15 @ - callsite (need to find caller location)
    3. .loc 1 799 16 - without comment (file_id, line, col only)

    Uses two-pass parsing: first collect all .file directives, then parse .loc
    """
    with open(ptx_path, "r") as f:
        lines = f.readlines()

    # First pass: collect all .file directives
    file_id_to_name = {}
    for line in lines:
        file_match = re.match(r'^\s*\.file\s+(\d+)\s+"([^"]+)"', line)
        if file_match:
            file_id = int(file_match.group(1))
            filename = file_match.group(2)
            file_id_to_name[file_id] = filename

    # Second pass: parse .loc directives
    ir_line_to_loc = {}
    prev_loc_info = None

    for i, line in enumerate(lines):
        line_num = i + 1

        # Skip .file directives (already processed in first pass)
        if re.match(r"^\s*\.file\s+", line):
            continue

        # Match: .loc 1 69 16 // vector_add.py:69:16
        loc_match = re.match(
            r"^\s*\.loc\s+(\d+)\s+(\d+)\s+(\d+)\s*//\s*([^:]+):(\d+):(\d+)", line
        )
        if loc_match:
            file_id = int(loc_match.group(1))
            ptx_line = int(loc_match.group(2))
            ptx_col = int(loc_match.group(3))
            filename = loc_match.group(4)
            src_line = int(loc_match.group(5))
            src_col = int(loc_match.group(6))

            prev_loc_info = (filename, src_line, src_col)
            ir_line_to_loc[line_num] = (filename, src_line, src_col)
            continue

        # Check if this line has a reference to a callsite (ends with @callee[line])
        # Pattern: .loc 2 291 36 // standard.py:291:36 @vector_add.py[77]
        # - file_id 2 may not be in .file table (external function)
        # - After @ is callee file (vector_add.py) with line in brackets [77]
        callsite_match = re.match(
            r"^\s*\.loc\s+(\d+)\s+(\d+)\s+(\d+)\s*//\s*([^:]+):(\d+):(\d+)\s*@([^[]+)\[(\d+)\]",
            line,
        )
        if callsite_match:
            file_id = int(callsite_match.group(1))
            ptx_line = int(callsite_match.group(2))
            ptx_col = int(callsite_match.group(3))
            callsite_file = callsite_match.group(4)
            callsite_line = int(callsite_match.group(5))
            callsite_col = int(callsite_match.group(6))
            callee_file = callsite_match.group(7)
            callee_line = int(callsite_match.group(8))

            # Look up callee_file in file_id_to_name to find actual source
            # file_id maps to actual source file
            if file_id in file_id_to_name:
                # Use actual source file from .file table with callee line
                actual_file = file_id_to_name[file_id]
                ir_line_to_loc[line_num] = (actual_file, callee_line, ptx_col)
            else:
                # file_id not in .file table - use callee filename from @ with callee line
                ir_line_to_loc[line_num] = (callee_file, callee_line, ptx_col)
            continue

        # Match: .loc 1 799 16 (no comment - file_id, line, col)
        # Skip if line is 0 (invalid)
        simple_loc_match = re.match(r"^\s*\.loc\s+(\d+)\s+(\d+)\s+(\d+)\s*$", line)
        if simple_loc_match:
            file_id = int(simple_loc_match.group(1))
            ptx_line = int(simple_loc_match.group(2))
            ptx_col = int(simple_loc_match.group(3))
            # Skip if line is 0 (invalid)
            if ptx_line > 0 and file_id in file_id_to_name:
                filename = file_id_to_name[file_id]
                prev_loc_info = (filename, ptx_line, ptx_col)
                ir_line_to_loc[line_num] = (filename, ptx_line, ptx_col)
            continue

    return ir_line_to_loc, {}


def resolve_location(loc_var, loc_defs):
    """Resolve a location variable to its final file:line:col"""
    if loc_var not in loc_defs:
        return None
    return loc_defs[loc_var]


def build_line_bindings(ir_line_to_loc, loc_defs, py_path, file_type):
    """
    Build mapping from IR line -> Python line(s)
    """
    binding = {}
    py_line_to_ir = {}

    # Extract just the filename from py_path for matching
    py_filename = py_path.split("/")[-1]

    for ir_line, loc_info in ir_line_to_loc.items():
        # For PTX, loc_info is already (filename, line, col)
        # For TTIR, loc_info is a loc_var string that needs resolution
        if file_type == "ptx":
            if loc_info is None:
                continue
            filepath, py_line, _ = loc_info
        else:
            resolved = resolve_location(loc_info, loc_defs)
            if resolved is None:
                continue
            filepath, py_line, _ = resolved

        # Match by filename
        if (
            filepath.endswith(py_path)
            or py_path in filepath
            or filepath.endswith(py_filename)
        ):
            binding.setdefault(ir_line, []).append(py_line)
            py_line_to_ir.setdefault(py_line, []).append(ir_line)

    return binding, py_line_to_ir


def read_file(path):
    """Read file lines"""
    with open(path, "r") as f:
        return f.readlines()


def generate_html(ir_path, py_path, output_path=None, file_type=None, verbose=True):
    """Generate the HTML comparison view"""
    import os

    # Auto-detect file type if not specified
    if file_type is None:
        file_type = detect_file_type(ir_path)
        if file_type == "unknown":
            raise ValueError(
                f"Unknown file type for {ir_path}. Supported: .ttir, .ttgir, .ptx"
            )

    # Auto-generate output_path if not specified
    if output_path is None:
        ir_dir = os.path.dirname(os.path.abspath(ir_path))
        ir_name = os.path.splitext(os.path.basename(ir_path))[0]
        output_path = os.path.join(ir_dir, f"{ir_name}_viewer_{file_type}.html")

    # Parse based on file type
    if file_type == "ptx":
        ir_line_to_loc, loc_defs = parse_ptx_locations(ir_path)
        panel_name = "PTX"
    else:
        ir_line_to_loc, loc_defs = parse_ttir_locations(ir_path)
        panel_name = "TTGIR" if file_type == "ttgir" else "TTIR"

    # Build bindings
    binding, py_line_to_ir = build_line_bindings(
        ir_line_to_loc, loc_defs, py_path, file_type
    )

    # Read source files
    ir_lines = read_file(ir_path)
    py_lines = read_file(py_path)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>TeraXLang IR Viewer</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; height: 100vh; display: flex; flex-direction: column; }}
        header {{ background: #1a1a2e; color: white; padding: 12px 20px; font-size: 14px; display: flex; justify-content: space-between; align-items: center; }}
        .title {{ font-size: 20px; font-weight: bold; }}
        .help-btn {{ background: #4caf50; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 13px; }}
        .help-btn:hover {{ background: #45a049; }}
        .help-modal {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 1000; }}
        .help-modal.show {{ display: flex; align-items: center; justify-content: center; }}
        .help-content {{ background: white; padding: 25px; border-radius: 8px; max-width: 500px; max-height: 80vh; overflow: auto; }}
        .help-content h2 {{ margin-top: 0; color: #1a1a2e; }}
        .help-content h3 {{ color: #333; margin-top: 15px; }}
        .help-content ul {{ padding-left: 20px; }}
        .help-content li {{ margin: 8px 0; }}
        .help-close {{ background: #1a1a2e; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin-top: 15px; }}
        .container {{ flex: 1; display: flex; overflow: hidden; }}
        .panel {{ flex: 1; display: flex; flex-direction: column; border-right: 1px solid #ddd; overflow: hidden; }}
        .panel:last-child {{ border-right: none; }}
        .panel-header {{ background: #f5f5f5; padding: 8px 12px; font-weight: 600; font-size: 13px; border-bottom: 1px solid #ddd; display: flex; justify-content: space-between; }}
        .panel-content {{ flex: 1; overflow: auto; font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace; font-size: 12px; line-height: 1.5; overflow-x: scroll; }}
        .line {{ display: flex; min-width: max-content; }}
        .line-num {{ width: 50px; text-align: right; padding-right: 12px; color: #999; user-select: none; border-right: 1px solid #eee; flex-shrink: 0; }}
        .line-content {{ white-space: pre; padding-left: 12px; }}
        .line:hover {{ background: #f0f0f0; cursor: pointer; }}
        .line.linked:hover {{ background: #e3f2fd; }}
        .line.linked.py-linked {{ background: #fff3e0; }}
        .line.linked.ir-linked {{ background: #e8f5e9; }}
        .line.highlight-py {{ background: #ff9800 !important; color: white; }}
        .line.highlight-ir {{ background: #4caf50 !important; color: white; }}
        .line.highlight-ir.highlight-py {{ background: #9c27b0 !important; }}
        .legend {{ font-size: 13px; color: #ccc; display: flex; gap: 20px; align-items: center; }}
        .legend span {{ display: inline-flex; align-items: center; gap: 5px; }}
        .legend .ir-linked {{ color: #81c784; }}
        .legend .py-linked {{ color: #ffb74d; }}
        .legend .both-linked {{ color: #ce93d8; }}
        .help-icon {{ background: #666; width: 18px; height: 18px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; }}
    </style>
</head>
<body>
    <header>
        <span class="title">TeraXLang IR Viewer</span>
        <div style="display: flex; align-items: center; gap: 20px;">
            <span class="legend">
                <span class="ir-linked">■ {panel_name} line → Python</span>
                <span class="py-linked">■ Python line → {panel_name}</span>
                <span class="both-linked">■ Both bound</span>
            </span>
            <button class="help-btn" onclick="document.getElementById('help-modal').classList.add('show')">?</button>
        </div>
    </header>
    
    <div id="help-modal" class="help-modal" onclick="if(event.target === this) this.classList.remove('show')">
        <div class="help-content">
            <h2>TeraXLang IR Viewer Help</h2>
            
            <h3>Overview</h3>
            <p>This tool shows the binding between IR code ({panel_name}) and Python source code. Click on any line to navigate between related code.</p>
            
            <h3>Navigation</h3>
            <ul>
                <li><strong>Click {panel_name} line</strong> → Highlights and scrolls to bound Python lines</li>
                <li><strong>Click Python line</strong> → Highlights and scrolls to bound {panel_name} lines</li>
            </ul>
            
            <h3>Color Legend</h3>
            <ul>
                <li><span class="ir-linked">■ Green</span> - {panel_name} line has Python binding</li>
                <li><span class="py-linked">■ Orange</span> - Python line has {panel_name} binding</li>
                <li><span class="both-linked">■ Purple</span> - Both have bindings</li>
            </ul>
            
            <h3>Keyboard Shortcuts</h3>
            <ul>
                <li><strong>Click + Hold</strong> - Highlight multiple lines</li>
                <li><strong>Escape</strong> - Clear all highlights</li>
            </ul>
            
            <button class="help-close" onclick="document.getElementById('help-modal').classList.remove('show')">Close</button>
        </div>
    </div>
    
    <div class="container">
        <div class="panel">
            <div class="panel-header">
                <span>{panel_name}</span>
                <span>{ir_path}</span>
            </div>
            <div class="panel-content" id="ir-panel">
"""

    # Generate IR lines
    for i, line in enumerate(ir_lines, 1):
        line_num = i
        has_binding = line_num in binding
        binding_py_lines = binding.get(line_num, [])
        classes = ["line"]
        if has_binding:
            classes.append("linked")
            classes.append("ir-linked")

        escaped_line = (
            line.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
        py_lines_attr = ",".join(map(str, binding_py_lines)) if binding_py_lines else ""

        html += f'''                <div class="{" ".join(classes)}" data-ir-line="{line_num}" data-py-lines="{py_lines_attr}">
                    <span class="line-num">{line_num}</span>
                    <span class="line-content">{escaped_line}</span>
                </div>
'''

    html += (
        """            </div>
        </div>
        <div class="panel">
            <div class="panel-header">
                <span>Python</span>
                <span>"""
        + py_path
        + """</span>
            </div>
            <div class="panel-content" id="py-panel">
"""
    )

    # Generate Python lines
    for i, line in enumerate(py_lines, 1):
        line_num = i
        has_binding = line_num in py_line_to_ir
        binding_ir_lines = py_line_to_ir.get(line_num, [])
        classes = ["line"]
        if has_binding:
            classes.append("linked")
            classes.append("py-linked")

        escaped_line = (
            line.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
        ir_lines_attr = ",".join(map(str, binding_ir_lines)) if binding_ir_lines else ""

        html += f'''                <div class="{" ".join(classes)}" data-py-line="{line_num}" data-ir-lines="{ir_lines_attr}">
                    <span class="line-num">{line_num}</span>
                    <span class="line-content">{escaped_line}</span>
                </div>
'''

    html += """            </div>
        </div>
    </div>
    <script>
        const irPanel = document.getElementById('ir-panel');
        const pyPanel = document.getElementById('py-panel');
        
        // IR line click -> highlight corresponding Python lines
        irPanel.addEventListener('click', (e) => {
            const line = e.target.closest('.line[data-ir-line]');
            if (!line) return;
            
            const irLine = parseInt(line.dataset.irLine);
            const pyLines = line.dataset.pyLines.split(',').filter(x => x).map(Number);
            
            // Clear previous highlights
            document.querySelectorAll('.highlight-ir').forEach(el => {
                el.classList.remove('highlight-ir');
            });
            document.querySelectorAll('.highlight-py').forEach(el => {
                el.classList.remove('highlight-py');
            });
            
            // Highlight clicked IR line
            line.classList.add('highlight-ir');
            
            // Scroll and highlight corresponding Python lines
            pyLines.forEach(pyLine => {
                const pyEl = pyPanel.querySelector(`[data-py-line="${pyLine}"]`);
                if (pyEl) {
                    pyEl.classList.add('highlight-py');
                    pyEl.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'nearest' });
                }
            });
            
            // Scroll IR line into view (only vertically)
            line.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'nearest' });
        });
        
        // Python line click -> highlight corresponding IR lines
        pyPanel.addEventListener('click', (e) => {
            const line = e.target.closest('.line[data-py-line]');
            if (!line) return;
            
            const pyLine = parseInt(line.dataset.pyLine);
            const irLines = line.dataset.irLines.split(',').filter(x => x).map(Number);
            
            // Clear previous highlights
            document.querySelectorAll('.highlight-ir').forEach(el => {
                el.classList.remove('highlight-ir');
            });
            document.querySelectorAll('.highlight-py').forEach(el => {
                el.classList.remove('highlight-py');
            });
            
            // Highlight clicked Python line
            line.classList.add('highlight-py');
            
            // Scroll and highlight corresponding IR lines
            let firstIrLine = null;
            irLines.forEach(irLine => {
                const irEl = irPanel.querySelector(`[data-ir-line="${irLine}"]`);
                if (irEl) {
                    irEl.classList.add('highlight-ir');
                    if (firstIrLine === null) {
                        firstIrLine = irLine;
                        irEl.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'nearest' });
                    }
                }
            });
            
            // Scroll Python line into view (only vertically)
            line.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'nearest' });
        });
        
        // Escape key to clear highlights
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                document.querySelectorAll('.highlight-ir').forEach(el => {
                    el.classList.remove('highlight-ir');
                });
                document.querySelectorAll('.highlight-py').forEach(el => {
                    el.classList.remove('highlight-py');
                });
                // Also close help modal if open
                document.getElementById('help-modal').classList.remove('show');
            }
        });
    </script>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(html)

    print(f"Generated HTML: {output_path}")
    print(f"{panel_name} lines: {len(ir_lines)}, Python lines: {len(py_lines)}")
    print(
        f"Bindings found: {panel_name}->Python: {len(binding)}, Python->{panel_name}: {len(py_line_to_ir)}"
    )


def get_file_paths_array(base_path):
    """Returns ALL IR files (not just one of each type)"""
    target_exts = [".ttir", ".ttgir", ".ptx"]
    found_files = []

    for file in base_path.rglob("*"):
        if file.is_file() and file.suffix in target_exts:
            found_files.append(str(file.resolve()))

    return found_files


def generate_htmls(ir_path, py_path, verbose=True):
    base_path = Path(ir_path)
    if base_path.is_file():
        print(f"FILE: {ir_path}")
        generate_html(ir_path, py_path, verbose=verbose)
        return

    ir_files = get_file_paths_array(base_path)
    print(f"Found {len(ir_files)} IR files")
    for f in ir_files:
        print(f"Processing: {f}")
        generate_html(f, py_path, verbose=verbose)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TeraXLang IR Viewer - Build HTML to compare IR (TTIR/TTGIR/PTX) with Python source",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s kernel.ttir kernel.py
  %(prog)s kernel.ttir kernel.py viewer.html
  %(prog)s kernel.ptx kernel.py

Supported IR formats:
  .ttir  - Triton TTIR (Triton Intermediate Representation)
  .ttgir - Triton TTGIR (Triton GPU IR)  
  .ptx   - NVIDIA PTX (Parallel Thread Execution)

The viewer shows interactive bindings between IR code lines and Python source lines.
Click on any line to navigate between related code.
        """,
    )
    parser.add_argument("ir_file", help="Path to IR file (.ttir, .ttgir, or .ptx)")
    parser.add_argument("py_file", help="Path to Python source file")
    parser.add_argument(
        "output_html",
        nargs="?",
        help="Output path for HTML viewer (default: same dir as ir_file)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=True,
        help="Show detailed binding info (default: True)",
    )
    parser.add_argument(
        "-t",
        "--type",
        choices=["ttir", "ttgir", "ptx"],
        help="IR file type (auto-detected if not specified)",
    )

    args = parser.parse_args()

    ir_path = args.ir_file
    py_path = args.py_file

    # generate_html(ir_path, py_path, args.output_html, args.type, args.verbose)
    generate_htmls(ir_path, py_path, args.verbose)
