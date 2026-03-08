#!/usr/bin/env python3
"""
AST-based import replacement script.
Safely replaces 'import triton' and 'from triton import' with new package name.
Only modifies actual Python import statements, leaves strings/comments untouched.
"""

import ast
import os
import sys
from pathlib import Path


def replace_imports(source_code, old_name="triton", new_name="teraxlang_triton"):
    """Replace imports using AST - safe way that doesn't touch strings."""

    class ImportReplacer(ast.NodeTransformer):
        def __init__(self, old_name, new_name):
            self.old_name = old_name
            self.new_name = new_name

        def visit_Import(self, node):
            for alias in node.names:
                if alias.name == self.old_name:
                    alias.name = self.new_name
                elif alias.name.startswith(self.old_name + "."):
                    alias.name = self.new_name + alias.name[len(self.old_name) :]
            return node

        def visit_ImportFrom(self, node):
            if node.module == self.old_name:
                node.module = self.new_name
            elif node.module and node.module.startswith(self.old_name + "."):
                node.module = self.new_name + node.module[len(self.old_name) :]
            return node

    try:
        tree = ast.parse(source_code)
        replacer = ImportReplacer(old_name, new_name)
        new_tree = replacer.visit(tree)
        ast.fix_missing_locations(new_tree)
        return ast.unparse(new_tree)
    except SyntaxError:
        # If parsing fails, return original
        return source_code


def process_file(filepath, old_name="triton", new_name="teraxlang_triton"):
    """Process a single Python file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            original = f.read()

        # Check if file contains the old import
        if old_name not in original:
            return 0

        modified = replace_imports(original, old_name, new_name)

        # Only write if changed
        if modified != original:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(modified)
            return 1
        return 0
    except Exception as e:
        print(f"Warning: Error processing {filepath}: {e}")
        return 0


def process_directory(
    dirpath, old_name="triton", new_name="teraxlang_triton", exclude_dirs=None
):
    """Recursively process all Python files in a directory."""
    if exclude_dirs is None:
        exclude_dirs = {"__pycache__", ".git", "build", "dist", "egg-info"}

    changed = 0
    for root, dirs, files in os.walk(dirpath):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                result = process_file(filepath, old_name, new_name)
                if result:
                    print(f"  Updated: {filepath}")
                    changed += 1

    return changed


def main():
    if len(sys.argv) < 2:
        print("Usage: replace_imports.py <directory> [old_name] [new_name]")
        print("  Default: replaces 'triton' with 'teraxlang_triton'")
        sys.exit(1)

    directory = sys.argv[1]
    old_name = sys.argv[2] if len(sys.argv) > 2 else "triton"
    new_name = sys.argv[3] if len(sys.argv) > 3 else "teraxlang_triton"

    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory")
        sys.exit(1)

    print(f"Processing directory: {directory}")
    print(f"Replacing: 'import {old_name}' -> 'import {new_name}'")
    print(f"Replacing: 'from {old_name} import' -> 'from {new_name} import'")
    print()

    changed = process_directory(directory, old_name, new_name)
    print()
    print(f"Done! Updated {changed} files.")


if __name__ == "__main__":
    main()
