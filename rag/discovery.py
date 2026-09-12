"""Path resolution and file discovery, without loading retrieval models."""

import os
import stat
from typing import Optional

from .formats import INDEXABLE_EXTENSIONS as _INDEXABLE_EXTENSIONS


def resolve_path(path: str) -> Optional[str]:
    """Resolve user-friendly paths to absolute paths"""
    if not path:
        return None
    
    # Handle common shortcuts
    shortcuts = {
        "desktop": "~/Desktop",
        "documents": "~/Documents",
        "downloads": "~/Downloads",
    }
    
    lower = path.lower().strip()
    if lower in shortcuts:
        path = shortcuts[lower]
    
    # Expand ~ and environment variables
    path = os.path.expanduser(path)
    path = os.path.expandvars(path)
    
    # Make absolute if relative
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    
    return path

def find_all_indexable_files(
    directory: str,
    max_depth: int = 5,
    include_patterns: Optional[list] = None,
    exclude_patterns: Optional[list] = None,
    max_files: Optional[int] = None,
    excluded_paths: Optional[set[str]] = None,
) -> list:
    """Recursively find indexable files with selective filtering"""
    if not os.path.isdir(directory):
        return []

    exclude_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'dist', 'build', 'target', '.ipynb_checkpoints', 'migrations', 'rag_db_v2'}

    all_files = []

    def matches_pattern(path: str, patterns: list) -> bool:
        """Check if path matches any glob pattern"""
        import fnmatch
        for pattern in patterns:
            if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern):
                return True
        return False

    def walk_dir(path: str, depth: int = 0):
        if depth > max_depth:
            return
        if max_files and len(all_files) >= max_files:
            return

        for entry in os.listdir(path):
            # Skip hidden files and excluded directories
            if entry.startswith('.') or entry in exclude_dirs:
                continue

            entry_path = os.path.join(path, entry)
            # Apply exclude patterns
            if exclude_patterns and matches_pattern(entry_path, exclude_patterns):
                continue

            # Read once without following links; metadata errors must not hide entries.
            mode = os.lstat(entry_path).st_mode
            if stat.S_ISREG(mode):
                if excluded_paths and os.path.realpath(entry_path) in excluded_paths:
                    continue
                if entry.lower().endswith(_INDEXABLE_EXTENSIONS):
                    # Apply include patterns
                    if include_patterns:
                        if matches_pattern(entry_path, include_patterns):
                            all_files.append(entry_path)
                    else:
                        all_files.append(entry_path)

                    if max_files and len(all_files) >= max_files:
                        return
            elif stat.S_ISDIR(mode):
                walk_dir(entry_path, depth + 1)

    print(f"[RAG] Scanning: {directory}")
    if include_patterns:
        print(f"[RAG] Include patterns: {include_patterns}")
    if exclude_patterns:
        print(f"[RAG] Exclude patterns: {exclude_patterns}")
    if max_files:
        print(f"[RAG] Max files: {max_files}")

    walk_dir(directory)
    print(f"[RAG] Found {len(all_files)} indexable files")

    return all_files


def find_relevant_files(query: str, directory: str, max_files: int = 5) -> list:
    """Find files in directory whose names match query keywords"""
    if not os.path.isdir(directory):
        return []
    
    print(f"[RAG V2] Searching for files matching: '{query}' in {directory}")
    
    query_words = query.lower().split()
    
    # If no meaningful query, return first few indexable files
    if not query_words:
        all_files = []
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path) and file.lower().endswith(_INDEXABLE_EXTENSIONS):
                all_files.append(file_path)
        return all_files[:max_files]

    scored_files = []
    
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if not os.path.isfile(file_path):
            continue
        if not file.lower().endswith(_INDEXABLE_EXTENSIONS):
            continue
        
        # Score based on filename match
        filename_lower = file.lower()
        score = sum(1 for word in query_words if word in filename_lower)
        
        if score > 0:
            scored_files.append((score, file_path))
    
    # Return top matches
    scored_files.sort(reverse=True, key=lambda x: x[0])
    return [path for score, path in scored_files[:max_files]]
