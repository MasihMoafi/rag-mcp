"""Flatten JSON values into the existing searchable-text representation."""

import json


def extract_json_text(json_path: str) -> str:
    """Extract text from JSON - handles various structures"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert JSON to searchable text
        def flatten_json(obj, prefix=''):
            text_parts = []
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, (dict, list)):
                        text_parts.extend(flatten_json(value, f"{prefix}{key}."))
                    else:
                        text_parts.append(f"{prefix}{key}: {value}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    text_parts.extend(flatten_json(item, f"{prefix}[{i}]."))
            else:
                text_parts.append(str(obj))
            return text_parts

        text_parts = flatten_json(data)
        return "\n".join(text_parts)
    except Exception as e:
        print(f"Error extracting text from {json_path}: {e}")
        return ""
