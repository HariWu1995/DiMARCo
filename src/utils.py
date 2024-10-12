from pathlib import Path

import json
import yaml


def normalize_path(file_path: Path or str):
    if not isinstance(file_path, str):
        file_path = str(file_path)
    return file_path


def load_json(file_path: Path or str):
    file_path = normalize_path(file_path)
    with open(file_path) as f:
        data = json.load(f)
    return data


def save_json(file_path: Path or str, data: dict):
    file_path = normalize_path(file_path)
    with open(file_path, 'w') as f:
        json.dump(data, f)


def load_yaml(file_path: Path or str):
    file_path = normalize_path(file_path)
    with open(file_path) as f:
        data = yaml.safe_load(f)
    return data


def save_yaml(file_path: Path or str, data: dict):
    file_path = normalize_path(file_path)
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
