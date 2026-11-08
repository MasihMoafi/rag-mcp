"""Include and validate the optional Studio bundle when present."""
from pathlib import Path
import importlib.util

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


_validator_path = Path(__file__).resolve().parent / 'rag' / 'studio_build_manifest.py'
_validator_spec = importlib.util.spec_from_file_location('_studio_build_manifest', _validator_path)
_validator = importlib.util.module_from_spec(_validator_spec)
_validator_spec.loader.exec_module(_validator)
read_bundle_manifest = _validator.read_bundle_manifest


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        bundle = Path(self.root) / 'rag' / 'studio_dist' / 'bundle.tar.gz'
        if not bundle.is_file():
            return
        try:
            read_bundle_manifest(bundle)
        except ValueError as error:
            raise RuntimeError('Studio bundle is incomplete or has invalid build provenance. Restage it before packaging.') from error
        build_data.setdefault('force_include', {})[str(bundle)] = 'rag/studio_dist/bundle.tar.gz'
