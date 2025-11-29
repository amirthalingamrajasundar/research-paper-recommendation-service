import yaml
from pathlib import Path
from types import SimpleNamespace

class Config:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        
        # Load both config files
        self._data_config = self._load_yaml('data_config.yaml')
        self._transform_paths(self._data_config)
        self._model_config = self._load_yaml('model_config.yaml')
        self._transform_paths(self._model_config)
        
        # Convert to namespaces for dot notation
        self.data = self._dict_to_namespace(self._data_config)
        self.model = self._dict_to_namespace(self._model_config)
    
    def _load_yaml(self, filename):
        """Load a YAML config file"""
        config_path = self.project_root / 'config' / filename
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _transform_paths(self, config):
        """Recursively convert path strings to absolute Path objects"""
        if isinstance(config, dict):
            for key, value in config.items():
                if key == 'paths' and isinstance(value, dict):
                    # Transform all paths in a 'paths' section
                    for path_key, path_value in value.items():
                        if isinstance(path_value, str):
                            config[key][path_key] = self.project_root / path_value
                elif isinstance(value, dict):
                    # Recurse into nested dicts
                    self._transform_paths(value)
    
    def _dict_to_namespace(self, d):
        """Recursively convert dict to SimpleNamespace for dot notation"""
        if isinstance(d, dict):
            return SimpleNamespace(**{k: self._dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return d
        return d

# Single instance
settings = Config()