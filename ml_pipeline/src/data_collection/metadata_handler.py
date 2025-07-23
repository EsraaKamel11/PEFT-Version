import logging
from typing import Dict, Any, List

class MetadataHandler:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metadata_store: List[Dict[str, Any]] = []

    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        try:
            self.metadata_store.append(metadata)
            self.logger.info(f"Added metadata: {metadata}")
        except Exception as e:
            self.logger.error(f"Failed to add metadata: {e}")

    def get_all_metadata(self) -> List[Dict[str, Any]]:
        return self.metadata_store

    def save_metadata(self, path: str) -> None:
        import json
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata_store, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved metadata to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")

    def load_metadata(self, path: str) -> None:
        import json
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.metadata_store = json.load(f)
            self.logger.info(f"Loaded metadata from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}") 