from pathlib import Path
import pandas as pd

class MultimodalDatasetManifest:
    def __init__(self, root='data'):
        self.root = Path(root)
    def check(self):
        required = [
            self.root/'numerical'/'numerical_features.csv',
            self.root/'textual'/'text_documents.csv',
            self.root/'images',
            self.root/'blockchain_graphs'
        ]
        return {str(p): p.exists() for p in required}
