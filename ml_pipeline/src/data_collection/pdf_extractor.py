import os
import pdfplumber
import PyPDF2
from typing import List, Dict, Any
from .base_collector import BaseCollector

class PDFExtractor(BaseCollector):
    def __init__(self, output_dir: str):
        super().__init__(output_dir)

    def collect(self, pdf_paths: List[str]) -> None:
        for pdf_path in pdf_paths:
            self.logger.info(f"Extracting PDF: {pdf_path}")
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    pages = []
                    for page in pdf.pages:
                        layout = page.extract_text(x_tolerance=1, y_tolerance=1, layout=True)
                        pages.append(layout)
                metadata = self._extract_metadata(pdf_path)
                filename = self._sanitize_filename(pdf_path) + '.json'
                self.save_data(pages, metadata, filename)
            except Exception as e:
                self.logger.error(f"Error extracting {pdf_path}: {e}")

    def _extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        meta = {'source': pdf_path, 'type': 'pdf'}
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                doc_info = reader.metadata
                meta.update({k: str(v) for k, v in doc_info.items()})
        except Exception as e:
            self.logger.error(f"Error extracting metadata from {pdf_path}: {e}")
        return meta

    def _sanitize_filename(self, path: str) -> str:
        import re
        return re.sub(r'[^a-zA-Z0-9]', '_', os.path.basename(path))

    def resume(self, pdf_paths: List[str]) -> None:
        # Example: skip PDFs already extracted
        extracted = set()
        for fname in os.listdir(self.output_dir):
            if fname.endswith('.json'):
                extracted.add(fname.replace('.json', ''))
        to_extract = [p for p in pdf_paths if self._sanitize_filename(p) not in extracted]
        self.logger.info(f"Resuming. {len(to_extract)} PDFs left.")
        self.collect(to_extract) 