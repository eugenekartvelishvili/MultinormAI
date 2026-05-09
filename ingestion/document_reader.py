# ingestion/document_reader.py

from pathlib import Path
from code.ingestion.models import RawDocument
import uuid
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling.datamodel.base_models import InputFormat
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.backend.msword_backend import MsWordDocumentBackend
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
    TesseractCliOcrOptions,
)




# -------------------------------
# Reader
# -------------------------------

class DocumentReader:
    def __init__(self, artifacts_path: str):
        self._converter = self._build_converter(artifacts_path)

    def read(self, path: Path) -> RawDocument:
        path = Path(path)
        doc_id = str(uuid.uuid4())

        result = self._converter.convert(str(path))
        document = result.document

        html = document.export_to_html()

        return RawDocument(
                doc_id=doc_id,
                html=html,
                metadata={
                    "source_path": str(path),
                    "format": path.suffix.lower(),
                },
        )

    # ---------------------------
    # Internal
    # ---------------------------

    @staticmethod
    def _build_converter(artifacts_path: str) -> DocumentConverter:
        ocr_options = TesseractCliOcrOptions(
            lang=["rus", "eng"],
        )

        pdf_options = PdfPipelineOptions(
            artifacts_path=artifacts_path,
            enable_remote_services=True,
            ocr_options=ocr_options,
        )

        pdf_options.generate_page_images = False
        pdf_options.do_ocr = True
        pdf_options.allow_external_plugins = True
        pdf_options.do_table_structure = True
        pdf_options.table_structure_options.do_cell_matching = True
        pdf_options.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.AUTO,
        )

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_options,
                    backend=DoclingParseV4DocumentBackend,
                ),
                InputFormat.DOCX: WordFormatOption(
                    backend=MsWordDocumentBackend,
                ),
            }
        )
