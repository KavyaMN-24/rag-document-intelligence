
from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass
class PageText:
    """Text extracted from a single PDF page."""

    page_number: int
    text: str


def extract_pdf(file_path: str | Path) -> list[PageText]:
    """
    Extract text from a PDF while preserving page numbers.

    Args:
        file_path: Path to the PDF document.

    Returns:
        A list of PageText objects, one per page.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If the file is not a PDF or is encrypted.
        RuntimeError: If PDF extraction fails.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file, received: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    try:
        with pymupdf.open(path) as document:

            if document.needs_pass:
                raise ValueError("Password-protected PDFs are not supported.")

            extracted_pages = []

            for page_index, page in enumerate(document):
                text = page.get_text("text").strip()

                extracted_pages.append(
                    PageText(
                        page_number=page_index + 1,
                        text=text,
                    )
                )

            return extracted_pages

    except ValueError:
        raise

    except Exception as exc:
        raise RuntimeError(
            f"Failed to extract PDF: {path}"
        ) from exc
