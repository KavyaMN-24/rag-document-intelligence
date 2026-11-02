
import pymupdf
import pytest

from app.ingestion.pdf_loader import extract_pdf


def create_sample_pdf(path):
    """Create a small PDF for testing."""
    document = pymupdf.open()

    page1 = document.new_page()
    page1.insert_text((72, 72), "Artificial Intelligence is growing.")

    page2 = document.new_page()
    page2.insert_text((72, 72), "Retrieval Augmented Generation.")

    document.save(path)
    document.close()


def test_extract_pdf_returns_all_pages(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    create_sample_pdf(pdf_path)

    pages = extract_pdf(pdf_path)

    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[1].page_number == 2


def test_extract_pdf_returns_correct_text(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    create_sample_pdf(pdf_path)

    pages = extract_pdf(pdf_path)

    assert "Artificial Intelligence" in pages[0].text
    assert "Retrieval Augmented Generation" in pages[1].text


def test_missing_pdf_raises_error():
    with pytest.raises(FileNotFoundError):
        extract_pdf("nonexistent.pdf")


def test_non_pdf_file_raises_error(tmp_path):
    text_file = tmp_path / "document.txt"
    text_file.write_text("This is not a PDF.")

    with pytest.raises(ValueError):
        extract_pdf(text_file)
