"""FFIEC manual parsing (font-size headings) and the pypdf log allow-list (decision 2026-11-21)."""

import logging

import pytest

from src.rag.chunk import chunk_sections, citation
from src.rag.parse import (
    PYPDF_LOG_ALLOWLIST, PdfLogError, Run, body_size, check_pdf_log, clean_ffiec_page,
    heading_groups, parse_ffiec, read_pdf_pages_logged,
)

SAR_SHA = "bd42e2637c61ba044f1ede8b8756858aaf23d425c9cf76c3c2bb5530a3eee67e"
ALLOWED = "Ignoring wrong pointing object 6 0 (offset 0)"


# ---- pypdf log allow-list -------------------------------------------------

def test_allow_list_is_exactly_one_message_for_one_hash():
    assert PYPDF_LOG_ALLOWLIST == {SAR_SHA: ("Ignoring wrong pointing object",)}


def test_allowed_message_for_the_pinned_file_passes():
    assert check_pdf_log([ALLOWED] * 19, SAR_SHA) == [ALLOWED] * 19


def test_allowed_message_from_another_file_or_hash_stops():
    with pytest.raises(PdfLogError):
        check_pdf_log([ALLOWED], "0" * 64, "ffiec_ctr.pdf")


def test_unexpected_message_stops_even_for_the_allow_listed_file():
    with pytest.raises(PdfLogError, match="unexpected"):
        check_pdf_log([ALLOWED, "Multiple definitions in dictionary at byte 0x1f"], SAR_SHA)


def test_no_messages_is_fine_for_any_file():
    assert check_pdf_log([], "f" * 64) == []


def test_pypdf_logger_is_captured_and_restored(tmp_path, monkeypatch):
    """A message logged by pypdf while reading is returned (not printed), and the logger is restored."""
    pypdf = pytest.importorskip("pypdf")

    class FakePage:
        def extract_text(self):
            logging.getLogger("pypdf._reader").warning("Ignoring wrong pointing object 9 0 (offset 0)")
            return "page text"

    class FakeReader:
        def __init__(self, path):
            self.pages = [FakePage()]

    monkeypatch.setattr(pypdf, "PdfReader", FakeReader)
    logger = logging.getLogger("pypdf")
    before = (list(logger.handlers), logger.propagate)
    texts, messages = read_pdf_pages_logged(tmp_path / "x.pdf")
    assert texts == ["page text"] and messages == ["Ignoring wrong pointing object 9 0 (offset 0)"]
    assert (list(logger.handlers), logger.propagate) == before


# ---- page cleaning and headings ---------------------------------------------

def test_ffiec_page_strips_title_manual_line_and_nav_link():
    text = ("Currency Transaction Reporting \nFFIEC BSA/AML Examination Manual 3 February 2021 \n"
            "Body text. \nReturn to Contents ")
    page = clean_ffiec_page(text, 3, "Currency Transaction Reporting")
    assert page.page_label == "3" and page.lines == ["Body text."]
    assert set(page.stripped) == {"running_title", "manual_line", "nav_link"}


def test_ffiec_page_without_printed_page_number():
    page = clean_ffiec_page("Suspicious Activity Reporting — Overview \nSuspicious Activity Reporting — Overview "
                            "\nObjective. Assess", 1, "Suspicious Activity Reporting — Overview")
    assert page.page_label == ""
    assert page.lines == ["Suspicious Activity Reporting — Overview", "Objective. Assess"]  # 2nd = heading


def test_appendix_letter_page_labels():
    page = clean_ffiec_page("Appendix F \nFFIEC BSA/AML Examination Manual F–2 2/27/2015.V2 \nx", 2, "Appendix F")
    assert page.page_label == "F–2"


def test_body_size_and_heading_groups():
    runs = [Run("Header", 10, False), Run("Funds", 14, True), Run("Transfers", 14, True),
            Run("A long body sentence about wires.", 12, False), Run("Sub head", 14, False),
            Run("Objective:", 12, True), Run("more body text here", 12, False)]
    assert body_size([runs]) == 12
    assert heading_groups(runs, 12) == [("Funds Transfers", (14, True)), ("Sub head", (14, False))]


def fixture_pages():
    """Two pages modelled on Appendix F: a bold parent heading, wrapped title, sub-headings."""
    texts = [
        "Appendix F \nFFIEC BSA/AML Examination Manual F–1 2/27/2015.V2 \nAppendix F: Red \nFlags \n"
        "Intro sentence one. \nMoney Laundering Flags \nFunds Transfers \n• Wire activity is unexplained. \n"
        "• Many small incoming wires are fol- \n",
        "Appendix F \nFFIEC BSA/AML Examination Manual F–2 2/27/2015.V2 \nlowed by one outgoing wire. \n"
        "Terrorist Financing Flags \nFunds Transfers \n• Wires to higher-risk countries. \n",
    ]
    runs = [
        [Run("Appendix F", 10, False), Run("FFIEC BSA/AML Examination Manual F–1 2/27/2015.V2", 10, False),
         Run("Appendix F: Red", 16, True), Run("Flags", 16, True), Run("Intro sentence one.", 12, False),
         Run("Money Laundering Flags", 14, True), Run("Funds Transfers", 14, False),
         Run("•", 12, False), Run("Wire activity is unexplained. Many small incoming wires are followed", 12, False)],
        [Run("Appendix F", 10, False), Run("lowed by one outgoing wire. and more body words", 12, False),
         Run("Terrorist Financing Flags", 14, True), Run("Funds Transfers", 14, False),
         Run("Wires to higher-risk countries.", 12, False)],
    ]
    return texts, runs


def test_ffiec_sections_parents_slugs_and_pages(monkeypatch):
    monkeypatch.setitem(__import__("src.rag.parse", fromlist=["DOC_KINDS"]).DOC_KINDS,
                        "ffiec_appf", ("ffiec", "Appendix F"))
    texts, runs = fixture_pages()
    sections, pages = parse_ffiec("ffiec_appf", texts, runs)
    assert [p.page_label for p in pages] == ["F–1", "F–2"]
    assert [(s.section, s.title) for s in sections] == [
        ("appendix-f-red-flags", "Appendix F: Red Flags"),
        ("funds-transfers", "Funds Transfers"),
        ("funds-transfers-2", "Funds Transfers"),
    ]
    ml, tf = sections[1], sections[2]
    assert ml.subpart == "Appendix F: Red Flags › Money Laundering Flags"
    assert tf.subpart == "Appendix F: Red Flags › Terrorist Financing Flags"
    # de-hyphenation across the page break, page of each line kept
    assert [(line.text, line.pdf_page) for line in ml.lines] == [
        ("• Wire activity is unexplained.", 1), ("• Many small incoming wires are followed", 1),
        ("by one outgoing wire.", 2)]


def test_ffiec_chunk_heading_and_citation(monkeypatch):
    import src.rag.parse as parse
    monkeypatch.setitem(parse.DOC_KINDS, "ffiec_appf", ("ffiec", "Appendix F"))
    texts, runs = fixture_pages()
    chunks = chunk_sections(parse_ffiec("ffiec_appf", texts, runs)[0])
    tf = chunks[-1]
    assert tf.chunk_id == "ffiec_appf:funds-transfers-2:001"
    assert tf.heading == ("FFIEC BSA/AML Examination Manual: Appendix F › Appendix F: Red Flags › "
                          "Terrorist Financing Flags › Funds Transfers")
    assert citation(tf.to_record()) == ("FFIEC BSA/AML Examination Manual, Appendix F, "
                                        "“Funds Transfers”, PDF p. 2, printed p. F–2")
    ml = chunks[1].to_record()
    ml.update(page_end=2, label_end="F–2")
    assert citation(ml).endswith("PDF p. 1–2, printed p. F–1 to F–2")


# ---- real files ---------------------------------------------------------------

@pytest.mark.corpus
def test_ffiec_sar_extraction_is_complete_and_only_allowed_messages():
    from src.rag.corpus import RAW_DIR
    texts, messages = read_pdf_pages_logged(RAW_DIR / "ffiec_sar.pdf")
    assert len(texts) == 17 and all(t.strip() for t in texts)
    assert sum(len(t.split()) for t in texts) > 7000
    assert messages and all(m.startswith("Ignoring wrong pointing object") for m in messages)


@pytest.mark.corpus
def test_other_ffiec_files_log_nothing():
    from src.rag.corpus import RAW_DIR
    for name in ("ffiec_ctr", "ffiec_cdd", "ffiec_cip", "ffiec_risk", "ffiec_appf"):
        texts, messages = read_pdf_pages_logged(RAW_DIR / f"{name}.pdf")
        assert messages == [] and all(t.strip() for t in texts), name


@pytest.mark.corpus
def test_real_ffiec_sections():
    from src.rag.corpus import RAW_DIR
    from src.rag.parse import load_document
    sar = {s.section for s in load_document("ffiec_sar", RAW_DIR / "ffiec_sar.pdf")[0]}
    assert {"timing-of-a-sar-filing", "record-retention-and-supporting-documentation",
            "prohibition-of-sar-disclosure"} <= sar
    appf = load_document("ffiec_appf", RAW_DIR / "ffiec_appf.pdf")[0]
    assert sum(s.title == "Funds Transfers" for s in appf) == 2
    assert any("Terrorist Financing" in s.subpart for s in appf if s.title == "Funds Transfers")
