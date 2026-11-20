"""PDF page cleaning, de-hyphenation and section splitting (fixtures: page text, no PDF)."""

import pytest

from src.rag.parse import (
    Line, PageClean, clean_cfr_page, clean_faq_page, dehyphenate, parse_document,
    split_cfr_sections, split_faq_sections,
)

SLUG = ("VerDate Sep<11>2014 07:58 Sep 25, 2025 Jkt 265133 PO 00000 Frm 01198 Fmt 8010 "
        "Sfmt 8010 Y:\\SGML\\265133.XXX 265133skersey on DSK4WB1RN3PROD with CFR")


def page(n, label, lines):
    return PageClean(n, label, lines)


def test_cfr_page_strips_number_header_and_slug():
    text = "1189 \nFinancial Crimes Enforc. Network, Treas. \u00a7 1020.210 \nbody line one \n\n" + SLUG
    clean = clean_cfr_page(text, 2)
    assert clean.page_label == "1189"
    assert clean.lines == ["body line one"]
    assert set(clean.stripped) == {"page_number", "running_header", "slug"}


def test_cfr_edition_header_and_image_page():
    text = ("1130 \n31 CFR Ch. X (7\u20131\u201325 Edition) \u00a7 1010.230 \n"
            "VerDate Sep<11>2014 07:58 Sep 25, 2025 Jkt 265133 PO 00000 Frm 01130 \n"
            "ER28SE17.006</GPH> ER28SE17.007</GPH> \nskersey on DSK4WB1RN3PROD with CFR")
    clean = clean_cfr_page(text, 16)
    assert clean.lines == []
    assert clean.stripped["slug"] == 2 and clean.stripped["graphic_placeholder"] == 1


def test_first_page_without_running_header_keeps_its_text():
    clean = clean_cfr_page("1105 \nPART 1010\u2014GENERAL PROVISIONS \n" + SLUG, 1)
    assert clean.lines == ["PART 1010\u2014GENERAL PROVISIONS"]
    assert "running_header" not in clean.stripped


def test_faq_page_strips_page_number_and_blank_lines():
    clean = clean_faq_page(" \n3 \n \nongoing suspicious activity", 3)
    assert clean.page_label == "3" and clean.lines == ["ongoing suspicious activity"]


def lines(*texts, pdf_page=1):
    return [Line(t, pdf_page, str(pdf_page)) for t in texts]


def test_dehyphenate_joins_line_break_hyphens_across_pages():
    src = [Line("by peri-", 1, "1"), Line("odic review, periodic again", 2, "2")]
    out = dehyphenate(src)
    assert out[0].text == "by periodic" and out[0].pdf_page == 1
    assert out[1].text == "review, periodic again"


def test_dehyphenate_keeps_a_real_hyphen_seen_elsewhere():
    out = dehyphenate(lines("the anti-", "money laundering program", "an anti-money program"))
    assert out[0].text == "the anti-money"


def test_dehyphenate_keeps_trailing_punctuation_and_skips_capitals():
    out = dehyphenate(lines("re-", "quirements. Next", "see Part-", "Two"))
    assert out[0].text == "requirements." and out[1].text == "Next"
    assert out[2].text == "see Part-"


CFR_PAGES = [
    page(1, "1188", ["the State\u2019s system (tail of the previous part)",
                     "PART 1020\u2014RULES FOR BANKS",
                     "Subpart A\u2014Definitions", "Sec.", "1020.100 Definitions.",
                     "1020.110 [Reserved]",
                     "Subpart C\u2014Reports Required To Be Made", "by Banks",
                     "1020.320 Reports by banks of suspicious transactions.",
                     "AUTHORITY: 12 U.S.C. 1829b.",
                     "Subpart A\u2014Definitions",
                     "\u00a7 1020.100 Definitions.",
                     "Refer to \u00a7 1010.100 of this chapter. A definition in",
                     "\u00a7 1020.100 of this chapter applies."]),
    page(2, "1189", ["\u00a7 1020.110 [Reserved]",
                     "Subpart C\u2014Reports Required To",
                     "Be Made By Banks",
                     "\u00a7 1020.320 Reports by banks of",
                     "suspicious transactions.",
                     "(a) General. Every bank shall file",
                     "\u00a7 1020.200 Not a heading because numbers only go up."]),
    page(3, "1190", ["(b) continues on the next page.",
                     "PART 1021\u2014RULES FOR CASINOS", "\u00a7 1021.100 Definitions."]),
]


def test_cfr_sections_toc_and_boundaries():
    sections, toc = split_cfr_sections("cfr31_1020", "1020", CFR_PAGES)
    assert [s.section for s in sections] == ["1020.100", "1020.320"]
    assert toc == ["1020.100", "1020.320"]
    first, sar = sections
    assert first.title == "Definitions"
    assert [line.text for line in first.lines] == [
        "Refer to \u00a7 1010.100 of this chapter. A definition in",
        "\u00a7 1020.100 of this chapter applies."]
    assert sar.title == "Reports by banks of suspicious transactions"
    assert sar.subpart == "Subpart C\u2014Reports Required To Be Made By Banks"
    assert [line.text for line in sar.lines] == [
        "(a) General. Every bank shall file",
        "\u00a7 1020.200 Not a heading because numbers only go up.",
        "(b) continues on the next page."]
    assert [line.pdf_page for line in sar.lines] == [2, 2, 3]
    joined = " ".join(line.text for s in sections for line in s.lines)
    assert "tail of the previous part" not in joined and "CASINOS" not in joined


def test_faq_sections():
    pages = [page(1, "1", ["Intro text.", "Question 1: SAR Filings", "Is a SAR required?", "No."]),
             page(2, "2", ["More on one.", "Question 2: Reviews", "Answer two."])]
    sections = split_faq_sections("fincen_sar_faq_2025", pages)
    assert [(s.section, s.title) for s in sections] == [
        ("intro", "Introduction"), ("Q1", "Question 1: SAR Filings"), ("Q2", "Question 2: Reviews")]
    assert [line.pdf_page for line in sections[1].lines] == [1, 1, 2]


def test_unknown_document_kind():
    with pytest.raises(KeyError):
        parse_document("ffiec_sar", ["text"])


@pytest.mark.corpus
def test_real_corpus_parses_cleanly():
    from src.rag.chunk import chunk_sections
    from src.rag.corpus import RAW_DIR, load_manifest
    from src.rag.parse import read_pdf_pages
    for d in load_manifest()["documents"]:
        if d["source"] == "manual":
            continue
        sections, pages, toc = parse_document(d["doc_id"], read_pdf_pages(RAW_DIR / d["filename"]))
        assert len(pages) == d["pages"]
        chunks = chunk_sections(sections)
        text = " ".join(c.text for c in chunks)
        for junk in ("VerDate", "</GPH>", "PROD with CFR", "Enforc. Network, Treas.", "Edition) \u00a7"):
            assert junk not in text, (d["doc_id"], junk)
        if toc:  # every TOC entry has a heading; the PDF's TOC lacks one line for 1010.663
            assert set(toc) <= {s.section for s in sections}
    sections, _, _ = parse_document("cfr31_1020", read_pdf_pages(
        RAW_DIR / "CFR-2025-title31-vol3-part1020.pdf"))
    assert "1020.320" in {s.section for s in sections}
