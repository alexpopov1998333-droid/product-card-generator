"""Проверки ретраев и свода пропусков."""
from __future__ import annotations

import core


PRODUCT = {
    "name": "Манометр ДМ 02",
    "manufacturer": "",
    "specs": {"диапазон измерений": "0–1,6 МПа"},
    "description": "",
}


def test_title_retry_called_when_too_long(monkeypatch):
    long_title = "Очень длинный заголовок который не влезает в лимит семьдесят символов вообще"
    short_title = "Манометр ДМ 02"
    calls = []

    def fake(cfg, product, hint=""):
        calls.append(hint)
        if hint and "seo_title" in hint:
            return {"seo_title": short_title, "seo_description": "д" * 900, "specs": [],
                    "search_queries": ["a", "b", "c"], "commercial_block": "б", "missing_fields": []}
        return {"seo_title": long_title, "seo_description": "д" * 900, "specs": [],
                "search_queries": ["a", "b", "c"], "commercial_block": "б", "missing_fields": []}

    monkeypatch.setattr(core, "call_llm", fake)
    res = core.generate(core.Product.from_dict(PRODUCT), core.LlmConfig(api_key="x"))
    assert res["seo_title"] == short_title
    assert len(calls) == 2


def test_missing_merged_without_duplicates():
    product = core.Product.from_dict(PRODUCT)
    merged = core.merge_missing(
        ["диапазон измерения (нижний предел)", "материал корпуса", "производитель"], product
    )
    assert len(merged) <= 8
    assert sum(1 for m in merged if "диапазон" in m) == 1
    assert "производитель" in merged
