"""Проверки ядра без живых вызовов LLM."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import core
from app import app


PRODUCT = {
    "name": "Манометр ДМ 02",
    "manufacturer": "Манотомь",
    "specs": {"диапазон измерений": "0–1,6 МПа", "класс точности": "1,5"},
    "description": "Прибор для измерения избыточного давления.",
}


def fake_llm(payload: dict):
    def _call(cfg, product, hint=""):
        return json.loads(json.dumps(payload))
    return _call


def test_invented_spec_marked_not_silently_dropped(monkeypatch):
    payload = {
        "seo_title": "Манометр ДМ 02",
        "seo_description": "текст",
        "specs": [
            {"name": "диапазон измерений", "value": "0–1,6 МПа"},
            {"name": "материал корпуса", "value": "нержавеющая сталь"},
        ],
        "search_queries": ["манометр дм 02"],
        "commercial_block": "блок",
        "missing_fields": [],
    }
    monkeypatch.setattr(core, "call_llm", fake_llm(payload))
    res = core.generate(core.Product.from_dict(PRODUCT), core.LlmConfig(api_key="x"))
    specs = {s["name"]: s["value"] for s in res["specs"]}
    assert specs["материал корпуса"] == "данных нет"
    assert any("выдуманное значение" in w for w in res["warnings"])
    assert specs["диапазон измерений"] == "0–1,6 МПа"


def test_input_specs_not_lost_when_model_skips_them(monkeypatch):
    payload = {
        "seo_title": "Манометр ДМ 02",
        "seo_description": "текст",
        "specs": [{"name": "диапазон измерений", "value": "0–1,6 МПа"}],
        "search_queries": ["a", "b", "c"],
        "commercial_block": "блок",
        "missing_fields": [],
    }
    monkeypatch.setattr(core, "call_llm", fake_llm(payload))
    res = core.generate(core.Product.from_dict(PRODUCT), core.LlmConfig(api_key="x"))
    specs = {s["name"]: s["value"] for s in res["specs"]}
    assert specs["класс точности"] == "1,5"


def test_title_trimmed_to_70(monkeypatch):
    payload = {
        "seo_title": "Очень длинный заголовок карточки товара который точно не влезет в семьдесят символов",
        "seo_description": "т",
        "specs": [],
        "search_queries": ["a", "b", "c"],
        "commercial_block": "б",
        "missing_fields": [],
    }
    monkeypatch.setattr(core, "call_llm", fake_llm(payload))
    res = core.generate(core.Product.from_dict(PRODUCT), core.LlmConfig(api_key="x"))
    assert len(res["seo_title"]) <= core.TITLE_MAX


def test_short_description_warned_after_retry(monkeypatch):
    payload = {
        "seo_title": "Манометр",
        "seo_description": "коротко",
        "specs": [],
        "search_queries": ["a", "b", "c"],
        "commercial_block": "б",
        "missing_fields": [],
    }
    monkeypatch.setattr(core, "call_llm", fake_llm(payload))
    res = core.generate(core.Product.from_dict(PRODUCT), core.LlmConfig(api_key="x"))
    assert any("описание короче нормы" in w for w in res["warnings"])


def test_missing_required_field_reported():
    product = core.Product.from_dict({"name": "Прибор", "specs": {}})
    missing = core.missing_groups(product)
    assert "диапазон измерения" in missing
    assert "класс точности" in missing


@pytest.mark.parametrize("bad", [{"specs": {}}, {"name": "  "}, "строка", {"name": "x", "specs": []}])
def test_invalid_input_raises(bad):
    with pytest.raises(ValueError):
        core.Product.from_dict(bad)


def test_api_rejects_invalid_json_body(monkeypatch):
    client = TestClient(app)
    r = client.post("/api/generate", json={"specs": {}})
    assert r.status_code == 422


def test_examples_available():
    client = TestClient(app)
    for num in (1, 2):
        r = client.get(f"/api/example/{num}")
        assert r.status_code == 200
        assert r.json()["name"] and r.json()["specs"]
    assert client.get("/api/example/3").status_code == 404


def test_api_returns_card(monkeypatch):
    class DummyConfig:
        model = "test"
        api_key = "x"

        @classmethod
        def from_env(cls):
            return cls()

    monkeypatch.setattr(core, "LlmConfig", DummyConfig)
    monkeypatch.setattr(core, "call_llm", fake_llm({
        "seo_title": "Манометр ДМ 02",
        "seo_description": "д" * 900,
        "specs": [{"name": "класс точности", "value": "1,5"}],
        "search_queries": ["a", "b", "c"],
        "commercial_block": "блок",
        "missing_fields": [],
    }))
    client = TestClient(app)
    r = client.post("/api/generate", json=PRODUCT)
    assert r.status_code == 200
    body = r.json()
    assert body["seo_title"] == "Манометр ДМ 02"
    assert body["specs"]
