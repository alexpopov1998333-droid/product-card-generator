"""Генерация черновика карточки товара по данным прибора.

Ядро без веб-слоя: сборка промпта, вызов LLM через OpenAI-совместимый API,
проверка ответа на выдуманные характеристики.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

TITLE_MAX = 70
DESC_MIN, DESC_MAX = 800, 1200

# Группы характеристик, которые для прибора важны. Если во входе их нет,
# это попадает в missing_fields, чтобы контент-менеджер видел пробел.
IMPORTANT_GROUPS = {
    "диапазон измерения": ["диапазон", "предел измерен"],
    "класс точности": ["класс точности", "погрешность"],
    "питание": ["питание", "напряжение питания"],
    "выходной сигнал или интерфейс": ["выходной сигнал", "интерфейс", "протокол", "выход"],
    "корпус и защита": ["корпус", "степень защиты", "ip"],
}

SYSTEM_PROMPT = """Ты пишешь карточки товаров для B2B-каталога контрольно-измерительных приборов.

Жёсткие правила:
1. Бери факты только из входных данных. Ничего не додумывай: ни диапазоны, ни классы точности, ни материалы, ни страну, ни аналоги.
2. Если характеристики нет во входе, не добавляй её в specs. Назови её в missing_fields.
3. seo_title до 70 символов, по делу, без слова "купить" и без воды.
4. seo_description от 800 до 1200 символов. Только факты из входа, без рекламных обещаний и без выдуманных цифр.
5. search_queries от 3 до 5 штук, живые формулировки покупателя, без брендов, которых нет во входе.
6. commercial_block 2-4 предложения: кому подходит прибор и какие задачи решает, опираясь только на вход.
7. Пиши по-русски, технически точно, без канцелярита.

Верни строго JSON без пояснений:
{"seo_title": "...", "seo_description": "...", "specs": [{"name": "...", "value": "..."}],
 "search_queries": ["..."], "commercial_block": "...", "missing_fields": ["..."]}"""


@dataclass
class Product:
    name: str
    manufacturer: str = ""
    specs: dict[str, str] = field(default_factory=dict)
    description: str = ""

    @classmethod
    def from_dict(cls, raw: Any) -> "Product":
        if not isinstance(raw, dict):
            raise ValueError("на входе должен быть объект JSON")
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ValueError("поле name обязательно")
        specs_raw = raw.get("specs")
        if specs_raw is None:
            specs_raw = {}
        if not isinstance(specs_raw, dict):
            raise ValueError("поле specs должно быть объектом ключ-значение")
        specs = {str(k).strip(): str(v).strip() for k, v in specs_raw.items()
                 if str(k).strip() and str(v).strip()}
        return cls(
            name=name,
            manufacturer=str(raw.get("manufacturer") or "").strip(),
            specs=specs,
            description=str(raw.get("description") or "").strip(),
        )

    def as_prompt_json(self) -> str:
        return json.dumps(
            {"name": self.name, "manufacturer": self.manufacturer,
             "specs": self.specs, "description": self.description},
            ensure_ascii=False, indent=1,
        )

    def source_text(self) -> str:
        parts = [self.name, self.manufacturer, self.description, *self.specs.keys(), *self.specs.values()]
        return " ".join(parts).lower().replace("ё", "е")


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower().replace("ё", "е")


@dataclass
class LlmConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    timeout: float = 90.0

    @classmethod
    def from_env(cls) -> "LlmConfig":
        key = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or ""
        if not key:
            raise RuntimeError("не задан LLM_API_KEY или DEEPSEEK_API_KEY")
        return cls(
            api_key=key,
            base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/"),
            model=os.getenv("LLM_MODEL", "deepseek-chat"),
            timeout=float(os.getenv("LLM_TIMEOUT", "90")),
        )


class LlmError(RuntimeError):
    pass


def call_llm(cfg: LlmConfig, product: Product, hint: str = "") -> dict:
    user = (
        "Данные прибора:\n" + product.as_prompt_json()
        + ("\n\nПоправка к предыдущему ответу: " + hint if hint else "")
    )
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    try:
        with httpx.Client(timeout=cfg.timeout) as client:
            r = client.post(
                cfg.base_url + "/chat/completions",
                headers={"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"},
                json=payload,
            )
    except httpx.HTTPError as exc:
        raise LlmError(f"сеть или таймаут при вызове LLM: {exc}") from exc
    if r.status_code != 200:
        raise LlmError(f"LLM вернула {r.status_code}: {r.text[:300]}")
    try:
        content = r.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as exc:
        raise LlmError("неожиданный формат ответа LLM") from exc
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LlmError(f"LLM вернула не JSON: {content[:200]}") from exc
    if not isinstance(data, dict):
        raise LlmError("LLM вернула JSON не объектом")
    return data


def _value_supported(value: str, source: str) -> bool:
    """Проверяет, что значение есть во входных данных.

    Сравниваем по числовым и буквенно-цифровым токенам: "0…0,4 МПа" проходит,
    если во входе есть 0,4. Выдуманное "1,0 МПа" не проходит.
    """
    tokens = re.findall(r"[a-zа-я]*\d+[a-zа-я0-9]*", _norm(value))
    if not tokens:
        # текстовое значение без цифр: требуем вхождения хотя бы значимого слова
        words = [w for w in re.findall(r"[a-zа-я]{4,}", _norm(value))]
        return bool(words) and any(w in source for w in words)
    return all(t in source for t in tokens)


def guard_specs(raw_specs: Any, product: Product) -> tuple[list[dict], list[str]]:
    """Оставляет только подтверждённые характеристики, остальное помечает.

    Возвращает (specs, warnings). Выдуманное значение не выбрасывается молча,
    а заменяется на "данных нет", чтобы это было видно в карточке.
    """
    source = product.source_text()
    out: list[dict] = []
    warnings: list[str] = []
    if isinstance(raw_specs, dict):
        raw_specs = [{"name": k, "value": v} for k, v in raw_specs.items()]
    if not isinstance(raw_specs, list):
        raw_specs = []
    for item in raw_specs:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        value = str(item.get("value") or "").strip()
        if not name:
            continue
        if not value or _norm(value) in ("данных нет", "нет данных", "не указано"):
            out.append({"name": name, "value": "данных нет"})
            continue
        if _value_supported(value, source):
            out.append({"name": name, "value": value})
        else:
            warnings.append(f"выдуманное значение убрано: {name} = {value}")
            out.append({"name": name, "value": "данных нет"})
    have = {_norm(i["name"]) for i in out}
    for key, value in product.specs.items():
        if _norm(key) not in have:
            out.append({"name": key, "value": value})
    return out, warnings


def missing_groups(product: Product) -> list[str]:
    source = product.source_text()
    miss = []
    for label, keys in IMPORTANT_GROUPS.items():
        if not any(k in source for k in keys):
            miss.append(label)
    return miss


def fit_title(title: str) -> tuple[str, list[str]]:
    title = re.sub(r"\s+", " ", title).strip()
    warnings = []
    if len(title) > TITLE_MAX:
        cut = title[:TITLE_MAX].rsplit(" ", 1)[0].rstrip(" ,.;")
        warnings.append(f"заголовок обрезан до {len(cut)} символов")
        title = cut
    return title, warnings


def check_description(text: str) -> list[str]:
    n = len(text.strip())
    if n < DESC_MIN:
        return [f"описание короче нормы: {n} символов вместо {DESC_MIN}-{DESC_MAX}"]
    if n > DESC_MAX:
        return [f"описание длиннее нормы: {n} символов вместо {DESC_MIN}-{DESC_MAX}"]
    return []


def merge_missing(model_items: list[str], product: Product) -> list[str]:
    """Сводит пропуски от модели с нашими группами: без дублей, не больше восьми."""
    out = missing_groups(product)
    stems = [set(re.findall(r"[а-я]{5,}", g.lower())) for g in out]
    for item in model_items:
        tokens = set(re.findall(r"[а-я]{5,}", item.lower()))
        if any(tokens & stem for stem in stems):
            continue
        out.append(item)
        stems.append(tokens)
    return out[:8]


def generate(product: Product, cfg: LlmConfig) -> dict:
    started = time.time()
    data = call_llm(cfg, product)
    warnings: list[str] = []

    description = str(data.get("seo_description") or "").strip()
    desc_warnings = check_description(description)
    if desc_warnings:
        hint = (
            f"описание вышло {len(description)} символов, нужно от {DESC_MIN} до {DESC_MAX}. "
            "Расширь или сократи только фактами из входа, ничего не выдумывай."
        )
        try:
            data = call_llm(cfg, product, hint)
            description = str(data.get("seo_description") or "").strip()
            desc_warnings = check_description(description)
        except LlmError as exc:
            warnings.append(f"повторная генерация не удалась: {exc}")
    warnings.extend(desc_warnings)

    title = re.sub(r"\s+", " ", str(data.get("seo_title") or "")).strip()
    if len(title) > TITLE_MAX:
        try:
            retry = call_llm(
                cfg, product,
                f"seo_title был {len(title)} символов, нужно не больше {TITLE_MAX}. "
                "Сократи, сохрани модель и самую сильную характеристику.",
            )
            candidate = re.sub(r"\s+", " ", str(retry.get("seo_title") or "")).strip()
            if candidate and len(candidate) <= TITLE_MAX:
                title = candidate
        except LlmError as exc:
            warnings.append(f"повторный заголовок не удался: {exc}")
    title, title_warnings = fit_title(title)
    warnings.extend(title_warnings)

    specs, spec_warnings = guard_specs(data.get("specs"), product)
    warnings.extend(spec_warnings)

    queries = data.get("search_queries") or []
    if isinstance(queries, str):
        queries = [queries]
    queries = [str(q).strip() for q in queries if str(q).strip()][:5]
    if len(queries) < 3:
        warnings.append("поисковых запросов меньше трёх")

    reported_missing = data.get("missing_fields") or []
    if isinstance(reported_missing, str):
        reported_missing = [reported_missing]
    model_missing = [str(m).strip() for m in reported_missing if str(m).strip()]
    missing = merge_missing(model_missing, product)

    return {
        "seo_title": title,
        "seo_description": description,
        "specs": specs,
        "search_queries": queries,
        "commercial_block": str(data.get("commercial_block") or "").strip(),
        "missing_fields": missing,
        "warnings": warnings,
        "meta": {
            "model": cfg.model,
            "seconds": round(time.time() - started, 1),
            "title_len": len(title),
            "description_len": len(description),
        },
    }
