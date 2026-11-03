"""Готовит примеры товаров из YML-выгрузки сайта.

    python scripts/examples_from_feed.py --vendor 58953 39432

Берёт offer по vendorCode из фида, открывает страницу товара, достаёт
описание из JSON-LD и таблицу характеристик. Кладёт input/product*.json.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

FEED = "https://www.souz-pribor.ru/yml/any/feed.xml"
SLUGS = {"58953": "mk110", "03869": "trm501"}


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def text(fragment: str) -> str:
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", fragment))).strip()


STOP = ("Отзывы", "Документы", "Похожие товары", "С этим товаром", "Вопросы", "Купить")


def body_description(page: str) -> str:
    """Текст из блока id="description": там реальное описание, а не SEO-заглушка."""
    i = page.find('id="description"')
    if i < 0:
        return ""
    text_ = text(page[i:i + 12000])
    text_ = text_.split("Описание", 2)[-1]
    for marker in STOP:
        text_ = text_.split(marker)[0]
    return text_.strip()[:2500]


def jsonld_description(page: str) -> str:
    for block in re.findall(r'<script type="application/ld\+json">(.*?)</script>', page, re.S):
        try:
            data = json.loads(block, strict=False)  # в фиде бывают сырые переводы строк
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("@type") == "Product":
            return str(data.get("description") or "").strip()
    return ""


def spec_table(page: str) -> dict[str, str]:
    """Характеристики со страницы товара.

    Часть страниц отдаёт одну таблицу с шапкой «Наименование / Значение»,
    остальные разбивают характеристики на несколько таблиц без шапки.
    """
    tables = re.findall(r"<table[^>]*>(.*?)</table>", page, re.S)
    named = [t for t in tables if "Наименование" in text(t)]
    specs: dict[str, str] = {}
    for t in (named or tables):
        for row in re.findall(r"<tr>(.*?)</tr>", t, re.S):
            cells = [text(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
            if len(cells) < 2:
                continue
            key, value = cells[0], cells[1]
            if not key or not value or key.lower() in ("наименование", "значение"):
                continue
            if len(key) > 90 or key.endswith("."):
                continue
            specs[key] = value
    return specs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feed", default=FEED)
    ap.add_argument("--vendor", nargs="+", required=True, help="vendorCode из выгрузки")
    ap.add_argument("--out", default="input")
    args = ap.parse_args()

    root = ET.fromstring(get(args.feed))
    offers = {o.findtext("vendorCode"): o for o in root.iter("offer")}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, vendor in enumerate(args.vendor, start=1):
        offer = offers.get(vendor)
        if offer is None:
            print(f"{vendor}: нет в выгрузке", file=sys.stderr)
            return 1
        url = offer.findtext("url") or ""
        page = get(url).decode("utf-8", "replace")
        slug = SLUGS.get(vendor, vendor)
        product = {
            "name": offer.findtext("name"),
            "manufacturer": offer.findtext("vendor") or "",
            "specs": spec_table(page),
            "description": body_description(page) or jsonld_description(page),
            "source_url": url,
            "price": offer.findtext("price"),
        }
        path = out_dir / f"product{i}_{slug}.json"
        path.write_text(json.dumps(product, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{path}: {len(product['specs'])} характеристик, описание {len(product['description'])} символов")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
