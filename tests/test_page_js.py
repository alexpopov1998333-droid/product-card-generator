"""Проверка, что встроенный скрипт страницы остаётся валидным JS."""
from __future__ import annotations

import re
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from app import app


@pytest.mark.skipif(shutil.which("node") is None, reason="нет node")
def test_page_script_parses(tmp_path):
    html = TestClient(app).get("/").text
    js = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
    path = tmp_path / "page.js"
    path.write_text(js, encoding="utf-8")
    check = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True)
    assert check.returncode == 0, check.stderr
