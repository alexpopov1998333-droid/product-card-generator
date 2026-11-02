"""Веб-прототип: форма и JSON API для генерации карточки товара."""
from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

import core

load_dotenv(Path(__file__).with_name(".env"))

app = FastAPI(title="AI-помощник для карточек товаров", version="0.1.0")

PAGE = """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>Карточка товара</title><style>
body{font-family:system-ui,Arial,sans-serif;margin:0;background:#f5f6f8;color:#1a1a1a}
.wrap{max-width:900px;margin:0 auto;padding:24px}
h1{font-size:22px;margin:0 0 4px}p.hint{color:#666;margin:0 0 20px;font-size:14px}
textarea{width:100%;min-height:320px;font-family:ui-monospace,Consolas,monospace;font-size:13px;
padding:12px;border:1px solid #ccc;border-radius:8px;background:#fff;box-sizing:border-box}
button{margin-top:12px;padding:10px 20px;border:0;border-radius:8px;background:#2456d6;color:#fff;font-size:15px;cursor:pointer}
button:disabled{opacity:.6;cursor:wait}
.card{background:#fff;border:1px solid #e2e4e8;border-radius:10px;padding:16px;margin-top:16px}
.card h2{font-size:15px;margin:0 0 8px;color:#2456d6}
table{border-collapse:collapse;width:100%;font-size:14px}td{padding:4px 8px;border-bottom:1px solid #eee;vertical-align:top}
td:first-child{color:#666;width:45%}
.warn{background:#fff7e6;border-color:#f0c36d}
.err{background:#fdecea;border-color:#e57373}
pre{white-space:pre-wrap;margin:0;font-family:inherit;font-size:14px}
</style></head><body><div class="wrap">
<h1>AI-помощник для подготовки товарных описаний</h1>
<p class="hint">Вставьте данные прибора в JSON и нажмите кнопку. Характеристики берутся только из входа.</p>
<textarea id="src">__SRC__</textarea>
<button id="go" onclick="run()">Сгенерировать карточку</button>
<div id="out"></div>
<script>
async function run(){
  const btn=document.getElementById('go'); btn.disabled=true; btn.textContent='Генерирую...';
  const out=document.getElementById('out'); out.innerHTML='';
  try{
    const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},
      body:document.getElementById('src').value});
    const d=await r.json();
    if(!r.ok){out.innerHTML='<div class="card err"><h2>Ошибка</h2><pre>'+esc(d.detail||JSON.stringify(d))+'</pre></div>';}
    else {out.innerHTML=render(d);}
  }catch(e){out.innerHTML='<div class="card err"><h2>Ошибка сети</h2><pre>'+esc(e)+'</pre></div>';}
  btn.disabled=false; btn.textContent='Сгенерировать карточку';
}
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function render(d){
  const specs=d.specs.map(s=>'<tr><td>'+esc(s.name)+'</td><td>'+esc(s.value)+'</td></tr>').join('');
  const q=d.search_queries.map(x=>'<li>'+esc(x)+'</li>').join('');
  const miss=(d.missing_fields||[]).map(x=>'<li>'+esc(x)+'</li>').join('');
  const warn=(d.warnings||[]).map(x=>'<li>'+esc(x)+'</li>').join('');
  return '<div class="card"><h2>SEO-заголовок</h2><pre>'+esc(d.seo_title)+'</pre></div>'+
   '<div class="card"><h2>SEO-описание</h2><pre>'+esc(d.seo_description)+'</pre></div>'+
   '<div class="card"><h2>Характеристики</h2><table>'+specs+'</table></div>'+
   '<div class="card"><h2>Поисковые запросы</h2><ul>'+q+'</ul></div>'+
   '<div class="card"><h2>Блок для менеджера</h2><pre>'+esc(d.commercial_block)+'</pre></div>'+
   (miss?'<div class="card warn"><h2>Не хватает данных</h2><ul>'+miss+'</ul></div>':'')+
   (warn?'<div class="card warn"><h2>Предупреждения проверки</h2><ul>'+warn+'</ul></div>':'')+
   '<div class="card"><h2>Метаданные</h2><pre>'+esc(JSON.stringify(d.meta))+'</pre></div>';
}
</script></div></body></html>"""

DEFAULT_SRC = Path(__file__).with_name("input").joinpath("product1_dm2010sg.json")


class GenerateRequest(BaseModel):
    name: str = Field(..., min_length=1)
    manufacturer: str = ""
    specs: dict[str, str] = Field(default_factory=dict)
    description: str = ""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    try:
        src = DEFAULT_SRC.read_text(encoding="utf-8")
    except OSError:
        src = json.dumps({"name": "", "manufacturer": "", "specs": {}, "description": ""},
                         ensure_ascii=False, indent=1)
    return PAGE.replace("__SRC__", src.replace("&", "&amp;").replace("<", "&lt;"))


@app.post("/api/generate")
def generate(payload: dict):
    try:
        product = core.Product.from_dict(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        cfg = core.LlmConfig.from_env()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    try:
        return core.generate(product, cfg)
    except core.LlmError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": f"внутренняя ошибка: {exc}"})


@app.get("/health")
def health():
    return {"ok": True}
