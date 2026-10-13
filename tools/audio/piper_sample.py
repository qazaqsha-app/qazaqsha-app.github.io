"""Пробный образец озвучки движком Piper (модель ISSAI kk_KZ, 6 дикторов).

Запуск (модель и piper-tts ставятся отдельно, в репозиторий не кладутся):

    python tools/audio/piper_sample.py путь/к/kk_KZ-issai-high.onnx

Пишет WAV и страницу прослушивания в audio-sample/ (папка в .gitignore).
Открывать через локальный сервер: http://localhost:5599/audio-sample/

Модель: https://huggingface.co/rhasspy/piper-voices (kk/kk_KZ/issai/high),
датасет KazakhTTS / KazakhTTS2, ISSAI, лицензия CC-BY-4.0.
"""
import html
import io
import os
import sys
import wave

from piper import PiperVoice, SynthesisConfig

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "audio-sample")

SPEAKERS = {
    0: "M2 · мужской",
    1: "Iseke · мужской",
    2: "F3 · женский",
    3: "Raya · женский",
    4: "F1 · женский",
    5: "F2 · женский",
}

PHRASE = "Сәлеметсіз бе! Менің атым — Данияр. Мен Алматыданмын."

# Слова из тем 0.1–0.2: обе гармонии и все буквы, которых нет в русском.
WORDS = [
    ("бала", "ребёнок"), ("әже", "бабушка"), ("қала", "город"), ("көше", "улица"),
    ("үй", "дом"), ("тау", "гора"), ("кітап", "книга"), ("мектеп", "школа"),
    ("дос", "друг"), ("ел", "страна"), ("адам", "человек"), ("қыз", "девочка"),
    ("мұғалім", "учитель"), ("жұмыс", "работа"), ("дүкен", "магазин"),
    ("сағат", "час"), ("ұл", "сын"), ("іні", "младший брат"), ("көл", "озеро"),
    ("күн", "день, солнце"),
]

# Диалог из темы 0.4 (тот же текст, что в приложении).
DIALOG = [
    ("Айгүл", "Сәлеметсіз бе!"),
    ("Данияр", "Сәлеметсіз бе! Менің атым — Данияр."),
    ("Айгүл", "Менің атым — Айгүл. Мен Алматыданмын."),
    ("Данияр", "Мен де Алматыданмын. Танысқаныма қуаныштымын!"),
    ("Айгүл", "Рақмет. Сау болыңыз!"),
]

# Где синтез обычно ошибается: числа, даты, спорная форма из 2.5.
HARD = [
    "Бүгін бесінші наурыз.",
    "Сағат жеті жарым.",
    "Біз Астанаға барғанмыз.",
    "Біз Астанаға барғанбыз.",
]


def synth(voice, text, path, speaker, length=1.0, pad=0.15):
    """Пишет WAV с тишиной по краям: браузер съедает начало коротких
    звуков, а одиночное слово без паузы звучит оборванным."""
    cfg = SynthesisConfig(speaker_id=speaker, length_scale=length)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        voice.synthesize_wav(text, w, syn_config=cfg)
    buf.seek(0)
    with wave.open(buf, "rb") as r:
        params, frames = r.getparams(), r.readframes(r.getnframes())
    silence = bytes(1) * (int(params.framerate * pad) * params.sampwidth * params.nchannels)
    with wave.open(path, "wb") as w:
        w.setparams(params)
        w.writeframes(silence + frames + silence)


def word_text(w):
    # Одиночное слово Piper комкает до 0,2–0,3 с. Точка даёт законченную
    # интонацию, а замедление ×1.3 — слышимые гласные.
    return w + "."


def main():
    model = sys.argv[1]
    female = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    male = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    os.makedirs(OUT, exist_ok=True)
    voice = PiperVoice.load(model)
    rows = []

    rows.append("<h2>1. Одна фраза — шесть дикторов</h2>")
    rows.append("<p class=kk>" + html.escape(PHRASE) + "</p>")
    for sid, name in SPEAKERS.items():
        f = "phrase-%d.wav" % sid
        synth(voice, PHRASE, os.path.join(OUT, f), sid)
        rows.append('<div class=row><span>%d · %s</span><audio controls preload=none src="%s"></audio></div>' % (sid, name, f))

    rows.append("<h2>2. Двадцать слов — женский и мужской голос</h2>")
    for i, (w, tr) in enumerate(WORDS):
        a, b = "w%02d-f.wav" % i, "w%02d-m.wav" % i
        synth(voice, word_text(w), os.path.join(OUT, a), female, 1.3)
        synth(voice, word_text(w), os.path.join(OUT, b), male, 1.3)
        rows.append('<div class=row><span class=kk>%s</span> <span class=tr>%s</span>'
                    '<audio controls preload=none src="%s"></audio><audio controls preload=none src="%s"></audio></div>'
                    % (html.escape(w), html.escape(tr), a, b))

    rows.append("<h2>3. Диалог по ролям — обычная скорость и медленно</h2>")
    for i, (who, line) in enumerate(DIALOG):
        sid = female if who == "Айгүл" else male
        a, b = "d%d.wav" % i, "d%d-slow.wav" % i
        synth(voice, line, os.path.join(OUT, a), sid)
        synth(voice, line, os.path.join(OUT, b), sid, 1.35)
        rows.append('<div class=row><span><b>%s:</b> <span class=kk>%s</span></span>'
                    '<audio controls preload=none src="%s"></audio><audio controls preload=none src="%s"></audio></div>'
                    % (who, html.escape(line), a, b))

    rows.append("<h2>4. Трудные места</h2>")
    for i, line in enumerate(HARD):
        f = "h%d.wav" % i
        synth(voice, line, os.path.join(OUT, f), female)
        rows.append('<div class=row><span class=kk>%s</span><audio controls preload=none src="%s"></audio></div>'
                    % (html.escape(line), f))

    page = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Образец озвучки · Piper</title>
<style>body{font-family:system-ui,sans-serif;max-width:760px;margin:0 auto;padding:16px;line-height:1.5}
.row{display:flex;flex-wrap:wrap;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid #e5e5e5}
.row>span:first-child{min-width:170px}.kk{font-weight:600}.tr{color:#777;font-size:.85em;min-width:110px}
audio{height:34px;max-width:100%%}</style>
<h1>Образец озвучки · Piper, модель ISSAI</h1>
<p>Женский голос — диктор %d, мужской — диктор %d. В разделе 2 слева женский, справа мужской.
В разделе 3 справа замедленная версия (×1.35).</p>
%s
<p style="color:#777;font-size:.85em">Модель Piper kk_KZ-issai-high, датасет KazakhTTS/KazakhTTS2 (ISSAI), CC-BY-4.0.</p>
""" % (female, male, "\n".join(rows))
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(page)
    print("ok", OUT)


if __name__ == "__main__":
    main()
