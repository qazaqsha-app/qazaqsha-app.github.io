"""Озвучка курса движком Piper (модель ISSAI kk_KZ) — файлы в audio/.

Запуск из корня проекта питоном из venv с piper-tts, onnx, soundfile, numpy:

    .tts/venv/Scripts/python tools/audio/build_audio.py            # только новые
    .tts/venv/Scripts/python tools/audio/build_audio.py --force    # всё заново
    .tts/venv/Scripts/python tools/audio/build_audio.py --review кітап дос ел

Что озвучивается — всё казахское, что берётся из index.html: слова наборов
«Слова N», примеры «Әліппе», реплики диалогов, тексты, фразы «Сөйлесу», ходы сцен. Имя файла —
хеш текста (audioKey в index.html считает тот же), поэтому манифест не нужен:
поменял строку в теме — у неё новый файл, старый удалится при следующем
прогоне.

Модель: https://huggingface.co/rhasspy/piper-voices (kk/kk_KZ/issai/high),
лежит в .tts/. Датасет KazakhTTS / KazakhTTS2, ISSAI, лицензия CC-BY-4.0.
"""
import html
import io
import os
import re
import sys

import numpy as np
import soundfile as sf
from piper import PiperVoice, SynthesisConfig

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
MODEL = os.path.join(ROOT, ".tts", "kk_KZ-issai-high.onnx")
OUT = os.path.join(ROOT, "audio")
REVIEW = os.path.join(ROOT, "audio-sample", "review")

# Дикторы модели. Основные выбраны на слух 2026-10-13: женский F3, мужской
# Iseke. Третий голос нужен ролям вроде «Сатушы», чтобы в диалоге с
# Айгүл или Данияром не звучали два одинаковых голоса.
FEMALE, MALE, ROLE_F, ROLE_M = 2, 1, 3, 0
SPEAKER = {
    "Айгүл": FEMALE, "Данияр": MALE, "Мейрам": MALE,
    "Сатушы": ROLE_F, "Көрші": ROLE_F, "Кеңсе": ROLE_F, "Даяшы": ROLE_F,
    "Жолаушы": ROLE_M, "Жүргізуші": ROLE_M,
}

# Чего синтез не прочтёт сам: числа цифрами.
SAY = {
    "1691 метр": "бір мың алты жүз тоқсан бір метр",
}

LENGTH = 1.35      # на обычной скорости голос «торопится и быстро дышит»
NOISE = 0.5        # по умолчанию 0.667; ниже — меньше шума и придыхания
NOISE_W = 0.8      # разброс длительностей оставлен: из него выбираем лучшую попытку
TRIES_WORD = 6
TRIES_LINE = 3
SR = 22050
SKIP = set("^$.,:;!?—–-«»\"' ˈˌ")


def norm(s):
    """Ключ строки. Обязан совпадать с normAudio() в index.html:
    textContent без разметки, пробелы схлопнуты."""
    return re.sub(r"\s+", " ", s).strip()


def audio_key(text):
    """Два 32-битных FNV-1a по UTF-16 — как audioKey() в index.html."""
    units = text.encode("utf-16-le")
    codes = [units[i] | (units[i + 1] << 8) for i in range(0, len(units), 2)]
    h1, h2 = 0x811C9DC5, 0x050C5D1F
    for c in codes:
        h1 = ((h1 ^ c) * 0x01000193) & 0xFFFFFFFF
    for c in reversed(codes):
        h2 = ((h2 ^ c) * 0x01000193) & 0xFFFFFFFF
    return "%08x%08x" % (h1, h2)


def strip_tags(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s))


def collect(src):
    """[(ключ-текст, текст для синтеза, диктор, вид)] без повторов."""
    units, seen = [], set()

    def add(text, speak, sid, kind):
        k = norm(text)
        if not k or k in seen:
            return
        seen.add(k)
        for a, b in SAY.items():
            speak = speak.replace(a, b)
        units.append((k, norm(speak), sid, kind))

    for ws in re.findall(r'<div class="wordset">(.*?)\n\s*</div>', src, re.S):
        for w in re.findall(r'<div><span class="w[^"]*">(.*?)</span>', ws):
            w = strip_tags(w)
            # Одиночное слово без точки Piper комкает: точка даёт законченную
            # интонацию.
            add(w, w + ".", FEMALE, "word")
    # Слова-примеры экрана «Әліппе»: не наборы, в Сөздік не идут.
    for w in re.findall(r'<span class="abc-kk">(.*?)</span>', src):
        w = strip_tags(w)
        add(w, w + ".", FEMALE, "word")
    for who, line in re.findall(r'<p class="dline"><span class="speaker">([^<:]*):</span>(.*?)</p>', src, re.S):
        add(strip_tags(line), strip_tags(line), SPEAKER.get(who, MALE), "line")
    for p in re.findall(r'<p class="storytext">(.*?)</p>', src, re.S):
        add(strip_tags(p), strip_tags(re.sub(r"<br\s*/?>", " ", p)), MALE, "story")
    for p in re.findall(r'<div class="ph-kk">(.*?)</div>', src, re.S):
        add(strip_tags(p), strip_tags(p), MALE, "phrase")
    scenes = src[src.index("var SCENES ="):]
    scenes = scenes[:scenes.index("];")]
    for who, kk in re.findall(r'\{ who:"([^"]*)", kk:"([^"]*)"', scenes):
        add(kk, kk, SPEAKER.get(who, MALE), "scene")
    for kk in re.findall(r'\{ kk:"([^"]*)"', scenes):
        add(kk, kk, MALE, "scene")
    return units


def best_chunks(voice, text, sid, tries):
    """Синтез по предложениям, из нескольких попыток — лучшая для каждого.

    Последний звук модель часто глотает (`кітап` без «п», `ел`): средняя
    длительность у неё короткая, и только разброс иногда даёт нормальную.
    Поэтому мерим по выравниванию длину последней фонемы и берём попытку,
    где она дольше, с небольшой поправкой на общую длину."""
    cfg = SynthesisConfig(speaker_id=sid, length_scale=LENGTH,
                          noise_scale=NOISE, noise_w_scale=NOISE_W)
    best = None
    for _ in range(tries):
        chunks = list(voice.synthesize(text, cfg, include_alignments=True))
        if best is None:
            best = [None] * len(chunks)
        for i, ch in enumerate(chunks[:len(best)]):
            al = ch.phoneme_alignments or []
            real = [j for j, p in enumerate(al) if p.phoneme not in SKIP]
            last = al[real[-1]].num_samples / SR if real else 0
            score = min(last, 0.16) + 0.1 * len(ch.audio_float_array) / SR
            if best[i] is None or score > best[i][0]:
                best[i] = (score, ch, al, real)
    return best


def trim(ch, al, real):
    """Срез по выравниванию: от начала первой фонемы до конца последней
    плюс запас. Хвост после слова — это «.» и «$», там и сидят придыхание
    и шум. Края гасятся плавно, чтобы не было щелчка."""
    a = ch.audio_float_array.astype(np.float32)
    if real:
        pos = np.cumsum([0] + [p.num_samples for p in al])
        start = max(0, pos[real[0]] - int(0.03 * SR))
        end = min(len(a), pos[real[-1] + 1] + int(0.08 * SR))
        a = a[start:end]
    fi, fo = int(0.006 * SR), int(0.05 * SR)
    a[:fi] *= np.linspace(0, 1, fi)
    a[-fo:] *= np.linspace(1, 0, fo) ** 2
    return a


def render(voice, speak, sid, kind):
    tries = TRIES_WORD if kind == "word" else TRIES_LINE
    parts, gap = [], np.zeros(int((0.45 if kind == "story" else 0.3) * SR), np.float32)
    for _, ch, al, real in best_chunks(voice, speak, sid, tries):
        if parts:
            parts.append(gap)
        parts.append(trim(ch, al, real))
    # Тишина по краям: браузер съедает начало коротких звуков.
    return np.concatenate([np.zeros(int(0.12 * SR), np.float32)] + parts +
                          [np.zeros(int(0.1 * SR), np.float32)])


def write_mp3(path, a):
    peak = float(np.max(np.abs(a))) or 1.0
    a = a * (0.89 / peak)
    sf.write(path, a, SR, format="MP3", bitrate_mode="VARIABLE", compression_level=0.6)


def review(voice, words):
    """Страница сравнения: старый способ (как в образце) и новый конвейер."""
    os.makedirs(REVIEW, exist_ok=True)
    rows = []
    for i, w in enumerate(words):
        cells = []
        for sid, name in ((FEMALE, "ж"), (MALE, "м")):
            old = "%d-%s-old.wav" % (i, name)
            buf = io.BytesIO()
            import wave
            with wave.open(buf, "wb") as wf:
                voice.synthesize_wav(w + ".", wf, syn_config=SynthesisConfig(speaker_id=sid, length_scale=1.3))
            open(os.path.join(REVIEW, old), "wb").write(buf.getvalue())
            new = "%d-%s-new.mp3" % (i, name)
            write_mp3(os.path.join(REVIEW, new), render(voice, w + ".", sid, "word"))
            cells.append('<td>%s <audio controls preload=none src="%s"></audio></td>'
                         '<td><audio controls preload=none src="%s"></audio></td>' % (name, old, new))
        rows.append("<tr><th>%s</th>%s</tr>" % (html.escape(w), "".join(cells)))
    page = ("<!doctype html><meta charset=utf-8><title>Было и стало · Piper</title>"
            "<style>body{font-family:system-ui;padding:16px}td,th{padding:4px 8px;text-align:left}"
            "audio{height:32px;width:170px}</style><h1>Было → стало</h1>"
            "<table><tr><th></th><th>женский, было</th><th>стало</th><th>мужской, было</th><th>стало</th></tr>"
            + "\n".join(rows) + "</table>")
    open(os.path.join(REVIEW, "index.html"), "w", encoding="utf-8").write(page)
    print("review:", REVIEW)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    voice = PiperVoice.load(MODEL, include_alignments=True)
    if args and args[0] == "--review":
        review(voice, args[1:] or ["кітап", "дос", "ел", "көл", "қыз", "әже", "үй"])
        return
    force = "--force" in args
    src = open(os.path.join(ROOT, "index.html"), encoding="utf-8").read()
    units = collect(src)
    keys = {}
    for k, *_ in units:
        h = audio_key(k)
        if h in keys:
            sys.exit("коллизия хеша: %r и %r" % (keys[h], k))
        keys[h] = k
    os.makedirs(OUT, exist_ok=True)
    made = 0
    for n, (k, speak, sid, kind) in enumerate(units, 1):
        path = os.path.join(OUT, audio_key(k) + ".mp3")
        if os.path.exists(path) and not force:
            continue
        write_mp3(path, render(voice, speak, sid, kind))
        made += 1
        if made % 50 == 0:
            print(n, "/", len(units))
    stale = [f for f in os.listdir(OUT) if f.endswith(".mp3") and f[:-4] not in keys]
    for f in stale:
        os.remove(os.path.join(OUT, f))
    size = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
    kinds = {}
    for u in units:
        kinds[u[3]] = kinds.get(u[3], 0) + 1
    print("строк %d %s, новых %d, удалено %d, всего %.1f МБ"
          % (len(units), kinds, made, len(stale), size / 1e6))


if __name__ == "__main__":
    main()
