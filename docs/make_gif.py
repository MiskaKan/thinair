"""Render the thinair hero GIF: a real REPL session with a rusty Hilux."""
import re
from PIL import Image, ImageDraw, ImageFont

W, PAD, LH, TITLE_H = 830, 24, 20, 34
FONT = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14, index=0)
CW = FONT.getlength("M")

BG, CHROME, BAR = "#0f1117", "#222634", "#171a22"
FG, DIM, COMMENT = "#c8cdd8", "#7a8394", "#5b6272"
KEY, STR, NUM, FUNC = "#c792ea", "#7fd693", "#f2a05a", "#64b5ef"
PROMPT, HEAD = "#62d38a", "#e6b3ff"

KEYWORDS = {"class", "def", "return", "if"}

# ("in", line) typed; ("out", line) printed; ("wait",) an inference beat
SESSION = [
    ("head", "# 1. Write the parts you are sure of"),
    ("in", ">>> class Car(Thing):"),
    ("in", "...     wheels = 4"),
    ("in", '...     def horn(self): return "beep"'),
    ("in", "..."),
    ("in", '>>> car = Car("a rusty 1990 Toyota Hilux, engine coughs, "'),
    ("in", '...           "radio stuck on a Finnish schlager station")'),
    ("in", ">>> car.wheels, car.horn()          # real code: free, certain"),
    ("out", "(4, 'beep')"),
    ("gap", ""),
    ("head", "# 2. Read a field nobody defined"),
    ("in", ">>> car.color                       # LLM-imagined on first read"),
    ("wait", ""),
    ("out", "'rusty red'                         # = Thing('rusty red', confidence=0.4)"),
    ("in", ">>> +car.color                      # + the value"),
    ("out", "'rusty red'"),
    ("in", ">>> ~car.color                      # ~ the probability"),
    ("out", "0.4"),
    ("gap", ""),
    ("head", "# 3. @ shapes a Thing — and can demand confidence"),
    ("in", ">>> price = car.resale_value_eur @ float     # typed: still a Thing"),
    ("wait", ""),
    ("in", ">>> +price, ~price"),
    ("out", "(1200.0, 0.15)"),
    ("in", ">>> +(price @ 0.9)                  # demand p >= 0.9"),
    ("out", "None                                # too unsure to flow"),
    ("gap", ""),
    ("head", "# 4. Call a method nobody wrote"),
    ("in", ">>> car.list_your_problems(returns=[str])"),
    ("wait", ""),
    ("out", "['engine coughs at cold start', 'heavy rust all over the body',"),
    ("out", " 'radio stuck on a Finnish schlager station']"),
]

CAPTION = "an LLM woven seamlessly into ordinary Python, where every guess knows how sure it is"


def spans(kind, line):
    if kind == "head":
        return [(line, HEAD)]
    if kind == "out":
        code, hash_, comment = line.partition("#")
        colour = (STR if code.lstrip().startswith(("'", '"', "[")) or
                  line.startswith(" ") else NUM)
        return [(code, colour)] + ([(hash_ + comment, COMMENT)] if hash_ else [])
    head, rest = line[:4], line[4:]
    out = [(head, PROMPT if head.startswith(">>>") else DIM)]
    i = 0
    for m in re.finditer(r'(#.*$)|("(?:[^"\\]|\\.)*")|(\b\d+\.?\d*\b)|(\b\w+\b)', rest):
        if m.start() > i:
            out.append((rest[i:m.start()], FG))
        tok = m.group(0)
        if m.group(1):
            out.append((tok, COMMENT))
        elif m.group(2):
            out.append((tok, STR))
        elif m.group(3):
            out.append((tok, NUM))
        elif tok in KEYWORDS:
            out.append((tok, KEY))
        elif rest[m.end():m.end() + 1] == "(":
            out.append((tok, FUNC))
        else:
            out.append((tok, FG))
        i = m.end()
    if i < len(rest):
        out.append((rest[i:], FG))
    return out


ROWS = sum(1 for k, _ in SESSION if k != "wait") + 2
H = TITLE_H + PAD + ROWS * LH + PAD


def frame(n, thinking=False):
    img = Image.new("RGB", (W, H), CHROME)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, TITLE_H], fill=BAR)
    for k, c in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        d.ellipse([18 + k * 20, 12, 28 + k * 20, 22], fill=c)
    d.text((W / 2 - 58, 9), "python — thinair", font=FONT, fill=DIM)
    d.rectangle([1, TITLE_H, W - 1, H - 1], fill=BG)

    y = TITLE_H + PAD
    for kind, line in SESSION[:n]:
        if kind in ("gap", "wait"):
            y += LH if kind == "gap" else 0
            continue
        x = PAD
        for text, colour in spans(kind, line):
            d.text((x, y), text, font=FONT, fill=colour)
            x += CW * len(text)
        y += LH
    if thinking:
        d.text((PAD, y), "…imagining", font=FONT, fill=DIM)
    if n >= len(SESSION):
        d.text((PAD, y + LH), CAPTION, font=FONT, fill=COMMENT)
    return img


frames, delays = [], []
for i, (kind, _) in enumerate(SESSION, start=1):
    if kind == "wait":
        for _ in range(2):
            frames.append(frame(i, thinking=True)); delays.append(420)
        continue
    frames.append(frame(i))
    delays.append({"in": 200, "out": 260, "head": 520}.get(kind, 200))
frames.append(frame(len(SESSION))); delays.append(4500)

master = frames[-1].quantize(colors=24, method=Image.MEDIANCUT)
frames = [f.quantize(palette=master, dither=Image.NONE) for f in frames]
frames[0].save("thinair.gif", save_all=True, append_images=frames[1:],
               duration=delays, loop=0, optimize=True, disposal=2)
print("frames:", len(frames), "canvas:", W, "x", H)
