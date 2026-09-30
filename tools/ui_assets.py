#!/usr/bin/env python3
"""Builds the game's art under assets/ui/ from free sources.

Usage: tools/ui_assets.py <source dir>

The source dir holds the downloaded originals:
  acg/<Id>/<Id>_1K-JPG_Color.jpg   ambientCG materials (CC0)
  sky/<id>.jpg                     Poly Haven tonemapped HDRIs (CC0)
  fonts/...                        DejaVu (Bitstream Vera license), Barlow (OFL)

Everything else (the City 17 style skyline layers, the sprites, the menu
backdrop) is painted here from code, so the output is reproducible.
"""
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

Image.MAX_IMAGE_PIXELS = None
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT = Path(__file__).resolve().parent.parent / "assets" / "ui"


def grade(img, desat=0.25, tint=(0.94, 1.0, 0.98), gamma=1.0, gain=1.0):
    """The Half-Life 2 grade: a little less color, a cold green-teal cast."""
    a = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    lum = (a * [0.299, 0.587, 0.114]).sum(axis=2, keepdims=True)
    a = a * (1 - desat) + lum * desat
    a = a * np.array(tint, np.float32) * gain
    a = np.clip(a, 0, 1) ** gamma
    return Image.fromarray((a * 255 + 0.5).astype(np.uint8))


def save_jpg(img, name, quality=86):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, quality=quality, optimize=True, progressive=False)
    print(f"{name}: {path.stat().st_size // 1024} KiB {img.size}")


def save_png(img, name):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, optimize=True)
    print(f"{name}: {path.stat().st_size // 1024} KiB {img.size}")


# ---------------------------------------------------------------- skies


def sky(src_id, name, **kw):
    """The band from just below the horizon to 48 degrees up of an
    equirectangular sky, 4096 wide so it tiles around 360 degrees."""
    im = Image.open(SRC / "sky" / f"{src_id}.jpg")
    w, h = im.size
    top = int(h * (0.5 - 48 / 180))
    bottom = int(h * (0.5 + 1.5 / 180))
    band = im.crop((0, top, w, bottom))
    width = 4096
    band = band.resize((width, int(width * (bottom - top) / w)), Image.LANCZOS)
    save_jpg(grade(band, **kw), f"sky/{name}.jpg", quality=84)


# ------------------------------------------------------------- materials


def material(acg_id, name, size=512, **kw):
    im = Image.open(SRC / "acg" / acg_id / f"{acg_id}_1K-JPG_Color.jpg")
    im = im.resize((size, size), Image.LANCZOS)
    save_jpg(grade(im, **kw), f"textures/{name}.jpg", quality=85)


# --------------------------------------------------------------- sprites


def sprites():
    n = 128
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    cx = cy = (n - 1) / 2
    r = n / 2 - 2
    dx, dy = (x - cx) / r, (y - cy) / r
    d2 = dx * dx + dy * dy
    inside = np.clip((1 - np.sqrt(d2)) * r, 0, 1)
    z = np.sqrt(np.clip(1 - d2, 0, 1))
    # Light from the upper left, a cool sky fill from above, a dark rim.
    light = np.array([-0.55, -0.6, 0.58])
    light /= np.linalg.norm(light)
    lambert = np.clip(dx * light[0] + dy * light[1] + z * light[2], 0, 1)
    fill = np.clip(-dy * 0.5 + 0.5, 0, 1) * 0.25
    spec = np.clip(dx * light[0] + dy * light[1] + z * light[2], 0, 1) ** 28
    shade = 0.18 + 0.72 * lambert + fill
    rim = np.clip(1 - z, 0, 1) ** 3 * 0.35
    v = np.clip(shade - rim, 0, 1)
    rgb = np.stack([v, v, v * 1.02], axis=2) + spec[..., None] * 0.9
    rgba = np.concatenate([np.clip(rgb, 0, 1), inside[..., None]], axis=2)
    save_png(Image.fromarray((rgba * 255).astype(np.uint8), "RGBA"), "sprites/sphere.png")
    # A soft glow: a gaussian falloff in white.
    g = np.exp(-d2 * 3.2) * np.clip(1 - np.sqrt(d2), 0, 1)
    glow = np.stack([np.ones_like(g)] * 3 + [g], axis=2)
    save_png(Image.fromarray((glow * 255).astype(np.uint8), "RGBA"), "sprites/glow.png")


# --------------------------------------------------------------- skyline


class Canvas:
    """A layer drawn at twice its size and shrunk, so edges are smooth.
    Every shape is drawn at x and x +/- width, so the layer tiles."""

    def __init__(self, w, h, ss=2):
        self.w, self.h, self.ss = w, h, ss
        self.img = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)

    def _xy(self, pts, shift):
        s = self.ss
        return [((x + shift) * s, y * s) for x, y in pts]

    def poly(self, pts, fill):
        for shift in (-self.w, 0, self.w):
            self.d.polygon(self._xy(pts, shift), fill=fill)

    def rect(self, x0, y0, x1, y1, fill):
        self.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], fill)

    def line(self, pts, fill, width):
        for shift in (-self.w, 0, self.w):
            self.d.line(self._xy(pts, shift), fill=fill, width=max(1, int(width * self.ss)))

    def ellipse(self, x0, y0, x1, y1, fill):
        s = self.ss
        for shift in (-self.w, 0, self.w):
            self.d.ellipse(((x0 + shift) * s, y0 * s, (x1 + shift) * s, y1 * s), fill=fill)

    def paste_texture(self, tex, box, tint, dark):
        """Fills a box with a material texture, tinted and darkened."""
        x0, y0, x1, y1 = [int(v * self.ss) for v in box]
        if x1 <= x0 or y1 <= y0:
            return
        t = np.asarray(tex.convert("RGB")).astype(np.float32) / 255
        th, tw = t.shape[:2]
        ys = (np.arange(y1 - y0) % th)
        xs = (np.arange(x1 - x0) % tw)
        patch = t[ys][:, xs] * np.array(tint, np.float32) * dark
        patch = np.clip(patch * 255, 0, 255).astype(np.uint8)
        piece = Image.fromarray(patch, "RGB").convert("RGBA")
        for shift in (-self.w, 0, self.w):
            self.img.paste(piece, (x0 + shift * self.ss, y0))

    def done(self):
        return self.img.resize((self.w, self.h), Image.LANCZOS)


def haze(img, color, amount):
    """Mixes the opaque parts toward the sky color, for distance."""
    a = np.asarray(img).astype(np.float32)
    rgb, alpha = a[..., :3], a[..., 3:]
    rgb = rgb * (1 - amount) + np.array(color, np.float32) * amount
    return Image.fromarray(np.concatenate([rgb, alpha], axis=2).clip(0, 255).astype(np.uint8), "RGBA")


def grime(img, strength=0.35, tile=384):
    """Breaks up flat paint with a photographed concrete texture, so every
    surface carries the dirt and wear of a real one."""
    tex = Image.open(SRC / "acg" / "Concrete031" / "Concrete031_1K-JPG_Color.jpg").convert("L").resize((tile, tile))
    t = np.asarray(tex).astype(np.float32) / 255.0
    t = (t - t.mean()) / (t.std() + 1e-6)
    a = np.asarray(img).astype(np.float32)
    h, w = a.shape[:2]
    reps = (h // tile + 1, w // tile + 1)
    field = np.tile(t, reps)[:h, :w]
    a[..., :3] *= (1.0 + strength * 0.25 * field)[..., None]
    return Image.fromarray(a.clip(0, 255).astype(np.uint8), "RGBA")


def vertical_fade(img, top_alpha, until):
    """Fades a layer's top into the clouds: alpha scales from top_alpha at
    the top edge to 1 at row `until`."""
    a = np.asarray(img).astype(np.float32)
    h = a.shape[0]
    ramp = np.clip(np.arange(h) / max(until, 1), 0, 1)
    scale = top_alpha + (1 - top_alpha) * ramp
    a[..., 3] *= scale[:, None]
    return Image.fromarray(a.clip(0, 255).astype(np.uint8), "RGBA")


def citadel(c, x, base_y, height, rng):
    """The great tower: a dark metal shaft that narrows as it climbs into
    the clouds, cut into slabs, with a lit edge and a few cold lights."""
    body = (58, 68, 80, 255)
    edge = (104, 118, 132, 255)
    shade = (40, 47, 56, 255)
    w0, w1 = 190, 64
    top = base_y - height

    def half(y):
        t = (base_y - y) / height
        return w0 / 2 + (w1 / 2 - w0 / 2) * t ** 0.8

    pts = [(x - half(base_y), base_y), (x - half(top), top), (x + half(top) * 0.6, top - 18),
           (x + half(top), top + 10), (x + half(base_y), base_y)]
    c.poly(pts, body)
    # The shaded right face.
    c.poly([(x + half(base_y) * 0.25, base_y), (x + half(top) * 0.25, top + 4),
            (x + half(top), top + 10), (x + half(base_y), base_y)], shade)
    # A thin lit left edge.
    c.poly([(x - half(base_y), base_y), (x - half(top), top), (x - half(top) + 4, top),
            (x - half(base_y) + 6, base_y)], edge)
    # Slabs and seams.
    y = base_y - 20
    while y > top + 20:
        h = half(y)
        c.line([(x - h + 3, y), (x + h - 3, y)], (32, 38, 46, 255), 1.4)
        y -= rng.uniform(18, 46)
    for k in range(5):
        t = (k + 1) / 6
        c.line([(x - half(base_y) + w0 * t * 0.9, base_y), (x - half(top) + w1 * t * 0.9, top + 8)],
               (46, 54, 64, 255), 1.0)
    # Spurs and struts around the base.
    for k in range(7):
        side = -1 if k % 2 else 1
        y0 = base_y - rng.uniform(40, 260)
        h = half(y0)
        length = rng.uniform(30, 90)
        c.poly([(x + side * h, y0), (x + side * (h + length), y0 + length * 0.9),
                (x + side * (h + length - 6), y0 + length * 0.9 + 4), (x + side * h, y0 + 12)], body)
    # Cold lights along the shaft.
    for k in range(9):
        y0 = base_y - rng.uniform(30, height * 0.8)
        xx = x + rng.uniform(-half(y0) * 0.7, half(y0) * 0.7)
        c.ellipse(xx - 1.6, y0 - 1.6, xx + 1.6, y0 + 1.6, (170, 214, 255, 255))


def block(c, x, w, h, base_y, color, rng, windows=True, lit=0.03, window_color=(20, 22, 24, 255), tex=None):
    """An apartment block with a grid of windows and a few lit ones."""
    if tex is not None:
        c.paste_texture(tex, (x, base_y - h, x + w, base_y), [v / 255 for v in color[:3]], 1.0)
    else:
        c.rect(x, base_y - h, x + w, base_y, color)
    if not windows:
        return
    cols = max(2, int(w / rng.uniform(9, 14)))
    rows = max(2, int(h / rng.uniform(10, 15)))
    cw, rh = w / cols, h / rows
    for i in range(cols):
        for j in range(1, rows):
            if rng.random() < 0.1:
                continue
            wx, wy = x + i * cw + cw * 0.28, base_y - h + j * rh + rh * 0.2
            col = (255, 196, 120, 230) if rng.random() < lit else window_color
            c.rect(wx, wy, wx + cw * 0.44, wy + rh * 0.5, col)


def crane(c, x, base_y, h, color, rng):
    top = base_y - h
    c.line([(x, base_y), (x, top)], color, 3)
    c.line([(x - 3, base_y), (x - 3, top)], color, 1.2)
    for y in np.arange(top, base_y, 8):
        c.line([(x - 3, y), (x, y + 8)], color, 0.8)
    jib = rng.uniform(90, 160)
    c.line([(x - 30, top + 4), (x + jib, top + 4)], color, 2.4)
    c.line([(x, top - 16), (x + jib * 0.8, top + 4)], color, 1.0)
    c.line([(x, top - 16), (x - 30, top + 4)], color, 1.0)
    c.line([(x + jib * 0.6, top + 4), (x + jib * 0.6, top + rng.uniform(30, 80))], color, 0.8)


def far_layer(rng):
    w, h = 4096, 720
    base = h - 4
    c = Canvas(w, h)
    blocks = (128, 136, 140, 255)
    far_tex = Image.open(SRC / "acg" / "Concrete031" / "Concrete031_1K-JPG_Color.jpg").resize((200, 200))
    far_tex = Image.fromarray(np.clip(np.asarray(far_tex).astype(np.float32) * 1.25, 0, 255).astype(np.uint8))
    for k in range(90):
        x = rng.uniform(0, w)
        bw = rng.uniform(50, 140)
        bh = rng.uniform(60, 190) * (1.5 if rng.random() < 0.2 else 1)
        shade = rng.uniform(0.9, 1.08)
        col = tuple(int(v * shade) for v in blocks[:3]) + (255,)
        block(c, x, bw, bh, base, col, rng, lit=0.0, window_color=(96, 106, 112, 255), tex=far_tex)
    for k in range(4):
        crane(c, rng.uniform(0, w), base - rng.uniform(30, 90), rng.uniform(120, 190), (108, 118, 124, 255), rng)
    for k in range(6):
        # Chimney stacks.
        x = rng.uniform(0, w)
        sh = rng.uniform(120, 220)
        c.poly([(x - 7, base), (x - 4, base - sh), (x + 4, base - sh), (x + 7, base)], (112, 122, 128, 255))
    far = shade_layer(c.done(), top_light=1.0, base_dark=0.8, fog=(150, 156, 156), fog_amount=0.6)
    far = haze(far, (148, 156, 158), 0.3)
    # The spire stands alone in front of the far blocks.
    c2 = Canvas(w, h)
    citadel(c2, 2900, base, 720, rng)
    spire = haze(c2.done(), (140, 150, 156), 0.28)
    spire = vertical_fade(spire, 0.25, 420)
    out = Image.alpha_composite(far, spire)
    save_png(grime(out, 0.22), "skyline/far.png")


def dome(c, x, y, r, color):
    """A drum with a bell-shaped copper roof and a spike, like the old
    station towers of City 17. `y` is the roof line it stands on."""
    drum = r * 0.55
    c.rect(x - r * 0.8, y - drum, x + r * 0.8, y, (58, 56, 52, 255))
    for k in range(4):
        wx = x - r * 0.6 + k * r * 0.36
        c.rect(wx, y - drum * 0.8, wx + r * 0.14, y - drum * 0.25, (14, 16, 18, 255))
    base = y - drum
    pts = []
    for i in range(25):
        a = math.pi * i / 24
        s = math.sin(a)
        pts.append((x - r * math.cos(a), base - r * 1.1 * s ** 0.8))
    c.poly(pts, color)
    c.line([(x, base - r * 1.1), (x, base - r * 1.9)], color, 2)
    c.ellipse(x - 2.5, base - r * 1.5 - 2.5, x + 2.5, base - r * 1.5 + 2.5, color)


def old_building(c, x, w, h, base_y, textures, rng):
    """A City 17 tenement: a textured facade, window rows with frames and
    ledges, a cornice, a slate mansard roof, chimneys and antennas."""
    tex = textures[rng.randrange(len(textures))]
    tint = rng.choice([(1.0, 0.84, 0.74), (0.95, 0.92, 0.86), (0.84, 0.87, 0.9), (0.92, 0.8, 0.66),
                       (0.78, 0.8, 0.74)])
    dark = rng.uniform(0.4, 0.58)
    top = base_y - h
    c.paste_texture(tex, (x, top, x + w, base_y), tint, dark)
    floors = max(3, int(h / rng.uniform(30, 40)))
    fh = h / floors
    cols = max(2, int(w / rng.uniform(34, 46)))
    cw = w / cols
    # A plinth of heavier stone at street level.
    c.rect(x, base_y - fh * 0.9, x + w, base_y, (40, 38, 36, 200))
    for f in range(1, floors):
        y = top + f * fh
        c.rect(x, y - 1.5, x + w, y + 1.0, (30, 29, 27, 160))
        for i in range(cols):
            wx = x + i * cw + cw * 0.3
            wy = y + fh * 0.2
            ww, wh = cw * 0.4, fh * 0.56
            lit = rng.random() < 0.045
            boarded = rng.random() < 0.06
            c.rect(wx - 2, wy - 2, wx + ww + 2, wy + wh + 2, (62, 58, 52, 255))
            glass = (226, 162, 88, 255) if lit else ((74, 60, 44, 255) if boarded else (14, 17, 19, 255))
            c.rect(wx, wy, wx + ww, wy + wh, glass)
            # The recess shadow along the top and left of the pane.
            c.rect(wx, wy, wx + ww, wy + 2.2, (6, 7, 8, 200))
            c.rect(wx, wy, wx + 1.8, wy + wh, (6, 7, 8, 160))
            c.line([(wx + ww / 2, wy), (wx + ww / 2, wy + wh)], (52, 50, 46, 255), 0.9)
            c.line([(wx, wy + wh * 0.4), (wx + ww, wy + wh * 0.4)], (52, 50, 46, 255), 0.9)
            c.rect(wx - 3, wy + wh + 1, wx + ww + 3, wy + wh + 3.5, (90, 86, 78, 255))
            if rng.random() < 0.5:
                # A grime streak down from the sill.
                sl = rng.uniform(6, fh * 0.8)
                c.rect(wx + rng.uniform(0, ww), wy + wh + 3.5, wx + rng.uniform(0, ww) + 1.4,
                       wy + wh + 3.5 + sl, (20, 18, 16, 70))
    # The cornice and the roof.
    c.rect(x - 4, top - 5, x + w + 4, top + 2, (76, 72, 66, 255))
    kind = rng.random()
    roof = (44, 48, 52, 255)
    if kind < 0.55:
        rh = rng.uniform(14, 30)
        c.poly([(x - 2, top - 5), (x + 10, top - 5 - rh), (x + w - 10, top - 5 - rh), (x + w + 2, top - 5)], roof)
        for i in range(max(1, cols // 2)):
            dx = x + 16 + i * (w - 32) / max(1, cols // 2 - 1 if cols > 3 else 1)
            c.rect(dx, top - 5 - rh * 0.7, dx + 8, top - 5, (60, 62, 64, 255))
            c.poly([(dx - 1, top - 5 - rh * 0.7), (dx + 4, top - 10 - rh * 0.7), (dx + 9, top - 5 - rh * 0.7)], roof)
        top_line = top - 5 - rh
    elif kind < 0.7:
        dome(c, x + w / 2, top - 4, w * 0.18, roof)
        top_line = top - 5
    else:
        top_line = top - 5
    for k in range(rng.randrange(1, 4)):
        cx = x + rng.uniform(8, w - 14)
        c.rect(cx, top_line - rng.uniform(10, 22), cx + 7, top_line + 2, (58, 52, 48, 255))
    if rng.random() < 0.6:
        ax = x + rng.uniform(10, w - 10)
        ah = rng.uniform(20, 38)
        c.line([(ax, top_line), (ax, top_line - ah)], (30, 30, 30, 255), 1.2)
        for k in range(3):
            yy = top_line - ah + k * 6
            c.line([(ax - 8 + k, yy), (ax + 8 - k, yy)], (30, 30, 30, 255), 1.0)


def combine_wall(c, x, w, h, base_y, plate):
    """A Combine barrier: dark ribbed metal, a slanted top, cold slits."""
    c.paste_texture(plate, (x, base_y - h, x + w, base_y), (0.7, 0.8, 0.95), 0.32)
    c.poly([(x, base_y - h), (x + w * 0.5, base_y - h - 26), (x + w, base_y - h), ], (22, 26, 32, 255))
    for i in range(int(w / 14)):
        xx = x + 7 + i * 14
        c.line([(xx, base_y - h + 4), (xx, base_y - 4)], (14, 16, 20, 255), 2)
    c.rect(x + 6, base_y - h * 0.62, x + w - 6, base_y - h * 0.62 + 2.5, (120, 200, 255, 255))


def shade_layer(img, top_light=1.05, base_dark=0.55, fog=(128, 136, 138), fog_amount=0.0):
    """Light from the overcast sky: tops a little brighter, the foot of
    everything darker, then a band of ground fog at the base."""
    a = np.asarray(img).astype(np.float32)
    h = a.shape[0]
    t = np.linspace(0, 1, h)[:, None, None]
    light = top_light + (base_dark - top_light) * t ** 1.6
    rgb = a[..., :3] * light
    fog_t = np.clip((t - 0.55) / 0.45, 0, 1) ** 1.5 * fog_amount
    rgb = rgb * (1 - fog_t) + np.array(fog, np.float32) * fog_t
    return Image.fromarray(np.concatenate([rgb, a[..., 3:]], axis=2).clip(0, 255).astype(np.uint8), "RGBA")


def mid_layer(rng):
    w, h = 4096, 460
    base = h - 2
    c = Canvas(w, h)
    textures = [Image.open(SRC / "acg" / i / f"{i}_1K-JPG_Color.jpg").resize((300, 300))
                for i in ("Bricks075A", "Concrete034", "Concrete047A", "Concrete031", "Concrete042A")]
    plate = Image.open(SRC / "acg" / "MetalPlates006" / "MetalPlates006_1K-JPG_Color.jpg").resize((192, 192))
    x = 0
    while x < w - 160:
        if 2300 < x < 2700:
            combine_wall(c, x, 300, 330, base, plate)
            x += 330 + rng.uniform(60, 160)
            continue
        # Clusters of two to four tenements, then a gap to the far city.
        for k in range(rng.randrange(2, 5)):
            bw = rng.uniform(150, 290)
            bh = rng.uniform(130, 330)
            if x + bw > w - 20:
                break
            old_building(c, x, bw, bh, base, textures, rng)
            x += bw + rng.uniform(-6, 6)
        x += rng.uniform(90, 320)
    img = shade_layer(c.done(), fog=(120, 128, 130), fog_amount=0.35)
    img = haze(img, (118, 128, 132), 0.12)
    save_png(grime(img, 0.8), "skyline/mid.png")


def catenary(p0, p1, sag, n=24):
    pts = []
    for i in range(n + 1):
        t = i / n
        x = p0[0] + (p1[0] - p0[0]) * t
        y = p0[1] + (p1[1] - p0[1]) * t + sag * 4 * t * (1 - t)
        pts.append((x, y))
    return pts


def tree(c, x, y, length, angle, width, depth, rng, color, leaves):
    if depth == 0 or length < 3:
        if rng.random() < 0.35:
            r = rng.uniform(1.5, 3.2)
            c.ellipse(x - r, y - r, x + r, y + r, leaves[rng.randrange(len(leaves))])
        return
    x2 = x + math.cos(angle) * length
    y2 = y - math.sin(angle) * length
    c.line([(x, y), (x2, y2)], color, width)
    for k in range(rng.choice([2, 2, 3])):
        tree(c, x2, y2, length * rng.uniform(0.62, 0.78), angle + rng.uniform(-0.65, 0.65),
             width * 0.66, depth - 1, rng, color, leaves)


def near_layer(rng):
    w, h = 4096, 560
    base = h - 2
    c = Canvas(w, h)
    dark = (26, 28, 28, 255)
    poles = []
    x = 80
    while x < w:
        poles.append(x)
        x += rng.uniform(520, 760)
    # Wires sag between poles, and across the seam to the first pole.
    tops = []
    for px in poles:
        ph = rng.uniform(330, 400)
        top = base - ph
        tops.append((px, top))
        c.rect(px - 4, top, px + 4, base, dark)
        c.rect(px - 38, top + 16, px + 38, top + 22, dark)
        c.rect(px - 26, top + 44, px + 26, top + 49, dark)
        for k in (-34, -14, 14, 34):
            c.rect(px + k - 2, top + 8, px + k + 2, top + 17, (40, 42, 42, 255))
    for i, (px, top) in enumerate(tops):
        nx, ntop = tops[(i + 1) % len(tops)]
        if i + 1 == len(tops):
            nx += w
        for k, dy in ((-34, 10), (-14, 10), (14, 10), (34, 10), (-22, 40), (22, 40)):
            c.line(catenary((px + k, top + dy), (nx + k, ntop + dy), rng.uniform(40, 70)), (22, 24, 24, 255), 1.3)
    leaves = [(150, 80, 30, 220), (120, 62, 26, 220), (92, 70, 40, 220)]
    for k in range(9):
        tx = rng.uniform(0, w)
        tree(c, tx, base, rng.uniform(60, 90), math.pi / 2 + rng.uniform(-0.1, 0.1), 7, 7, rng, (30, 28, 26, 255), leaves)
    for k in range(7):
        # Street lamps.
        lx = rng.uniform(0, w)
        lh = rng.uniform(170, 210)
        c.rect(lx - 2.5, base - lh, lx + 2.5, base, dark)
        c.line([(lx, base - lh), (lx + 10, base - lh - 10), (lx + 34, base - lh - 8)], dark, 3)
        c.rect(lx + 26, base - lh - 10, lx + 42, base - lh - 4, (40, 40, 38, 255))
    for k in range(18):
        # Striped bollards.
        bx = rng.uniform(0, w)
        for s in range(4):
            col = (214, 214, 206, 255) if s % 2 == 0 else (20, 20, 20, 255)
            c.rect(bx - 4, base - 30 + s * 7.5, bx + 4, base - 30 + (s + 1) * 7.5, col)
        c.ellipse(bx - 5, base - 35, bx + 5, base - 26, (20, 20, 20, 255))
    for k in range(6):
        # Railings.
        fx = rng.uniform(0, w)
        fw = rng.uniform(120, 260)
        c.rect(fx, base - 40, fx + fw, base - 37, dark)
        c.rect(fx, base - 12, fx + fw, base - 10, dark)
        for i in range(int(fw / 9)):
            c.rect(fx + i * 9, base - 40, fx + i * 9 + 1.6, base, dark)
    for k in range(3):
        # A Breen screen on a pole, glowing.
        sx = rng.uniform(0, w)
        c.rect(sx - 4, base - 280, sx + 4, base, dark)
        c.rect(sx + 4, base - 272, sx + 38, base - 230, (18, 22, 26, 255))
        c.rect(sx + 7, base - 269, sx + 35, base - 233, (96, 150, 160, 255))
        c.rect(sx + 7, base - 269, sx + 35, base - 262, (150, 200, 206, 255))
    save_png(grime(shade_layer(c.done(), top_light=1.0, base_dark=0.85), 1.0), "skyline/near.png")


# ---------------------------------------------------------- menu backdrop


def backdrop():
    """The picture behind the whole window, like the live scene behind a
    Half-Life 2 menu: the city under the overcast sky, blurred and dark."""
    W, H = 1920, 1080
    sky_img = Image.open(OUT / "sky" / "city.jpg")
    sw, sh = sky_img.size
    crop = sky_img.crop((1400, 0, 1400 + int(sh * W / H * 1.0), sh)).resize((W, int(H * 0.78)), Image.LANCZOS)
    canvas = Image.new("RGBA", (W, H), (40, 44, 44, 255))
    canvas.paste(crop, (0, 0))
    horizon = int(H * 0.78)
    for name, scale, shift in (("far", 0.95, 900), ("mid", 0.55, 300), ("near", 0.75, 1200)):
        layer = Image.open(OUT / "skyline" / f"{name}.png")
        lw, lh = layer.size
        th = int(H * scale)
        tw = int(lw * th / lh)
        layer = layer.resize((tw, th), Image.LANCZOS)
        x = -shift
        while x < W:
            canvas.alpha_composite(layer, (x, horizon - th))
            x += tw
    ground = Image.open(OUT / "textures" / "cobble.jpg").resize((256, 256))
    for gx in range(0, W, 256):
        for gy in range(horizon, H, 256):
            canvas.paste(ground, (gx, gy))
    img = canvas.convert("RGB").filter(ImageFilter.GaussianBlur(7))
    img = grade(img, desat=0.2, gain=0.42)
    save_jpg(img, "backdrop.jpg", quality=82)


def main():
    rng = random.Random(17)
    sky("kloofendal_overcast_puresky", "city", desat=0.28, tint=(0.93, 1.0, 0.98))
    sky("overcast_soil_puresky", "storm", desat=0.3, tint=(0.92, 0.99, 1.0), gain=0.92)
    sky("kloppenheim_01_puresky", "dusk", desat=0.12, tint=(1.0, 0.95, 0.88))
    material("Concrete042A", "concrete_dark", desat=0.2)
    material("Concrete034", "concrete_light", desat=0.2, gain=0.85)
    material("PavingStones070", "cobble", desat=0.3, gain=0.8)
    material("MetalPlates013", "rust_steel", desat=0.15)
    material("MetalPlates006", "combine_plate", desat=0.1, tint=(0.8, 0.9, 1.0))
    material("Ground036", "mud", desat=0.2, gain=0.8)
    material("Ground054", "sand", desat=0.25, gain=0.9)
    material("Ground023", "dirt", desat=0.2, gain=0.85)
    material("Rust009", "rust", desat=0.2)
    sprites()
    far_layer(rng)
    mid_layer(rng)
    near_layer(rng)
    backdrop()


if __name__ == "__main__":
    main()
