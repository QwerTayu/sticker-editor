"""editor.html で作った配置から、入稿用のステッカー PNG を書き出す。

    python render_layout.py sticker-project.json [出力.png]
    python render_layout.py layout.json [出力.png] --images 画像フォルダ

「プロジェクトを保存」で出るファイルは画像を埋め込んでいるので、それだけで書き出せる。
「配置をコピー」で出る JSON は画像を含まないので、--images のフォルダ（省略時は JSON と同じフォルダ）から
<name>.png（.webp / .jpg / .jpeg / .gif も可）を探す。出力を省略すると、JSON と同じ名前の .png に書き出す。

出力の横に、黒いカット線入りの確認用画像（*-preview.png、背景は透明）も作る。Pillow, numpy, scipy が必要。
"""
import argparse
import base64
import io
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage as ndi

EXTS = ['.png', '.webp', '.jpg', '.jpeg', '.gif']
DPI, PAD, GAP = 300, 30, 14
MARGIN = 500  # 台紙（とくに円）がキャンバスからはみ出しても切れないよう、周りに足す余白


def dilate(m, r):
    """半径 r の円で膨張（距離変換を使うので、大きな r でも速い）"""
    return ndi.distance_transform_edt(~m) <= r


def erode(m, r):
    return ndi.distance_transform_edt(m) > r


def open_image(name, cfg, images_dir):
    """プロジェクトファイルに埋め込まれていればそれを、なければ images_dir の <name>.png などを開く"""
    src = cfg.get('images', {}).get(name)
    if src:
        return Image.open(io.BytesIO(base64.b64decode(src.split(',', 1)[1])))
    for ext in EXTS:
        if (images_dir / f'{name}{ext}').exists():
            return Image.open(images_dir / f'{name}{ext}')
    raise SystemExit(f'{name} の画像が見つかりません（{images_dir} に {name}.png などを置くか、プロジェクトファイルを使ってください）')


def load(d, cfg, images_dir):
    """editor.html の prepare と同じ手順。denoise が真なら透過部分のノイズを除去する（細い線は消えることがある）"""
    im = open_image(d['name'], cfg, images_dir).convert('RGBA')
    m = np.array(im)[..., 3] > 40
    if d.get('denoise', True):
        m = ndi.binary_opening(m, iterations=2)
    arr = np.array(im)
    arr[..., 3] = np.where(m, arr[..., 3], 0)
    im = Image.fromarray(arr)
    return im.crop(im.getbbox())


def body_mask(d, m):
    """重なりチェックに使う本体。body が main なら一番大きい塊＋本体より上の飾り（下の風の線などは除く）、それ以外は全体"""
    if d.get('body', 'whole') != 'main':
        return m
    filled = ndi.binary_fill_holes(m)
    lab, n = ndi.label(filled)
    sizes = ndi.sum(filled, lab, range(1, n + 1))
    main = 1 + int(np.argmax(sizes))
    body = lab == main
    bottom = np.where(body)[0].max()
    for k in range(1, n + 1):
        if k != main and np.where(lab == k)[0].mean() < bottom - 10:
            body |= lab == k
    return body


def sheet_spec(cfg):
    """新しい形式（sheet）と以前の形式（border / shape / corner が直下）の両方を読む"""
    if 'sheet' in cfg:
        return cfg['sheet']
    return {'shape': cfg.get('shape', 'contour'), 'margin': cfg.get('border', 40), 'corner': cfg.get('corner', 0)}


def contour_sheet(content, border):
    big = dilate(ndi.binary_fill_holes(content), border)
    big = erode(dilate(big, border * 2), border * 2)  # クロージング
    big = ndi.binary_fill_holes(big)
    if ndi.label(big)[1] > 1:
        print('warning: 台紙が複数に分かれています')
    return big


def shape_sheet(spec, size):
    """円・長方形・正方形の台紙。x, y（中心）と大きさは JSON の値を使う（auto でも editor が計算した値が入っている）"""
    if 'x' not in spec:
        raise SystemExit('円・長方形・正方形の台紙には x, y と大きさが必要です（editor.html で書き出した JSON を使ってください）')
    shape = spec['shape']
    w = spec.get('diameter') or spec.get('size') or spec['width']
    h = spec.get('diameter') or spec.get('size') or spec['height']
    cx, cy = spec['x'] + MARGIN, spec['y'] + MARGIN
    box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    img = Image.new('L', size, 0)
    draw = ImageDraw.Draw(img)
    if shape == 'circle':
        draw.ellipse(box, fill=255)
    else:
        draw.rounded_rectangle(box, radius=min(spec.get('corner', 0), w / 2, h / 2), fill=255)
    return np.array(img) > 127


def main():
    ap = argparse.ArgumentParser(description='editor.html の配置からステッカー PNG を書き出す')
    ap.add_argument('layout', type=Path, help='プロジェクトファイル、または配置の JSON')
    ap.add_argument('out', type=Path, nargs='?', help='出力 PNG（省略時は JSON と同じ名前の .png）')
    ap.add_argument('--images', type=Path, help='配置の JSON のときに画像を探すフォルダ（省略時は JSON と同じフォルダ）')
    args = ap.parse_args()
    cfg = json.loads(args.layout.read_text().replace('```json', '').replace('```', ''))
    out_path = args.out or args.layout.with_suffix('.png')
    images_dir = args.images or args.layout.parent
    cw, ch = cfg['canvas']
    width_cm = cfg.get('width_cm', 10)
    W, H = cw + 2 * MARGIN, ch + 2 * MARGIN

    canvas = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    bodies = {}
    for d in cfg['layout']:
        im = load(d, cfg, images_dir)
        s = d['scale']
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS).rotate(d['rot'], Image.BICUBIC, expand=True)
        x, y = round(d['x']) + MARGIN - im.width // 2, round(d['y']) + MARGIN - im.height // 2
        canvas.alpha_composite(im, (x, y))
        m = np.zeros((H, W), bool)
        m[y:y + im.height, x:x + im.width] = np.array(im)[..., 3] > 40
        bodies[d['name']] = body_mask(d, m)

    grown = {k: dilate(v, GAP / 2) for k, v in bodies.items()}
    pairs = [(a, b) for a, b in itertools.combinations(grown, 2) if (grown[a] & grown[b]).any()]
    print('本体の重なり：' + ('、'.join(f'{a} × {b}' for a, b in pairs) if pairs else 'なし'))

    content = np.array(canvas)[..., 3] > 40
    spec = sheet_spec(cfg)
    if spec['shape'] == 'contour':
        sheet = contour_sheet(content, spec.get('margin', 40))
    else:
        sheet = shape_sheet(spec, (W, H))
        if (content & ~sheet).any():
            print('warning: 中身が台紙からはみ出しています')
    sheet = np.array(Image.fromarray((sheet * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(4))) > 127

    ys, xs = np.where(sheet)
    box = (xs.min() - PAD, ys.min() - PAD, xs.max() + PAD + 1, ys.max() + PAD + 1)
    mask = Image.fromarray((sheet * 255).astype(np.uint8))
    out = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    out.paste(Image.new('RGBA', (W, H), (255, 255, 255, 255)), (0, 0), mask)
    out.alpha_composite(canvas)
    # 台紙の外にはみ出した部分は切り落とす（実物もカット線で切られる）
    arr = np.array(out)
    arr[..., 3] = np.minimum(arr[..., 3], np.array(mask))
    out = Image.fromarray(arr)
    out, mask = out.crop(box), mask.crop(box)
    tw = round(width_cm / 2.54 * DPI)
    th = round(out.height * tw / out.width)
    out = out.resize((tw, th), Image.LANCZOS)
    out.save(out_path, dpi=(DPI, DPI))

    edge = np.array(mask.resize((tw, th), Image.LANCZOS).filter(ImageFilter.FIND_EDGES)) > 60
    p = np.array(out)  # 背景は透明のまま、カット線を黒で描く
    p[edge] = (0, 0, 0, 255)
    Image.fromarray(p).save(out_path.with_name(out_path.stem + '-preview.png'))
    print(f'{out_path.name}: {tw}x{th}px（{tw / DPI * 2.54:.1f} x {th / DPI * 2.54:.1f} cm）、台紙 {spec["shape"]}')


if __name__ == '__main__':
    main()
