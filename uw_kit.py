from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
REAL = ['85004', '85281', '85016']
NBHD = {'85004': 'Downtown (high-rise core)', '85281': 'Tempe / ASU (mid-rise)', '85016': 'Arcadia (suburban)'}
COLORS = {'85004': '#d95f0e', '85281': '#31a354', '85016': '#2c7fb8'}


def ensure_dataset(root, url):
    import urllib.request, tarfile
    root = Path(root)
    if (root / 'data' / 'per_viewpoint.parquet').exists():
        return root
    root.mkdir(parents=True, exist_ok=True)
    arc = root / '_uw_data.tar.gz'
    print('downloading dataset (~185 MB), one time...', flush=True)
    urllib.request.urlretrieve(url, arc)
    with tarfile.open(arc) as t:
        t.extractall(root)
    arc.unlink()
    print('dataset ready under', root)
    return root

def load_fullcity(root, url):
    import urllib.request, zipfile, tarfile, shutil
    root = Path(root)
    data = root / 'data_citywide'
    data.mkdir(parents=True, exist_ok=True)
    arc = root / '_fullcity_arc'
    urllib.request.urlretrieve(url, arc)
    tmp = root / '_fullcity'
    if zipfile.is_zipfile(arc):
        with zipfile.ZipFile(arc) as z:
            z.extractall(tmp)
    else:
        with tarfile.open(arc) as t:
            t.extractall(tmp)
    for name in ['per_viewpoint.parquet', 'phoenix_building_green_heat.gpkg']:
        shutil.copy(next(tmp.rglob(name)), data / name)
    arc.unlink()
    shutil.rmtree(tmp)
    print('full-city tables loaded into', data)
    return data

def viewpoints_for_box(width=40, depth=20, height=30, step=10.0, floor_h=3.0):
    levels = max(1, int(height // floor_h))
    corners = [(0, 0), (width, 0), (width, depth), (0, depth)]
    pts = []
    for i in range(4):
        (x0, y0), (x1, y1) = (corners[i], corners[(i + 1) % 4])
        seg = np.hypot(x1 - x0, y1 - y0)
        n = max(1, int(seg // step))
        nx, ny = ((y1 - y0) / seg, -(x1 - x0) / seg)
        for k in range(n):
            t = (k + 0.5) / n
            px, py = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
            for lv in range(levels):
                pts.append((px, py, lv * floor_h + 1.5, nx, ny))
    return (levels, np.array(pts))

def plot_viewpoints(vp, width=40, depth=20, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.add_patch(plt.Rectangle((0, 0), width, depth, fill=False, lw=2))
    ground = vp[np.isclose(vp[:, 2], 1.5)]
    ax.quiver(ground[:, 0], ground[:, 1], ground[:, 3], ground[:, 4], color='green', scale=20, width=0.005)
    ax.scatter(ground[:, 0], ground[:, 1], c='green', s=20)
    ax.set_aspect('equal')
    pad = 0.25 * max(width, depth)
    ax.set_xlim(-pad, width + pad)
    ax.set_ylim(-pad, depth + pad)
    ax.set_xlabel('metres')
    ax.set_title('Viewpoints around one building (ground floor; arrows = looking direction)')
    return ax

def load_segmenter(weights):
    import torch, torch.nn as nn
    from torchvision import models
    m = models.segmentation.deeplabv3_resnet50(weights=None, weights_backbone=None)
    m.classifier[4] = nn.Conv2d(256, 8, kernel_size=1)
    ck = torch.load(weights, map_location='cpu')
    sd = ck.get('state_dict', ck.get('model', ck)) if isinstance(ck, dict) else ck
    m.load_state_dict(sd, strict=False)
    return m.eval()

def _preprocess():
    from torchvision import transforms
    return transforms.Compose([transforms.Resize((900, 900)), transforms.ToTensor()])

def _palette(label_colors):
    names = list(label_colors.keys())
    idx2color = np.array([[0, 0, 0]] + [label_colors[n]['rgb'] for n in names], dtype=np.uint8)
    return (names, idx2color)

def segment_image(img, model, label_colors):
    import torch
    names, idx2color = _palette(label_colors)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = _preprocess()(img).unsqueeze(0).to(dev)
    with torch.no_grad():
        pred = model.to(dev)(x)['out'].argmax(1)[0].cpu().numpy()
    counts = np.bincount(pred.ravel(), minlength=len(idx2color))
    ratios = {names[i - 1]: float(counts[i] / pred.size) for i in range(1, len(idx2color))}
    return (idx2color[pred], ratios)

def segment_folder(frames_dir, model, label_colors, limit=None):
    import torch
    from PIL import Image
    names, idx2color = _palette(label_colors)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(dev)
    paths = sorted(Path(frames_dir).glob('*.jpg'))
    if limit:
        paths = paths[:limit]
    pre = _preprocess()
    rows = []
    with torch.no_grad():
        for p in paths:
            x = pre(Image.open(p).convert('RGB')).unsqueeze(0).to(dev)
            pred = model(x)['out'].argmax(1)[0].cpu().numpy()
            counts = np.bincount(pred.ravel(), minlength=len(idx2color))
            row = {'filename': p.name}
            row.update({names[i - 1]: float(counts[i] / pred.size) for i in range(1, len(idx2color))})
            rows.append(row)
    return pd.DataFrame(rows).rename(columns={'Vegetation': 'green'})

def load_viewpoints(path):
    pv = pd.read_parquet(path)
    pv['green'] = pv['r_Vegetation']
    return pv

def load_buildings(path):
    import geopandas as gpd
    return gpd.read_file(path, layer='buildings')

def load_houston(path):
    return pd.read_csv(path)

def compare_cities(g_phx, hou):
    def s(d):
        return {'green~LST': round(float(d.green.corr(d.lst)), 2), 'SVI~LST': round(float(d.svi.corr(d.lst)), 2), 'n': int(len(d))}
    return {'Phoenix': s(g_phx), 'Houston': s(hou)}

def per_building_green_ndvi(pv):
    d = pv[pv.zcta.isin(REAL)]
    return d.groupby(['osm_id', 'zcta']).agg(green=('green', 'mean'), ndvi=('ndvi', 'mean')).reset_index()

def heat_divergence(g):
    hidden = g[(g.ndvi_pct < 0.33) & (g.green_pct > 0.66)]
    exposed = g[(g.ndvi_pct > 0.66) & (g.green_pct < 0.33)]
    typ = g[g.green_pct.between(0.33, 0.66)]
    return {'hidden_green': (hidden.lst.mean(), len(hidden)), 'typical': (typ.lst.mean(), len(typ)), 'exposed': (exposed.lst.mean(), len(exposed))}

def equity_split(g):
    p = g['svi'].rank(pct=True)
    hi, lo = (g[p > 0.66], g[p < 0.33])
    return {'most_vulnerable': (hi.green.mean(), hi.lst.mean(), len(hi)), 'least_vulnerable': (lo.green.mean(), lo.lst.mean(), len(lo))}

def double_burden(g):
    return g[(g.green_lisa == 'LL') & (g.lst_lisa == 'HH')]

def morans(g, cols=('green', 'lst', 'svi'), k=8, permutations=199):
    import libpysal
    from esda.moran import Moran
    W = libpysal.weights.KNN.from_dataframe(g, k=k)
    W.transform = 'r'
    return {c: Moran(g[c].values, W, permutations=permutations) for c in cols}

def plot_green_vs_ndvi(pb, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    r = pb['green'].corr(pb['ndvi'])
    for z in REAL:
        s = pb[pb.zcta == z]
        ax.scatter(s.ndvi, s.green, s=10, alpha=0.5, color=COLORS[z], label=NBHD[z])
    ax.set_xlabel('top-down NDVI')
    ax.set_ylabel('window-view green ratio')
    ax.set_title(f'Window-view green vs. top-down NDVI (r = {r:.2f})')
    ax.legend()
    return ax

def plot_green_by_floor(pv, ax=None, zctas=REAL):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    d = (pv if zctas is None else pv[pv.zcta.isin(zctas)]).copy()
    bins = [0, 1, 3, 6, 10, 1000]
    labs = ['1\n(street)', '2-3', '4-6', '7-10', '11+']
    d['fb'] = pd.cut(d['floor'], bins=bins, labels=labs)
    xp = {l: i for i, l in enumerate(labs)}
    for rc, col in [('low_rise', '#2c7fb8'), ('mid_rise', '#31a354'), ('high_rise', '#d95f0e')]:
        s = d[d.rise_class == rc]
        gg = s.groupby('fb', observed=True)['green'].agg(['mean', 'count'])
        gg = gg[gg['count'] >= 15]
        ax.plot([xp[i] for i in gg.index], gg['mean'], marker='o', lw=2.2, color=col, label=rc)
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs)
    ax.set_xlabel('floor (binned)')
    ax.set_ylabel('mean window-view green ratio')
    ax.set_title('Mean window-view green by floor and building form')
    ax.legend()
    return ax

def plot_equity_corner(g, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(g.green, g.lst, c=g.svi, cmap='plasma', s=12, alpha=0.7, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label='social vulnerability (SVI)')
    ax.set_xlabel('window-view green')
    ax.set_ylabel('surface temperature (C)')
    ax.set_title('Window-view green vs. surface temperature, colored by vulnerability')
    return ax

def plot_maps(g, basemap=True):
    gm = g.to_crs(3857)
    cents = {}
    if 'zcta' in gm.columns:
        cents = {z: (sub.geometry.x.mean(), sub.geometry.y.mean()) for z, sub in gm.groupby(gm['zcta'].astype(str)) if z in NBHD}
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    for ax, (col, cmap, ttl) in zip(axes.ravel(), [('green', 'Greens', 'window-view green'), ('lst', 'inferno', 'surface heat (LST)'), ('svi', 'plasma', 'social vulnerability'), ('gn_gap', 'RdBu', 'green minus NDVI rank (divergence)')]):
        gm.plot(column=col, cmap=cmap, markersize=6, ax=ax, legend=True)
        if basemap:
            try:
                import contextily as cx
                cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, attribution=False)
            except Exception:
                pass
        for z, (px, py) in cents.items():
            ax.annotate(NBHD.get(z, z).split('(')[0].strip(), (px, py), ha='center', va='center', fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.5', alpha=0.85))
        ax.set_title(ttl)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig
