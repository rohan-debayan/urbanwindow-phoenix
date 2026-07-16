from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
REAL = ['85004', '85281', '85016']
NBHD = {'85004': 'Downtown (high-rise core)', '85281': 'Tempe / ASU (mid-rise)', '85016': 'Arcadia (suburban)'}
COLORS = {'85004': '#d95f0e', '85281': '#31a354', '85016': '#2c7fb8'}
SVI_THEMES = {'RPL_THEME1': 'socioeconomic status', 'RPL_THEME2': 'household characteristics', 'RPL_THEME3': 'racial & ethnic minority', 'RPL_THEME4': 'housing & transport'}


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

# Full-name labels for the drivers window-view green is compared against.
DRIVER_LABELS = {'ndvi': 'NDVI', 'lst': 'urban heat', 'svi': 'social vulnerability'}

def green_correlations(cities, drivers=('ndvi', 'lst', 'svi')):
    """Building-level correlation of window-view green with each driver, per city.
    `cities` is a dict {name: df}. Drivers missing from a city's table (e.g.
    Houston NDVI, pending) come back as NaN rather than raising."""
    out = {}
    for name, d in cities.items():
        out[name] = {
            f'green ~ {DRIVER_LABELS.get(x, x)}': (round(float(d['green'].corr(d[x])), 2) if x in d.columns else float('nan'))
            for x in drivers
        }
    return pd.DataFrame(out)

RISE_ORDER = ['low_rise', 'mid_rise', 'high_rise']
RISE_LABELS = {'low_rise': 'low-rise', 'mid_rise': 'mid-rise', 'high_rise': 'high-rise'}

def _norm_rise(d):
    return d['rise_class'].astype(str).str.replace('-', '_')

def green_correlations_by_rise(cities, drivers=('ndvi', 'lst', 'svi')):
    """green~driver correlation split by rise class, per city. Returns a MultiIndex
    DataFrame: index = rise class (low/mid/high), columns = (city, driver-label) with
    a trailing (city, 'n') giving the buildings in that class. Missing drivers or
    classes with <2 buildings come back NaN."""
    frames = {}
    for name, d in cities.items():
        rc = _norm_rise(d)
        rows = {}
        for key in RISE_ORDER:
            sub = d[rc == key]
            row = {DRIVER_LABELS.get(x, x): (round(float(sub['green'].corr(sub[x])), 2)
                                             if (x in sub.columns and len(sub) > 1) else float('nan'))
                   for x in drivers}
            row['n'] = len(sub)
            rows[RISE_LABELS[key]] = row
        frames[name] = pd.DataFrame(rows).T
    return pd.concat(frames, axis=1)

def plot_green_correlations_by_rise(cities, drivers=('ndvi', 'lst', 'svi')):
    """One panel per driver; x = rise class, grouped bars = cities. NaN (missing
    driver or too few buildings) draws no bar and is annotated 'n/a'."""
    tbl = green_correlations_by_rise(cities, drivers)
    names = list(cities.keys())
    rise_lab = list(tbl.index)
    x = np.arange(len(rise_lab))
    w = 0.8 / len(names)
    fig, axes = plt.subplots(1, len(drivers), figsize=(5 * len(drivers), 4.5), sharey=True)
    if len(drivers) == 1:
        axes = [axes]
    for j, drv in enumerate(drivers):
        ax, label = axes[j], DRIVER_LABELS.get(drv, drv)
        for i, name in enumerate(names):
            vals = tbl[(name, label)].to_numpy(dtype=float)
            pos = x + (i - (len(names) - 1) / 2) * w
            ax.bar(pos, np.nan_to_num(vals), width=w, label=name)
            for px, v in zip(pos, vals):
                if np.isnan(v):
                    ax.text(px, 0.005, 'n/a', rotation=90, ha='center', va='bottom', fontsize=7, color='gray')
                else:
                    ax.text(px, v + (0.01 if v >= 0 else -0.01), f'{v:+.2f}',
                            ha='center', va='bottom' if v >= 0 else 'top', fontsize=7)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(rise_lab)
        ax.set_title(f'green ~ {label}')
        if j == 0:
            ax.set_ylabel('Pearson r  (with window-view green)')
    axes[-1].legend()
    fig.suptitle('Green correlations by rise class: Phoenix vs Houston', y=1.02)
    fig.tight_layout()
    return fig

def plot_green_correlations(cities, drivers=('ndvi', 'lst', 'svi'), ax=None):
    """Grouped bar chart of window-view green's correlation with each driver,
    Phoenix vs Houston. Missing drivers (NaN, e.g. Houston NDVI) draw no bar and
    are annotated 'pending'."""
    df = green_correlations(cities, drivers)          # rows = 'green ~ <label>', cols = cities
    names = list(df.columns)
    labels = [DRIVER_LABELS.get(x, x) for x in drivers]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(drivers))
    w = 0.8 / len(names)
    for i, name in enumerate(names):
        vals = df[name].to_numpy(dtype=float)
        pos = x + (i - (len(names) - 1) / 2) * w
        ax.bar(pos, np.nan_to_num(vals), width=w, label=name)
        for px, v in zip(pos, vals):
            if np.isnan(v):
                ax.text(px, 0.005, 'pending', rotation=90, ha='center', va='bottom', fontsize=8, color='gray')
            else:
                ax.text(px, v + (0.01 if v >= 0 else -0.01), f'{v:+.2f}',
                        ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Pearson r  (with window-view green)')
    ax.set_title('What window-view green tracks with: Phoenix vs Houston')
    ax.legend()
    return ax

def city_summary(phx, hou):
    """Building-level summary comparing Phoenix and Houston on shared metrics.
    Means for the continuous window-view/heat/vulnerability fields, counts for
    rise_class (labels normalized to underscores), and total buildings last."""
    def col(d):
        n = len(d)
        rc = d['rise_class'].astype(str).str.replace('-', '_').value_counts()
        def cp(key):
            c = int(rc.get(key, 0))
            return f'{c:,} ({c / n:.1%})'
        return {
            'Window green (mean)': f'{d.green.mean():.3f}',
            'LST (mean)': f'{d.lst.mean():.2f}',
            'SVI (mean)': f'{d.svi.mean():.3f}',
            'Low-rise (count, %)': cp('low_rise'),
            'Mid-rise (count, %)': cp('mid_rise'),
            'High-rise (count, %)': cp('high_rise'),
            'Total buildings': f'{n:,}',
        }
    return pd.DataFrame({'Phoenix': col(phx), 'Houston': col(hou)})

def per_building_green_ndvi(pv):
    d = pv[pv.zcta.isin(REAL)]
    return d.groupby(['osm_id', 'zcta']).agg(green=('green', 'mean'), ndvi=('ndvi', 'mean')).reset_index()

def heat_divergence(g):
    hidden = g[(g.ndvi_pct < 0.33) & (g.green_pct > 0.66)]
    exposed = g[(g.ndvi_pct > 0.66) & (g.green_pct < 0.33)]
    typ = g[g.green_pct.between(0.33, 0.66)]
    return {'hidden_green': (hidden.lst.mean(), len(hidden)), 'typical': (typ.lst.mean(), len(typ)), 'exposed': (exposed.lst.mean(), len(exposed))}

def equity_split(g, col='svi'):
    p = g[col].rank(pct=True)
    hi, lo = (g[p > 0.66], g[p < 0.33])
    return {'most_vulnerable': (hi.green.mean(), hi.lst.mean(), len(hi)), 'least_vulnerable': (lo.green.mean(), lo.lst.mean(), len(lo))}

def attach_svi_themes(g, svi_path):
    import geopandas as gpd
    themes = list(SVI_THEMES)
    pts = gpd.GeoDataFrame({'_i': range(len(g))}, geometry=gpd.points_from_xy(g['lon'], g['lat']), crs='EPSG:4326')
    svi = gpd.read_file(svi_path).to_crs('EPSG:4326')
    j = gpd.sjoin(pts, svi[['GEOID'] + themes + ['geometry']], how='left', predicate='within')
    j = j[~j['_i'].duplicated(keep='first')].set_index('_i')
    out = g.copy()
    out['GEOID'] = j['GEOID'].to_numpy()
    for t in themes:
        out[t] = j[t].to_numpy()
    return out

def svi_theme_summary(g, targets=('green', 'lst')):
    rows = []
    for key in ['svi'] + list(SVI_THEMES):
        name = 'composite (RPL_THEMES)' if key == 'svi' else f'{key} ({SVI_THEMES[key]})'
        row = {'theme': name}
        row.update({f'r~{t}': round(float(g[key].corr(g[t])), 3) for t in targets})
        rows.append(row)
    return pd.DataFrame(rows)

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

def to_tract(g, cols=('green', 'lst', 'svi')):
    preds = list(SVI_THEMES)
    agg = list(dict.fromkeys(list(cols) + preds))
    return g.dropna(subset=['GEOID']).groupby('GEOID')[agg].mean().reset_index()

def svi_relimp(g, target='green'):
    from itertools import combinations
    from math import factorial
    preds = list(SVI_THEMES)
    d = g.dropna(subset=preds + [target])
    y = d[target].to_numpy(float) - d[target].mean()
    X = {p: d[p].to_numpy(float) - d[p].mean() for p in preds}
    def r2(cols):
        if not cols:
            return 0.0
        A = np.column_stack([X[c] for c in cols])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        return 1 - ((y - A @ beta) ** 2).sum() / (y ** 2).sum()
    k = len(preds)
    share = {p: 0.0 for p in preds}
    for p in preds:
        rest = [q for q in preds if q != p]
        for r in range(len(rest) + 1):
            for S in combinations(rest, r):
                w = factorial(len(S)) * factorial(k - len(S) - 1) / factorial(k)
                share[p] += w * (r2(list(S) + [p]) - r2(list(S)))
    total = r2(preds)
    out = pd.DataFrame({'theme': [f'{p} ({SVI_THEMES[p]})' for p in preds],
                        'lmg_R2_share': [round(share[p], 4) for p in preds]})
    out['pct_of_explained'] = (out['lmg_R2_share'] / total * 100).round(1)
    out.attrs['R2_total'] = round(total, 4)
    return out

def svi_regression(g, level='tract', target='green'):
    import statsmodels.api as sm
    preds = list(SVI_THEMES)
    d = g.dropna(subset=preds + [target, 'GEOID']).copy()
    if level == 'tract':
        d = d.groupby('GEOID')[preds + [target]].mean().reset_index()
    z = lambda a: (a - a.mean()) / a.std()
    X = sm.add_constant(np.column_stack([z(d[p].to_numpy(float)) for p in preds]))
    y = z(d[target].to_numpy(float))
    if level == 'tract':
        m = sm.OLS(y, X).fit()
    else:
        m = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': d['GEOID'].to_numpy()})
    return pd.DataFrame({'theme': preds,
                         'std_beta': np.round(m.params[1:], 3),
                         'p': np.round(m.pvalues[1:], 4),
                         'R2': round(m.rsquared, 3),
                         'n': int(m.nobs)})

def plot_svi_theme_maps_pts(g, basemap=True):
    gm = g.to_crs(3857)
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    for ax, k in zip(axes.ravel(), list(SVI_THEMES)):
        gm.plot(column=k, cmap='plasma', markersize=2, ax=ax, legend=True, vmin=0, vmax=1)
        if basemap:
            try:
                import contextily as cx
                cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, attribution=False)
            except Exception:
                pass
        ax.set_title(f'{k}: {SVI_THEMES[k]}')
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('SVI sub-themes (per building)', fontsize=14)
    fig.tight_layout()
    return fig

def to_gdf(df, lon='lon', lat='lat', crs='EPSG:4326'):
    """Build a point GeoDataFrame from a lon/lat table (e.g. the Houston CSV)."""
    import geopandas as gpd
    return gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df[lon], df[lat]), crs=crs)

def plot_svi_green_maps_pts(g, basemap=True, title='Composite SVI vs. window-green (per building)'):
    if getattr(g, 'geometry', None) is None:   # plain lon/lat table (e.g. Houston CSV)
        g = to_gdf(g)
    gm = g.to_crs(3857)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, (col, cmap, ttl) in zip(axes, [('svi', 'plasma', 'composite SVI'), ('green', 'Greens', 'window-view green')]):
        gm.plot(column=col, cmap=cmap, markersize=2, ax=ax, legend=True)
        if basemap:
            try:
                import contextily as cx
                cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, attribution=False)
            except Exception:
                pass
        ax.set_title(ttl)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig

def plot_svi_green_maps_by_rise(g, basemap=True, title='SVI vs. window-green by rise class (per building)'):
    """The SVI-vs-green equity map, one row per rise class (low/mid/high) x two
    columns (composite SVI, window-green). Green shares one colour scale across
    rows so classes are comparable; building counts are shown per panel."""
    if getattr(g, 'geometry', None) is None:   # plain lon/lat table (e.g. Houston CSV)
        g = to_gdf(g)
    gm = g.to_crs(3857)
    rc = _norm_rise(gm)
    gvmax = float(gm['green'].quantile(0.98))
    fig, axes = plt.subplots(3, 2, figsize=(13, 18))
    for row, key in enumerate(RISE_ORDER):
        sub = gm[rc == key]
        specs = [('svi', 'plasma', 'composite SVI', 0, 1), ('green', 'Greens', 'window-view green', 0, gvmax)]
        for ax, (col, cmap, ttl, vmn, vmx) in zip(axes[row], specs):
            sub.plot(column=col, cmap=cmap, markersize=3, ax=ax, legend=True, vmin=vmn, vmax=vmx)
            if basemap:
                try:
                    import contextily as cx
                    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, attribution=False)
                except Exception:
                    pass
            ax.set_title(f'{RISE_LABELS[key]} (n={len(sub):,}) — {ttl}')
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig

def plot_tract_svi_green(svi_path, g, basemap=True):
    import geopandas as gpd
    gt = to_tract(g)[['GEOID', 'green']]
    svi = gpd.read_file(svi_path)
    svi = svi[svi['GEOID'].isin(set(g['GEOID'].dropna()))].merge(gt, on='GEOID', how='left').to_crs(3857)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, (col, cmap, ttl) in zip(axes, [('RPL_THEMES', 'plasma', 'composite SVI'), ('green', 'Greens', 'window-view green (tract mean)')]):
        svi.plot(column=col, cmap=cmap, ax=ax, legend=True, alpha=0.75, missing_kwds={'color': 'lightgrey'})
        if basemap:
            try:
                import contextily as cx
                cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, attribution=False)
            except Exception:
                pass
        ax.set_title(ttl)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Composite vulnerability vs. window-green, by census tract', fontsize=13)
    fig.tight_layout()
    return fig

def plot_svi_theme_maps(svi_path, geoids=None, basemap=True):
    import geopandas as gpd
    svi = gpd.read_file(svi_path)
    if geoids is not None:
        svi = svi[svi['GEOID'].isin(set(geoids))]
    svi = svi.to_crs(3857)
    keys = list(SVI_THEMES)
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    for ax, k in zip(axes.ravel(), keys):
        svi.plot(column=k, cmap='plasma', ax=ax, legend=True, vmin=0, vmax=1, alpha=0.75,
                 missing_kwds={'color': 'lightgrey'})
        if basemap:
            try:
                import contextily as cx
                cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, attribution=False)
            except Exception:
                pass
        ax.set_title(f'{k}: {SVI_THEMES[k]}')
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('SVI sub-themes by census tract', fontsize=14)
    fig.tight_layout()
    return fig

def _scatter_fit(ax, x, y, color):
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    sparse = len(x) < 1500  # tract level has few points; give them larger markers
    ax.scatter(x, y, s=30 if sparse else 6, alpha=0.6 if sparse else 0.2, color=color, edgecolors='none')
    slope, intr = np.polyfit(x, y, 1)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, slope * xs + intr, color='0.15', lw=2)
    return np.corrcoef(x, y)[0, 1]

def plot_svi_vs_green(g, col='svi', ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    r = _scatter_fit(ax, g[col].to_numpy(dtype=float), g['green'].to_numpy(dtype=float), '#2c7fb8')
    name = 'composite SVI' if col == 'svi' else f'{col} ({SVI_THEMES.get(col, col)})'
    ax.set_xlabel(name)
    ax.set_ylabel('window-view green ratio')
    ax.set_title(f'Window-view green vs. {name}  (r = {r:.2f})')
    return ax

def plot_svi_themes_vs_green(g):
    keys = list(SVI_THEMES)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, k in zip(axes.ravel(), keys):
        r = _scatter_fit(ax, g[k].to_numpy(dtype=float), g['green'].to_numpy(dtype=float), '#31a354')
        ax.set_xlabel(f'{k} percentile')
        ax.set_ylabel('window-view green ratio')
        ax.set_title(f'{k}: {SVI_THEMES[k]}  (r = {r:.2f})')
    fig.suptitle('Window-view green vs. SVI sub-themes', fontsize=14)
    fig.tight_layout()
    return fig

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


def plot_con_green_by_floor(
    pv,
    ax=None,
    zctas=None,
    min_buildings=5,
    building_weighted=True,
    max_floor=None,
    show_counts=False
):
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 6))
    # Geographic subset
    if zctas is None:
        d = pv.copy()
    else:
        d = pv[pv['zcta'].isin(zctas)].copy()
    required = ['osm_id', 'floor', 'green', 'rise_class']
    d = d.dropna(subset=required).copy()
    d['floor'] = pd.to_numeric(d['floor'], errors='coerce')
    d['green'] = pd.to_numeric(d['green'], errors='coerce')
    d = d.dropna(subset=['floor', 'green'])
    d = d[d['floor'] >= 1]
    if max_floor is not None:
        d = d[d['floor'] <= max_floor]
    # Make floors integer-valued
    d['floor'] = d['floor'].round().astype(int)
    # Give each building equal weight at each floor
    if building_weighted:
        d = (
            d.groupby(
                ['osm_id', 'rise_class', 'floor'],
                observed=True,
                as_index=False
            )
            .agg(green=('green', 'mean'))
        )
    styles = [
        ('low_rise', '#2c7fb8'),
        ('mid_rise', '#31a354'),
        ('high_rise', '#d95f0e')
    ]
    plotted_max_floor = 1
    for rise_class, color in styles:
        s = d[d['rise_class'] == rise_class].copy()
        summary = (
            s.groupby('floor', observed=True)
            .agg(
                mean=('green', 'mean'),
                std=('green', 'std'),
                n_buildings=('osm_id', 'nunique')
            )
            .reset_index()
            .sort_values('floor')
        )
        # Keep floors supported by enough unique buildings
        summary = summary[
            summary['n_buildings'] >= min_buildings
        ].copy()
        if summary.empty:
            continue
        summary['std'] = summary['std'].fillna(0)
        lower = (summary['mean'] - summary['std']).clip(lower=0)
        upper = (summary['mean'] + summary['std']).clip(upper=1)
        plotted_max_floor = max(
            plotted_max_floor,
            int(summary['floor'].max())
        )
        ax.fill_between(
            summary['floor'].to_numpy(),
            lower.to_numpy(),
            upper.to_numpy(),
            color=color,
            alpha=0.18,
            linewidth=0,
            label=f"{rise_class.replace('_', '-')} (±1 SD)"
        )
        ax.plot(
            summary['floor'],
            summary['mean'],
            color=color,
            marker='o',
            markersize=5,
            linewidth=2.2,
            label=rise_class.replace('_', '-')
        )
        if show_counts:
            for _, row in summary.iterrows():
                ax.annotate(
                    f"n={int(row['n_buildings'])}",
                    (row['floor'], row['mean']),
                    xytext=(0, 7),
                    textcoords='offset points',
                    ha='center',
                    fontsize=8
                )
    ax.set_xlabel('Floor')
    ax.set_ylabel('Mean simulated window-view green ratio')
    ax.set_title(
        'Simulated window-view green by floor and building form'
    )
    ax.set_xticks(
        range(1, plotted_max_floor + 1)
    )
    ax.set_xlim(
        0.7,
        plotted_max_floor + 0.3
    )
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)
    ax.legend()
    return ax



def plot_lst_by_floor(pv, g, ax=None, zctas=REAL):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    # Select the necessary building-level variables.
    building_lst = (
        g[['osm_id', 'lst']]
        .dropna(subset=['osm_id', 'lst'])
        .drop_duplicates(subset='osm_id')
    )
    # Add each building's LST to all viewpoints belonging to that building.
    d = pv.merge(
        building_lst,
        on='osm_id',
        how='left',
        validate='many_to_one'
    )
    # Apply the same geographic filter used by plot_green_by_floor().
    if zctas is not None:
        d = d[d['zcta'].isin(zctas)]

    d = d.dropna(subset=['floor', 'rise_class', 'lst']).copy()

    bins = [0, 1, 3, 6, 10, 1000]
    labs = ['1\n(street)', '2-3', '4-6', '7-10', '11+']

    d['fb'] = pd.cut(
        d['floor'],
        bins=bins,
        labels=labs
    )
    xp = {label: i for i, label in enumerate(labs)}

    for rc, col in [
        ('low_rise', '#2c7fb8'),
        ('mid_rise', '#31a354'),
        ('high_rise', '#d95f0e')
    ]:
        s = d[d['rise_class'] == rc]

        gg = (
            s.groupby('fb', observed=True)['lst']
            .agg(['mean', 'count'])
        )

        # Keep the same minimum-sample rule as the original function.
        gg = gg[gg['count'] >= 15]

        ax.plot(
            [xp[i] for i in gg.index],
            gg['mean'],
            marker='o',
            lw=2.2,
            color=col,
            label=rc
        )
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs)
    ax.set_xlabel('floor (binned)')
    ax.set_ylabel('mean land-surface temperature (°C)')
    ax.set_title(
        'Mean land-surface temperature by floor and building form'
    )
    ax.legend()

    return ax

def plot_lst_vs_green_by_floor(pv, g, ax=None, zctas=REAL):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    # Add building-level LST to each viewpoint
    building_lst = (
        g[['osm_id', 'lst']]
        .dropna(subset=['osm_id', 'lst'])
        .drop_duplicates('osm_id')
    )

    d = pv.merge(
        building_lst,
        on='osm_id',
        how='left',
        validate='many_to_one'
    )

    # Keep selected study areas
    if zctas is not None:
        d = d[d['zcta'].isin(zctas)]

    d = d.dropna(subset=['green', 'lst', 'floor']).copy()

    # Create floor categories
    bins = [0, 1, 3, 6, 10, 1000]
    labs = ['1 (street)', '2-3', '4-6', '7-10', '11+']

    d['floor_bin'] = pd.cut(
        d['floor'],
        bins=bins,
        labels=labs
    )

    # Reduce repeated viewpoints:
    # one observation per building and floor bin
    d_plot = (
        d.groupby(
            ['osm_id', 'floor_bin'],
            observed=True,
            as_index=False
        )
        .agg(
            green=('green', 'mean'),
            lst=('lst', 'first')
        )
    )

    for floor_bin in labs:
        s = d_plot[d_plot['floor_bin'] == floor_bin]
    
        if len(s) < 5:
            continue
    
        r = s['green'].corr(s['lst'])
    
        ax.scatter(
            s['green'],
            s['lst'],
            s=22,
            alpha=0.55,
            label=f'{floor_bin}: r={r:.2f}, n={len(s)}'
        )

    ax.set_xlabel('mean window-view green ratio')
    ax.set_ylabel('land-surface temperature (°C)')
    ax.set_title('Surface temperature vs. window-view green by floor')
    ax.legend(title='floor')

    return ax


def plot_lst_vs_green_by_building_form(pv, g, zctas=REAL):
    building_data = (
        g[['osm_id', 'lst']]
        .dropna(subset=['osm_id', 'lst'])
        .drop_duplicates('osm_id')
    )

    d = pv.merge(
        building_data,
        on='osm_id',
        how='left',
        validate='many_to_one'
    )

    if zctas is not None:
        d = d[d['zcta'].isin(zctas)]

    d = d.dropna(
        subset=['green', 'lst', 'floor', 'rise_class']
    ).copy()

    bins = [0, 1, 3, 6, 10, 1000]
    labs = ['1 (street)', '2-3', '4-6', '7-10', '11+']

    d['floor_bin'] = pd.cut(
        d['floor'],
        bins=bins,
        labels=labs
    )

    # Average the viewpoints for each building and floor group
    d_plot = (
        d.groupby(
            ['osm_id', 'rise_class', 'floor_bin'],
            observed=True,
            as_index=False
        )
        .agg(
            green=('green', 'mean'),
            lst=('lst', 'first')
        )
    )

    forms = ['low_rise', 'mid_rise', 'high_rise']

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 5),
        sharex=True,
        sharey=True
    )

    for ax, form in zip(axes, forms):
        form_data = d_plot[d_plot['rise_class'] == form]

        for floor_bin in labs:
            s = form_data[
                form_data['floor_bin'] == floor_bin
            ]

            if len(s) < 5:
                continue

            ax.scatter(
                s['green'],
                s['lst'],
                s=20,
                alpha=0.55,
                label=floor_bin
            )

        r = form_data['green'].corr(form_data['lst'])

        ax.set_title(
            f"{form.replace('_', ' ').title()}\n"
            f"overall r = {r:.2f}"
        )
        ax.set_xlabel('window-view green ratio')

    axes[0].set_ylabel('land-surface temperature (°C)')
    axes[-1].legend(title='floor')

    fig.suptitle(
        'Surface temperature vs. window-view green\n'
        'by floor and building form'
    )
    fig.tight_layout()

    return fig, axes


def prepare_citywide_heat_data(g_city):
    required = {"green", "ndvi", "svi", "lst"}
    missing = required.difference(g_city.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    uhi = g_city.dropna(
        subset=["green", "ndvi", "svi", "lst"]
    ).copy()

    uhi["green_pct"] = uhi["green"].rank(pct=True)
    uhi["ndvi_pct"] = uhi["ndvi"].rank(pct=True)

    return uhi

def heat_divergence_city(uhi):
    hidden = uhi[
        (uhi["ndvi_pct"] < 0.33) &
        (uhi["green_pct"] > 0.66)
    ]

    exposed = uhi[
        (uhi["ndvi_pct"] > 0.66) &
        (uhi["green_pct"] < 0.33)
    ]

    typical = uhi[
        uhi["green_pct"].between(0.33, 0.66)
    ]

    return {
        "hidden_green": (hidden["lst"].mean(), len(hidden)),
        "typical": (typical["lst"].mean(), len(typical)),
        "exposed": (exposed["lst"].mean(), len(exposed))
    }


def per_building_green_ndvi_city(g_city):
    return (
        g_city.dropna(subset=["osm_id", "green", "ndvi"])
              .groupby("osm_id", as_index=False)
              .agg(
                  green=("green", "mean"),
                  ndvi=("ndvi", "mean")
              )
    )

def plot_green_vs_ndvi_city(pb, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    r = pb["green"].corr(pb["ndvi"])

    ax.scatter(
        pb["ndvi"],
        pb["green"],
        s=10,
        alpha=0.4
    )

    ax.set_xlabel("Top-down NDVI")
    ax.set_ylabel("Building green fraction")
    ax.set_title(f"City-wide green fraction vs NDVI (r = {r:.2f})")

    return ax


def plot_heat_panels(g_city):
    import numpy as np
    from matplotlib.lines import Line2D
    from scipy.stats import sem, t
    group_order = ['Hidden Green', 'Typical', 'Exposed']
    group_colors = {'Hidden Green': 'lime', 'Typical': 'Olive', 'Exposed': '#B2182B', 'Other': '#D9D9D9'}
    g = g_city.dropna(subset=['green', 'ndvi', 'lst', 'geometry']).copy()
    g['green_pct'] = g['green'].rank(pct=True)
    g['ndvi_pct'] = g['ndvi'].rank(pct=True)
    cond = [(g['ndvi_pct'] < 0.33) & (g['green_pct'] > 0.66),
            (g['ndvi_pct'] > 0.66) & (g['green_pct'] < 0.33),
            g['green_pct'].between(0.33, 0.66)]
    g['heat_group'] = np.select(cond, ['Hidden Green', 'Exposed', 'Typical'], default='Other')
    rows = []
    for grp in group_order:
        v = g.loc[g['heat_group'] == grp, 'lst'].dropna()
        n = len(v)
        ci = float(t.ppf(0.975, df=n - 1) * sem(v)) if n > 1 else float('nan')
        rows.append({'Group': grp, 'Mean_LST': v.mean(), 'Median_LST': v.median(),
                     'SD': v.std(ddof=1), 'CI95': ci, 'n': n})
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    data = [g.loc[g['heat_group'] == grp, 'lst'].dropna().values for grp in group_order]
    labels_n = ['%s\n(n=%d)' % (grp, len(d)) for grp, d in zip(group_order, data)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axm, axv, axs = axes
    for grp in group_order:
        gd = g[g['heat_group'] == grp]
        if len(gd):
            gd.plot(ax=axm, color=group_colors[grp], edgecolor='none', alpha=0.5)
    handles = [Line2D([0], [0], marker='o', linestyle='', markerfacecolor=group_colors[grp],
                      markeredgecolor='none', markersize=7, label=grp) for grp in group_order]
    axm.legend(handles=handles, title='Vegetation-view category', loc='upper left',
               frameon=True, fontsize=9, title_fontsize=9)
    axm.set_title('A. Spatial distribution')
    axm.set_axis_off()
    pos = np.arange(1, len(group_order) + 1)
    vp = axv.violinplot(data, positions=pos, widths=0.8, showextrema=False)
    for body, grp in zip(vp['bodies'], group_order):
        body.set_facecolor(group_colors[grp])
        body.set_edgecolor('black')
        body.set_alpha(0.6)
    axv.boxplot(data, positions=pos, widths=0.2, patch_artist=True, showmeans=True)
    axv.set_xticks(pos)
    axv.set_xticklabels(labels_n)
    axv.set_ylabel('Land surface temperature (°C)')
    axv.set_title('B. LST distribution')
    axv.grid(axis='y', alpha=0.2)
    sc = axs.scatter(g['ndvi_pct'], g['green_pct'], c=g['lst'], cmap='coolwarm', s=8, alpha=0.6)
    for xv in (0.33, 0.66):
        axs.axvline(xv, color='black', linestyle='--')
    axs.axhline(0.33, color='black', linestyle='--')
    axs.set_xlabel('NDVI percentile')
    axs.set_ylabel('Visible green percentile')
    axs.set_title('C. Green versus NDVI')
    fig.colorbar(sc, ax=axs, label='Land surface temperature (°C)')
    fig.tight_layout()
    return fig, axes
