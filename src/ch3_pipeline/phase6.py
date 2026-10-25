"""Privacy-safe SVG and dashboard helpers for Phase 6."""
from __future__ import annotations

from hashlib import sha256
from html import escape
from pathlib import Path
from typing import Iterable, Sequence
import math

WIDTH, HEIGHT = 1600, 1000
PAD_L, PAD_R, PAD_T, PAD_B = 150, 70, 150, 150
BLUE, TEAL, ORANGE, RED, GREY = "#1769aa", "#00897b", "#ef6c00", "#c62828", "#607d8b"


def file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _ticks(maximum: float, n: int = 5) -> list[float]:
    if maximum <= 0:
        return [0.0]
    raw = maximum / n
    power = 10 ** math.floor(math.log10(raw))
    step = math.ceil(raw / power) * power
    return [i * step for i in range(n + 1) if i * step <= maximum + step]


def _fmt(v: float) -> str:
    if abs(v) >= 1_000_000: return f"{v/1_000_000:.1f}M"
    if abs(v) >= 1_000: return f"{v/1_000:.1f}k"
    return f"{v:g}"


def bar_svg(path: Path, title: str, subtitle: str, labels: Sequence[str], values: Sequence[float],
            x_label: str, y_label: str, source: str, denominator: str,
            colors: Sequence[str] | None = None, horizontal: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = list(colors or [BLUE] * len(values))
    maximum = max(values) if len(values) else 1
    out = _base(title, subtitle)
    if horizontal:
        plot_w, plot_h = WIDTH-PAD_L-PAD_R, HEIGHT-PAD_T-PAD_B
        bh = plot_h / max(len(values), 1) * .68
        for i,(lab,val) in enumerate(zip(labels,values)):
            y=PAD_T+(i+.16)*plot_h/max(len(values),1); w=(val/maximum)*plot_w
            out.append(f'<rect x="{PAD_L}" y="{y:.1f}" width="{w:.1f}" height="{bh:.1f}" fill="{colors[i%len(colors)]}"/>')
            out.append(_text(PAD_L-12,y+bh*.72,escape(str(lab)),"end",24))
            out.append(_text(PAD_L+w+10,y+bh*.72,_fmt(val),"start",22))
        out.append(_text((PAD_L+WIDTH-PAD_R)/2,HEIGHT-95,escape(x_label),"middle",26))
        out.append(_text(35,(PAD_T+HEIGHT-PAD_B)/2,escape(y_label),"middle",26,' transform="rotate(-90 35 500)"'))
    else:
        out += _axes(maximum, x_label, y_label)
        plot_w, plot_h = WIDTH-PAD_L-PAD_R, HEIGHT-PAD_T-PAD_B
        slot=plot_w/max(len(values),1); bw=slot*.72
        for i,(lab,val) in enumerate(zip(labels,values)):
            x=PAD_L+i*slot+slot*.14; h=(val/maximum)*plot_h; y=HEIGHT-PAD_B-h
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{colors[i%len(colors)]}"/>')
            angle=-45 if len(labels)>12 else 0
            extra=f' transform="rotate({angle} {x+bw/2:.1f} {HEIGHT-PAD_B+30})"' if angle else ''
            out.append(_text(x+bw/2,HEIGHT-PAD_B+30,escape(str(lab)),"middle",20,extra))
    out += _footer(source,denominator)
    _write(path,out)


def line_svg(path: Path, title: str, subtitle: str, labels: Sequence[str], series: dict[str, Sequence[float]],
             x_label: str, y_label: str, source: str, denominator: str, log_y: bool=False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    transformed={k:[math.log10(max(v,1)) if log_y else float(v) for v in vals] for k,vals in series.items()}
    maximum=max((max(v) for v in transformed.values()),default=1)
    out=_base(title,subtitle)+_axes(maximum,x_label,y_label + (" (log10 scale)" if log_y else ""))
    plot_w,plot_h=WIDTH-PAD_L-PAD_R,HEIGHT-PAD_T-PAD_B
    palette=[BLUE,TEAL,ORANGE,RED,"#6a1b9a",GREY]
    for si,(name,vals) in enumerate(transformed.items()):
        pts=[]
        for i,v in enumerate(vals):
            x=PAD_L+i*plot_w/max(len(vals)-1,1); y=HEIGHT-PAD_B-(v/maximum)*plot_h
            pts.append(f"{x:.1f},{y:.1f}")
        color=palette[si%len(palette)]
        out.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="5"/>')
        for p in pts: out.append(f'<circle cx="{p.split(",")[0]}" cy="{p.split(",")[1]}" r="5" fill="{color}"/>')
        out.append(f'<rect x="{WIDTH-410}" y="{PAD_T+si*38}" width="24" height="8" fill="{color}"/>')
        out.append(_text(WIDTH-375,PAD_T+si*38+10,escape(name),"start",22))
    step=max(1,math.ceil(len(labels)/12))
    for i,lab in enumerate(labels):
        if i%step: continue
        x=PAD_L+i*plot_w/max(len(labels)-1,1)
        out.append(_text(x,HEIGHT-PAD_B+32,escape(str(lab)),"middle",19))
    out += _footer(source,denominator)
    _write(path,out)


def map_svg(path: Path, title: str, subtitle: str, cameras: Sequence[tuple[str,float,float]],
            links: Sequence[tuple[str,str,float]] | None, source: str, denominator: str) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    lons=[x[1] for x in cameras]; lats=[x[2] for x in cameras]
    minx,maxx,miny,maxy=min(lons),max(lons),min(lats),max(lats)
    def xy(lon,lat):
        x=PAD_L+(lon-minx)/max(maxx-minx,1e-9)*(WIDTH-PAD_L-PAD_R)
        y=HEIGHT-PAD_B-(lat-miny)/max(maxy-miny,1e-9)*(HEIGHT-PAD_T-PAD_B)
        return x,y
    coords={str(cid):xy(lon,lat) for cid,lon,lat in cameras}
    out=_base(title,subtitle)
    out.append(f'<rect x="{PAD_L}" y="{PAD_T}" width="{WIDTH-PAD_L-PAD_R}" height="{HEIGHT-PAD_T-PAD_B}" fill="#f5f7f8" stroke="#9e9e9e"/>')
    if links:
        maxv=max(v for _,_,v in links)
        for a,b,v in links:
            if str(a) not in coords or str(b) not in coords: continue
            x1,y1=coords[str(a)];x2,y2=coords[str(b)]
            out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{ORANGE}" stroke-opacity="0.35" stroke-width="{1+8*v/maxv:.2f}"/>')
    for cid,lon,lat in cameras:
        x,y=coords[str(cid)]; out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{BLUE}" stroke="white" stroke-width="2"/>')
    out.append(_text(PAD_L,HEIGHT-PAD_B+42,"Longitude (WGS84 degrees)","start",24))
    out.append(_text(45,(PAD_T+HEIGHT-PAD_B)/2,"Latitude (WGS84 degrees)","middle",24,' transform="rotate(-90 45 500)"'))
    out += _footer(source,denominator)
    _write(path,out)


def _base(title: str, subtitle: str) -> list[str]:
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{escape(title)}">',
            '<rect width="100%" height="100%" fill="white"/>',_text(70,60,escape(title),"start",38),_text(70,105,escape(subtitle),"start",24)]


def _axes(maximum: float, x_label: str, y_label: str) -> list[str]:
    out=[f'<line x1="{PAD_L}" y1="{HEIGHT-PAD_B}" x2="{WIDTH-PAD_R}" y2="{HEIGHT-PAD_B}" stroke="#222" stroke-width="2"/>',
         f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{HEIGHT-PAD_B}" stroke="#222" stroke-width="2"/>']
    for t in _ticks(maximum):
        y=HEIGHT-PAD_B-(t/maximum)*(HEIGHT-PAD_T-PAD_B) if maximum else HEIGHT-PAD_B
        out.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{WIDTH-PAD_R}" y2="{y:.1f}" stroke="#ddd"/>')
        out.append(_text(PAD_L-15,y+7,_fmt(t),"end",20))
    out.append(_text((PAD_L+WIDTH-PAD_R)/2,HEIGHT-75,escape(x_label),"middle",26))
    out.append(_text(40,500,escape(y_label),"middle",26,' transform="rotate(-90 40 500)"'))
    return out


def _footer(source: str, denominator: str) -> list[str]:
    return [_text(70,930,escape(f"Denominator: {denominator}"),"start",19),_text(70,963,escape(f"Source/provenance: {source}"),"start",18)]


def _text(x: float,y: float,text: str,anchor: str,size: int,extra: str="") -> str:
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Arial, sans-serif" font-size="{size}" fill="#202124"{extra}>{text}</text>'


def _write(path: Path, parts: Iterable[str]) -> None:
    path.write_text("\n".join([*parts,"</svg>"])+"\n",encoding="utf-8")


def suppressed(values: Sequence[int], minimum_count: int=10) -> list[int | None]:
    return [int(v) if int(v) >= minimum_count else None for v in values]


def prohibited_identifier_text(text: str) -> bool:
    lowered=text.lower()
    return any(term in lowered for term in ("vehicle_token", "source_token", "sequence_key"))
