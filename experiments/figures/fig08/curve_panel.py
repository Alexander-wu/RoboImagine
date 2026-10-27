#!/usr/bin/env python3
"""Plot measured R2/SFT checkpoint evaluations without inventing intermediate values.

Input schema (data.json):
{
  "baseline": {"label": "SFT 50k", "metrics": {"psnr": ..., "ssim": ...,
    "lpips": ...}, "source": "path/to/verified/result.json"},
  "series": [
    {"id": "r2_hybrid", "label": "RoboImagine", "points": [
      {"step": 500, "metrics": {"psnr": ..., "ssim": ..., "lpips": ...},
       "source": "path/to/verified/result.json"}]},
    {"id": "continued_sft", "label": "Continued SFT", "points": []}
  ],
  "planned_steps": [0, 100, 250, 500],
  "protocol": {"dataset": "RT-1", "trajectories": 128,
    "prediction_frames": 32, "initial_sft_updates": 50000}
}

Only evaluated points belong in `points`. No null-valued placeholders are
accepted. Lines connect consecutive available measured checkpoints, including step 0.
Connecting segments are visual guides, not measurements at intervening steps.
No smoothing, extrapolation or confidence intervals are inferred. Editable SVG retains Garamond text; submission SVG outlines it.
"""

from __future__ import annotations

import argparse
import ctypes.util
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


INK = "#253141"
BASE = "#7C8792"
STYLES = {
    "r2_hybrid": {"color": "#9B84BC", "marker": "o", "linestyle": "-"},
    "continued_sft": {"color": "#225D91", "marker": "o", "linestyle": "-"},
    "mcts": {"color": "#BB7149", "marker": "^", "linestyle": "--"},
    "ordinary": {"color": "#268879", "marker": "s", "linestyle": "-"},
}
METRICS = [("psnr", "(a) PSNR", "Higher is better", 2),
           ("ssim", "(b) SSIM", "Higher is better", 4),
           ("lpips", "(c) LPIPS", "Lower is better", 4),
           ("mae", "(d) MAE", "Lower is better", 4)]


def get_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=HERE / "data.json")
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--font-dir", type=Path)
    parser.add_argument("--cairo-library", type=Path)
    parser.add_argument("--native-export", action="store_true",
                        help="Write vector PDF and PNG with Matplotlib; no local Cairo export helper required.")
    parser.add_argument("--width", type=float, default=8.0,
                        help="Figure width in inches (default: compact four-panel row).")
    parser.add_argument("--height", type=float, default=2.02)
    parser.add_argument("--preview-width", type=int, default=3200)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def validate(data):
    baseline = data["baseline"]
    protocol = data["protocol"]
    assert protocol["initial_sft_updates"] == 50000, "This figure uses the SFT50k initialization."
    assert protocol["trajectories"] > 0 and protocol["prediction_frames"] > 0
    planned = data.get("planned_steps", [0, 100, 250, 500])
    assert planned == sorted(set(planned)) and planned[0] == 0
    assert all(isinstance(step, int) and step >= 0 for step in planned)
    assert baseline.get("source"), "The shared baseline needs a source."
    groups = data["series"]
    assert {s["id"] for s in groups} == set(STYLES) and len(groups) == 4
    for record in [baseline] + [p for s in groups for p in s["points"]]:
        assert record.get("source"), "Every measured checkpoint needs a source."
        for key, _, _, _ in METRICS:
            value = record["metrics"][key]
            assert isinstance(value, (int, float)) and not isinstance(value, bool)
            assert math.isfinite(value), f"Invalid {key}: {value}"
        assert 0 <= record["metrics"]["ssim"] <= 1
        assert record["metrics"]["lpips"] >= 0
        if "protocol" in record:
            assert record["protocol"] == protocol, "Do not mix evaluation protocols."
    for series in groups:
        steps = [p["step"] for p in series["points"]]
        assert steps == sorted(set(steps)), "Checkpoint steps must be unique and ordered."
        assert all(step > 0 and step in planned for step in steps)
    return baseline, groups, planned, protocol


def register_fonts(requested):
    candidates = ([requested] if requested else []) + [
        HERE / "assets",
        HERE.parents[1]/"common/fonts",
        HERE.parent / "roboalign_r2_motivation_bridge_20261012/assets",
    ]
    for directory in candidates:
        if directory is None:
            continue
        loaded = {}
        for face in ["Regular", "Bold", "Italic", "BoldItalic"]:
            match = next((directory / f"EBGaramond-{face}.{ext}" for ext in ["otf", "ttf"]
                          if (directory / f"EBGaramond-{face}.{ext}").is_file()), None)
            if match:
                font_manager.fontManager.addfont(str(match))
                loaded[face] = str(match.resolve())
        if len(loaded) == 4:
            font_manager.findfont("EB Garamond", fallback_to_default=False)
            return loaded
    raise RuntimeError("All four EB Garamond faces are required; pass --font-dir.")


def draw(data, width, height, single=False):
    baseline, series, planned, protocol = validate(data)
    plt.rcParams.update({
        "font.family": "EB Garamond", "font.size": 9,
        "font.weight": "normal", "axes.labelsize": 9,
        "axes.titlesize": 11.2, "axes.titleweight": "bold",
        "xtick.labelsize": 8.2, "ytick.labelsize": 8.5,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": "#48596D", "ytick.color": "#48596D",
        "axes.linewidth": .7, "xtick.major.width": .65, "ytick.major.width": .65,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "pdf.fonttype": 42, "savefig.facecolor": "white",
    })
    keys = [next(m for m in METRICS if m[0]=="lpips")] if single else METRICS
    fig, axs = plt.subplots(1, len(keys), figsize=(width, height), squeeze=False)
    axes=axs[0]
    fig.subplots_adjust(left=.145 if single else .045, right=.994,
                        bottom=.155 if single else .115,
                        top=.74 if single else .707,
                        wspace=.30 if single else .25)
    reports=[]
    gains={}
    for ax,(key,title,direction,decimals) in zip(axes,keys):
        base=baseline["metrics"][key]
        measured_values=[base]+[p["metrics"][key] for group in series for p in group["points"]]
        low,high=min(measured_values),max(measured_values)
        span=max(high-low,{"psnr":.5,"ssim":.01,"lpips":.01,"mae":.005}[key])
        ax.set_ylim(low-.18*span,high+.30*span)
        ax.set_xlim(-70,580)
        ax.axvspan(-70,0,color="#E7F0FC",zorder=0)
        ax.axvspan(0,580,color="#F3EEFA",zorder=0)
        ax.axvline(0,color="#7E91AE",linestyle=(0,(4,3)),linewidth=1.0,zorder=2)
        ax.set_axisbelow(True)
        ax.grid(axis="y",color="#FFFFFF",linewidth=1.0)
        ax.spines[["top","right"]].set_visible(False)
        ax.spines[["left","bottom"]].set_color("#8B9BAF")
        ax.tick_params(pad=3)
        heading=("LPIPS" if single else title)+(" ↓" if key in ("lpips","mae") else " ↑")
        ax.set_title(heading,loc="left",pad=10 if single else 12,color=INK)
        ax.text(-35,.96,"SFT",transform=ax.get_xaxis_transform(),ha="center",va="top",
                color="#315F8D",fontsize=7.2 if not single else 7.8,weight="bold")
        ax.set_xticks(planned)
        ax.set_xticklabels(["50k/0"]+[str(x) for x in planned[1:]])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4,min_n_ticks=3,steps=[1,2,2.5,5,10]))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f" if key=="psnr" else "%.4f" if key=="mae" else "%.3f"))
        # Step 0 is a shared measured initialization, not a pretraining curve.
        ax.scatter([0],[base],s=36,color="white",edgecolor="#3B5573",linewidth=1.5,zorder=6)
        for group in series:
            color=STYLES[group["id"]]["color"]
            points=[{"step":0,"metrics":baseline["metrics"]}]+group["points"]
            measured={p["step"]:p["metrics"][key] for p in points}
            segments=[]
            # Connect observed checkpoints only. The segment itself does not
            # add observations or markers at the unevaluated steps it spans.
            available = sorted(measured)
            for left,right in zip(available[:-1],available[1:]):
                ax.plot([left,right],[measured[left],measured[right]],
                        color=color,linewidth=2.1 if group["id"]=="r2_hybrid" else 1.6,linestyle=STYLES[group["id"]]["linestyle"],zorder=4,
                        solid_capstyle="round")
                segments.append([left,right])
            for p in group["points"]:
                is_rl_end=group["id"]=="r2_hybrid" and p["step"]==planned[-1]
                ax.scatter([p["step"]],[p["metrics"][key]],
                           marker="*" if is_rl_end else STYLES[group["id"]]["marker"],
                           s=75 if is_rl_end else 25,
                           facecolor=color if is_rl_end else "white",
                           edgecolor=color,linewidth=1.4,zorder=7)
            if False:  # values remain in the accompanying data and tables
                last=group["points"][-1]
                offset=(-4,7)
                ax.annotate(f"{last['metrics'][key]:.{decimals}f}",
                            (last["step"],last["metrics"][key]),xytext=offset,
                            textcoords="offset points",ha="right",
                            va="top" if offset[1]<0 else "bottom",
                            fontsize=8.5 if not single else 10.2,weight="bold",color=color)
            if key==keys[0][0]:
                reports.append({"id":group["id"],"measured_steps":list(measured),
                                "drawn_line_segments":segments})
        rl=next(g for g in series if g["id"]=="r2_hybrid")
        final=next((p for p in rl["points"] if p["step"]==planned[-1]),None)
        if final:
            value=final["metrics"][key]
            delta=value-base
            label=(f"{delta:+.2f} dB" if key=="psnr" else
                   f"{delta:+.4f}" if key=="ssim" else
                   f"{delta/base*100:+.2f}%").replace("-","−")
            gains[key]={"baseline":base,"final":value,"absolute_change":delta,
                        "relative_percent":100*delta/base,"display":label,
                        "reference":"shared SFT 50k"}
            arrow_x=547
            # Dotted guides depict the reference comparison only, never a missing SFT curve.
            ax.plot([500,arrow_x],[value,value],color="#9179AE",linewidth=.8,linestyle=(0,(2,2)),zorder=2)
            ax.plot([510,arrow_x],[base,base],color="#9179AE",linewidth=.8,linestyle=(0,(2,2)),zorder=2)
            ax.annotate("",xy=(arrow_x,value),xytext=(arrow_x,base),
                        arrowprops={"arrowstyle":"<->","color":"#9B84BC","linewidth":1.5,
                                    "shrinkA":0,"shrinkB":0,"mutation_scale":8},zorder=3)
            ax.text(.58,.965,label,transform=ax.transAxes,ha="center",va="top",color="#BF3F45",
                    fontsize=10.8,weight="bold")
    handles=[Line2D([],[],color=STYLES[id]["color"],marker=STYLES[id]["marker"],markerfacecolor="white",
                    markeredgewidth=1.1,linewidth=1.8,linestyle=STYLES[id]["linestyle"],markersize=4.2,label=label)
             for id,label in [("continued_sft","Continued SFT"),("mcts","MCTS"),("ordinary","Ordinary beam"),("r2_hybrid","RoboImagine (GDBS)")]]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.52,.995),frameon=False,
               ncol=4,handlelength=1.7,handletextpad=.4,columnspacing=1.25,fontsize=9.2)
    # Scale only the curve-panel text; the lower case panel is added later.
    from matplotlib.text import Text
    for text_artist in fig.findobj(match=Text):
        text_artist.set_fontsize(text_artist.get_fontsize() * 0.92)
    return fig,reports,gains


def export_figure(fig, out_dir, stem, preview_width):
    outputs=[]
    for suffix,fontmode in [(".editable.svg","none"),(".svg","path")]:
        path=out_dir/(stem+suffix)
        with plt.rc_context({"svg.fonttype":fontmode}):
            fig.savefig(path,format="svg",metadata={"Creator":"Measured R2 training checkpoints"})
        outputs.append(str(path.resolve()))
    for suffix in [".pdf",".png"]:
        path=out_dir/(stem+suffix)
        fig.savefig(path,format=suffix[1:],dpi=preview_width/fig.get_figwidth())
        outputs.append(str(path.resolve()))
    plt.close(fig)
    return outputs

def main():
    args=get_args()
    raw=args.data.read_bytes()
    data=json.loads(raw)
    validate(data)
    if args.validate_only:
        print("DATA_VALID")
        return
    fonts=register_fonts(args.font_dir)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    fig,reports,gains=draw(data,args.width,args.height)
    outputs=export_figure(fig,args.out_dir,"figure",args.preview_width)
    fig,_,_=draw(data,3.45,2.65,single=True)
    outputs+=export_figure(fig,args.out_dir,"lpips",1500)
    metadata={"input":str(args.data.resolve()),"input_sha256":hashlib.sha256(raw).hexdigest(),
              "protocol":data["protocol"],"sources":data,"fonts":fonts,
              "figure_inches":[args.width,args.height],"series_rendering":reports,
              "gain_annotations":gains,"no_smoothing":True,"no_unmeasured_points":True,
              "no_pretraining_curve":True,
              "line_policy":"straight guides connecting observed checkpoints only",
              "export_backend":"matplotlib",
              "outputs":outputs}
    (args.out_dir/"render_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps({"state":"RENDERED","outputs":outputs},indent=2))


if __name__=="__main__":
    main()
