#!/usr/bin/env python3
"""Combine vector learning curves and unchanged, matched checkpoint imagery."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE/".mplconfig"))

import curve_panel
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
from PIL import Image

INK, BLUE, PURPLE, GOLD = "#253141", "#225D91", "#9B84BC", "#B98B36"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE/"curve_data.json")
    parser.add_argument("--assets", type=Path, default=HERE/'assets')
    parser.add_argument("--case", type=int, default=34)
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"),
                        help="Identical pixel ROI for all methods; must match the verified manifest")
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--font-dir", type=Path,
                        default=HERE.parents[1]/"common/fonts")
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    curve_panel.validate(data)
    assert data["status"] == "COMPLETE"
    assert all([p["step"] for p in group["points"]] == [100, 250, 500] for group in data["series"])
    manifest_path = args.assets/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    case = manifest["cases"][str(args.case)]
    assert manifest["selection"]["selected_index"] == args.case
    roi_xywh = args.roi or manifest["selection"]["roi_xywh"]
    assert manifest["selection"]["roi_xywh"] == roi_xywh
    assert len(roi_xywh) == 4
    x, y, width, height = roi_xywh
    assert x >= 0 and y >= 0 and width > 0 and height > 0
    assert x+width <= 320 and y+height <= 256
    assert case["sources"]["initial_sft"]["checkpoint_digest"] == data["protocol"]["initial_digest"]
    assert case["matched_gt_observations_instruction"] is True
    assert manifest["protocol"]["prediction_frame_offset"] == 32
    assert manifest["protocol"]["additional_updates"] == [100, 250, 500]
    fonts = curve_panel.register_fonts(args.font_dir)

    W, H, CURVE_H = 8.0, 6.12, 2.22
    fig, series_rendering, gains = curve_panel.draw(data, W, CURVE_H)
    # Reuse the native Matplotlib artists rather than embedding a curve image.
    original_positions = [ax.get_position().bounds for ax in fig.axes]
    fig.set_size_inches(W, H)
    for ax, (left, bottom, width, height) in zip(fig.axes, original_positions):
        ax.set_position([left, (H-CURVE_H+bottom*CURVE_H)/H,
                         width, height*CURVE_H/H])

    halo = [patheffects.Stroke(linewidth=1.8, foreground="white"),
            patheffects.Normal()]
    audit = []

    def text(x, y, value, **kwargs):
        return fig.text(x/W, y/H, value, **kwargs)

    def load(key):
        name = f"case_{args.case:03d}/{key}.png"
        entry = manifest["files"][name]
        path = args.assets/name
        assert path.relative_to(args.assets).as_posix() == name  # Portable relative path; original SHA check follows.
        assert sha(path) == entry["sha256"]
        assert entry["prediction_frame_offset"] == 32 and entry["frame_index"] == 31
        assert entry["dimensions"] == [320, 256]
        with Image.open(path) as source:
            assert source.size == (320, 256) and source.mode == "RGB"
            image = source.copy()
        return image, {"key": key, "file": str(path.resolve()), **entry,
                       "training_updates": 0 if key in ("gt", "initial_sft") else int(key.rsplit("_", 1)[1])}

    def image_ax(x, y, image, width=1.0, height=.8, color="#D8DCE3", thickness=.4):
        # Uniform 6% reduction within the original slot. Preserve every pixel
        # and the common full-frame / crop aspect ratio; no image retouching.
        scale=.94
        x += width*(1-scale)/2
        y += height*(1-scale)/2
        width *= scale
        height *= scale
        rim=.012
        # Offset vector shadows and a narrow white mat, behind the image axes.
        for dx,dy,alpha in [(.018,-.018,.055),(.009,-.009,.10)]:
            fig.add_artist(Rectangle(((x-rim+dx)/W,(y-rim+dy)/H),
                (width+2*rim)/W,(height+2*rim)/H,
                transform=fig.transFigure,facecolor="#405269",edgecolor="none",
                alpha=alpha,zorder=-.5))
        fig.add_artist(Rectangle(((x-rim)/W,(y-rim)/H),
            (width+2*rim)/W,(height+2*rim)/H,
            transform=fig.transFigure,facecolor="white",edgecolor="#D4DCE5",
            linewidth=.35,zorder=-.2))
        ax = fig.add_axes([x/W, y/H, width/W, height/H])
        ax.imshow(image, interpolation="none")
        ax.set_axis_off()
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               fill=False, edgecolor=color, linewidth=thickness,
                               clip_on=False, zorder=5))
        return ax

    x0, y0, rw, rh = roi_xywh
    x1, y1 = x0+rw, y0+rh
    assert rw/rh == 320/256

    def draw_frame(key, x, y, with_zoom):
        image, entry = load(key)
        ax = image_ax(x, y, image)
        ax.add_patch(Rectangle((x0-.5, y0-.5), rw, rh, fill=False,
                               edgecolor=GOLD, linewidth=.8,
                               path_effects=halo, zorder=7))
        record = {"source": entry, "roi_xywh": roi_xywh, "full_frame_unmodified": True}
        if with_zoom:
            crop = image.crop((x0, y0, x1, y1))
            detail = image_ax(x+1.05, y, crop, color=GOLD, thickness=.8)
            for point, destination in [((x1-.5, y0-.5), (0, 1)),
                                        ((x1-.5, y1-.5), (0, 0))]:
                fig.add_artist(ConnectionPatch(
                    point, destination, "data", "axes fraction", axesA=ax, axesB=detail,
                    color=GOLD, linewidth=.65, clip_on=False,
                    path_effects=halo, zorder=4))
            record["crop_pixel_sha256"] = hashlib.sha256(crop.tobytes()).hexdigest()
        audit.append(record)

    text(.20, 3.84, "(e) Prediction frame +32", ha="left", va="center",
         fontsize=10.3, weight="bold", color=INK)
    fig.add_artist(Rectangle((.13/W, .035/H), 1.14/W, 3.62/H,
                             transform=fig.transFigure, facecolor="#F6F8FB",
                             edgecolor="none", zorder=-1))
    fig.add_artist(Line2D([1.35/W, 1.35/W], [.035/H, 3.64/H],
                          transform=fig.transFigure, color="#BEC9D5", linewidth=.75))
    text(.70, 3.66, "Ground truth", ha="center", va="center",
         fontsize=8.3, weight="bold", color=INK)
    for key,label,y,color in [('gt','Ground truth',2.76,INK),('initial_sft','Initial SFT (50k)',.96,BLUE)]:
        if key != "gt": text(.70,y+.82,label,ha="center",va="center",fontsize=7.7,weight="bold",color=color)
        draw_frame(key,.20,y,False)
        image,entry=load(key)
        image_ax(.20,y-.87,image.crop((x0,y0,x1,y1)),color=GOLD,thickness=.8)
    methods=[('sft','Continued SFT',BLUE,2.76),('mcts','MCTS','#BB7149',1.86),
             ('ordinary','Ordinary beam','#268879',.96),('hybrid','RoboImagine','#80639E',.06)]
    for step,left in zip([100,250,500],[1.48,3.63,5.78]):
        text(left+1.025,3.66,f"{step} updates",ha="center",va="center",fontsize=9.1,weight="bold",color=INK)
        for key,label,color,y in methods: draw_frame(f"{key}_{step}",left,y,True)
    for key,label,color,y in methods:
        text(7.91,y+.40,label,ha="center",va="center",rotation=90,fontsize=7.4,weight="bold",color=color,linespacing=.88)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in fig.texts:
        bounds = item.get_window_extent(renderer)
        assert bounds.x0 >= 0 and bounds.y0 >= 0 and bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height, (item.get_text(), bounds)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for suffix, fonttype in [(".editable.svg", "none"), (".svg", "path")]:
        with plt.rc_context({"svg.fonttype": fonttype}):
            fig.savefig(args.out_dir/("figure"+suffix), format="svg")
    fig.savefig(args.out_dir/"figure.pdf", format="pdf")
    fig.savefig(args.out_dir/"figure.png", dpi=400)
    plt.close(fig)

    caption = (
        "Post-training dynamics from a shared 50k-step SFT initialization. "
        "(a–d) Metrics are averaged over 32 predicted frames on the same 128 RT-1 "
        "trajectories; markers denote evaluated checkpoints at 0, 100, 250, and 500 "
        "additional training updates. Curves compare continued SFT, MCTS, ordinary beam, and RoboImagine (GDBS); search variants use combined feedback. (e) A selected drawer-closing trajectory "
        "compares continued SFT, MCTS, ordinary beam, and RoboImagine (GDBS) at the three training checkpoints. "
        "Every image in (e) shows the fixed prediction frame +32, rather than a "
        "different time within a rollout. Ground truth and the initial SFT prediction "
        "are shared references. Gold boxes and enlarged details use identical crop "
        "coordinates across all checkpoints. The single example illustrates visual "
        "differences and is distinct from the aggregate curves.\n")
    (args.out_dir/"caption.txt").write_text(caption)
    (args.out_dir/"source_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    metadata = {"figure_inches": [W, H], "curve_region_inches": [W, CURVE_H],
                "curve_text_scale": 0.92, "gain_text_color": "#BF3F45",
                "lower_case_panel_change": "Four matched model rows at100/250/500; all images/ROIs unchanged; shared GT/SFT references with detail crops",
                "curve_rendering": "Native vector Matplotlib artists; no curve raster embedded",
                "curve_data": str(args.data.resolve()), "curve_data_sha256": sha(args.data),
                "curve_helper_sha256": sha(HERE/"curve_panel.py"),
                "series_rendering": series_rendering, "gain_annotations": gains,
                "source_manifest": str(manifest_path.resolve()),
                "source_manifest_sha256": sha(manifest_path),
                "dataset": "RT-1", "case_index": args.case, "case_source": case,
                "selection": manifest["selection"],
                "selected_example": True, "prediction_frame_offset": 32,
                "roi_xywh": roi_xywh, "frame_color_editing": "None",
                "font_paths": fonts, "image_audit": audit}
    (args.out_dir/"render_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(f"Rendered {W} x {H} in; native vector curves and {len(audit)} verified source frames.")


if __name__ == "__main__":
    main()
