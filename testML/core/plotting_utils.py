import base64
import io
import os
import subprocess
import tempfile
import textwrap
import uuid

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import scienceplots  # noqa: F401
    HAS_SCIENCEPLOTS = True
except ImportError:
    HAS_SCIENCEPLOTS = False


NATURE_COLORS = {
    "blue": "#1f77b4",
    "teal": "#1b9e77",
    "orange": "#d95f02",
    "red": "#c44e52",
    "gold": "#b8860b",
    "purple": "#7b6fd0",
    "slate": "#4c566a",
    "gray": "#b0b7c3",
    "mint": "#66c2a5",
}


def apply_nature_style():
    if HAS_SCIENCEPLOTS:
        plt.style.use(["science", "nature", "no-latex"])
    sns.set_theme(style="whitegrid", palette="deep")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "SimSun"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
        }
    )


def _powerpoint_export(svg_bytes, output_ext, export_code):
    temp_dir = tempfile.mkdtemp(prefix="natureml_emf_")
    svg_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}.svg")
    out_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}.{output_ext}")
    try:
        with open(svg_path, "wb") as handle:
            handle.write(svg_bytes)

        escaped_svg = svg_path.replace("'", "''")
        escaped_out = out_path.replace("'", "''")
        ps_script = textwrap.dedent(
            f"""
            $ErrorActionPreference = 'Stop'
            $ppt = $null
            $pres = $null
            $slide = $null
            $shape = $null
            try {{
                $ppt = New-Object -ComObject PowerPoint.Application
                $ppt.Visible = -1
                $pres = $ppt.Presentations.Add()
                $slide = $pres.Slides.Add(1, 12)
                $shape = $slide.Shapes.AddPicture('{escaped_svg}', $false, $true, 10, 10, -1, -1)
                $shape.Export('{escaped_out}', {export_code})
                $pres.Close()
                $ppt.Quit()
            }} finally {{
                if ($shape -ne $null) {{ [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($shape) }}
                if ($slide -ne $null) {{ [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($slide) }}
                if ($pres -ne $null) {{ [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($pres) }}
                if ($ppt -ne $null) {{ [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) }}
                [GC]::Collect()
                [GC]::WaitForPendingFinalizers()
            }}
            """
        ).strip()
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        if completed.returncode != 0 or not os.path.exists(out_path):
            detail = completed.stderr.strip() or completed.stdout.strip() or "Unknown PowerPoint COM error."
            raise RuntimeError(f"{output_ext.upper()} conversion failed: {detail}")
        with open(out_path, "rb") as handle:
            return handle.read()
    finally:
        for path in (svg_path, out_path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass


def _convert_svg_bytes_to_emf_bytes(svg_bytes):
    return _powerpoint_export(svg_bytes, "emf", 5)


def convert_svg_bytes_to_png_bytes(svg_bytes):
    return _powerpoint_export(svg_bytes, "png", 2)


def fig_to_base64(fig, fmt="png", dpi=300):
    buf = io.BytesIO()
    save_format = "svg" if fmt == "emf" else fmt
    fig.savefig(buf, format=save_format, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    payload = buf.read()
    if fmt == "emf":
        payload = _convert_svg_bytes_to_emf_bytes(payload)
    img_str = base64.b64encode(payload).decode("utf-8")
    if fmt == "svg":
        mime_type = "image/svg+xml"
    elif fmt == "emf":
        mime_type = "image/emf"
    elif fmt == "pdf":
        mime_type = "application/pdf"
    else:
        mime_type = f"image/{fmt}"
    return f"data:{mime_type};base64,{img_str}"
