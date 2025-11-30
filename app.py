from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any, cast, Sequence

# ====================================================================
# 0. Auto-Dependency Installation (Deployment Helper)
# ====================================================================

AUTO_DEPENDENCY_MAP = {
    "flask": "Flask>=3.0.0",
    "dash": "dash>=2.17.0,<3.0.0",
    "dash_bootstrap_components": "dash-bootstrap-components>=1.6.0",
    "pandas": "pandas>=1.5.0",
    "plotly": "plotly>=5.20.0",
    "numpy": "numpy>=1.22.0",
}

_AUTO_DEPENDENCY_SENTINEL = Path(__file__).with_name(".auto_dependencies_installed")


def _ensure_user_site_on_path() -> None:
    try:
        import site
    except Exception:  # pragma: no cover - site module should exist but guard just in case
        return

    candidates = set()
    try:
        user_site = site.getusersitepackages()
    except Exception:
        user_site = ""
    if user_site:
        candidates.add(user_site)

    try:
        user_base = site.getuserbase()
    except Exception:
        user_base = ""

    if user_base:
        py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        candidates.add(os.path.join(user_base, "lib", py_ver, "site-packages"))

    for path in candidates:
        if path and os.path.isdir(path) and path not in sys.path:
            try:
                site.addsitedir(path)
            except Exception:
                if path not in sys.path:
                    sys.path.insert(0, path)


_ensure_user_site_on_path()


def _should_use_user_site() -> bool:
    return (
        not os.environ.get("VIRTUAL_ENV")
        and getattr(sys, "base_prefix", sys.prefix) == sys.prefix
    )


def _run_magic_pip(packages: List[str]) -> None:
    if not packages:
        return

    try:
        from IPython import get_ipython  # type: ignore
    except Exception:  # pragma: no cover - IPython not available
        get_ipython = None  # type: ignore[assignment]

    ip = get_ipython() if callable(get_ipython) else None  # type: ignore[misc]

    pip_args: List[str] = ["install", "--quiet"]

    if _should_use_user_site():
        pip_args.append("--user")

    pip_args.extend(packages)

    os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")

    command_display = "!" + " ".join(["pip", *pip_args])

    ip_error: Optional[BaseException] = None
    if ip is not None:
        try:
            ip.run_cell(command_display)
            return
        except BaseException as error:  # pragma: no cover - IPython failure path
            ip_error = error

    try:
        from pip._internal.cli.main import main as pip_main  # type: ignore
    except Exception as exc:  # pragma: no cover - pip internal unavailable
        hint = f"Automatic dependency installation failed. Please run manually: {command_display[1:]}"
        if ip_error:
            hint += f"\nOriginal IPython error: {ip_error}"
        raise RuntimeError(hint) from exc

    status = pip_main(pip_args)
    if status != 0:
        hint = f"Automatic dependency installation failed. Please run manually: {command_display[1:]}"
        raise RuntimeError(hint)

    _ensure_user_site_on_path()


def _auto_install_dependencies() -> None:
    if _AUTO_DEPENDENCY_SENTINEL.exists():
        packages: List[str] = []
        for module_name, requirement in AUTO_DEPENDENCY_MAP.items():
            try:
                importlib.import_module(module_name)
            except ImportError:
                packages.append(requirement)
        if not packages:
            return
    else:
        packages = list(AUTO_DEPENDENCY_MAP.values())

    _run_magic_pip(packages)

    missing_after_install: List[str] = []
    for module_name in AUTO_DEPENDENCY_MAP:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing_after_install.append(module_name)

    if missing_after_install:
        cmd = "pip install --quiet" + (" --user" if _should_use_user_site() else "")
        requirements = " ".join(AUTO_DEPENDENCY_MAP[name] for name in missing_after_install)
        raise RuntimeError(
            "Missing modules after attempting automatic installation: {}. Please run: {} {}".format(
                ", ".join(missing_after_install), cmd, requirements
            )
        )

    try:
        _AUTO_DEPENDENCY_SENTINEL.write_text(
            "auto dependencies installed; delete this file to force reinstall\n",
            encoding="utf-8",
        )
    except OSError:
        # If the file system is read-only (some hosting environments), ignore the sentinel write.
        pass


_auto_install_dependencies()

# ====================================================================
# 1. Library Import and Environment Setup
# ====================================================================

import dash
from dash import html, dcc, Input, Output, State, ctx, dash_table, ALL, MATCH
import dash_bootstrap_components as dbc
import plotly
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import base64
import io
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque, defaultdict
import copy
import uuid
from dash.exceptions import PreventUpdate
import contextlib

# Set random seed for reproducibility
np.random.seed(42)

# Initialize the Dash app
# suppress_callback_exceptions=True is required for dynamic components (like the Inspector)
# Use a unique name to avoid "setup method called after start" errors on re-run
app_name = f"Interactive_Editor_{uuid.uuid4().hex[:8]}"
app = dash.Dash(app_name, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Expose application for WSGI
application = app.server

def clean_figure_dict(fig_dict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Ensure figure dictionary is valid for go.Figure."""
    if fig_dict is None:
        return {"data": [], "layout": {}}
    return fig_dict

print("✅ Libraries imported successfully.")
print("📦 Dash version:", dash.__version__)
print("📊 Plotly version:", plotly.__version__)
print(f"🚀 App initialized as: {app_name}")

# ====================================================================
# 2. Core Data Model
# ====================================================================

@dataclass
class TraceDataset:
    """Container for a single logical plot layer."""
    key: str
    name: str
    df: pd.DataFrame
    color: str = "#1f77b4"
    line_width: float = 2.5
    marker_size: float = 6.0
    visible: bool = True
    chart_type: str = "scatter"

    def to_plotly_trace(self):
        """Create a Plotly trace from the dataset."""
        # Safety check for columns
        # Use explicit column access to satisfy Pylance
        x = self.df['x'] if 'x' in self.df.columns else None
        y = self.df['y'] if 'y' in self.df.columns else None
        
        if x is None or y is None:
            # Fallback if x/y columns don't exist (e.g. custom data)
            # Try to use first two columns
            if self.df.shape[1] >= 2:
                x = self.df.iloc[:, 0]
                y = self.df.iloc[:, 1]
            else:
                return go.Scatter(name=f"{self.name} (Empty)")

        if self.chart_type == "bar":
            trace = go.Bar(x=x, y=y, name=self.name)
        elif self.chart_type == "pie":
            trace = go.Pie(labels=x, values=y, name=self.name)
        elif self.chart_type == "histogram":
            trace = go.Histogram(x=y, name=self.name)
        elif self.chart_type == "box":
            trace = go.Box(y=y, name=self.name)
        else:  # default scatter
            trace = go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=self.name,
            )

        # Apply common styling using update() to avoid Pylance attribute errors
        updates = {}
        if hasattr(trace, "marker"):
            updates["marker"] = dict(size=self.marker_size, color=self.color)
        if hasattr(trace, "line"):
            updates["line"] = dict(width=self.line_width, color=self.color)
        
        if updates:
            trace.update(**updates)

        # Visibility: True -> normal, False -> legend only
        trace.visible = True if self.visible else "legendonly"
        return trace

# ====================================================================
# 3. Core State Management (Figure Store)
# ====================================================================

class FigureStore:
    """Owns the current Plotly figure and its logical datasets."""

    def __init__(self, theme: str = "plotly_white") -> None:
        self.current_theme: str = theme
        self.figure: Optional[go.Figure] = None
        self.datasets: Dict[str, TraceDataset] = {}
        self.dataset_order: List[str] = []
        # Global Data Repository: Stores raw DataFrames imported by the user
        self.data_repository: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": None,
            "version": "1.0.0",
        }

        # Initialise with a reasonable default figure
        self._init_default_figure()

    def _init_default_figure(self) -> None:
        """Create a simple damped sine wave as the initial demo figure."""
        t = np.linspace(0, 10, 200)
        signal = np.sin(t) * np.exp(-0.15 * t)
        # Add Z column for 3D compatibility
        z_val = np.cos(t) * t 
        df = pd.DataFrame({"x": t, "y": signal, "z": z_val})
        
        # Add to repository as well
        self.add_dataframe("demo_signal", df)

        self.add_dataset(
            key="trace_1",
            name="Demo Signal",
            df=df,
            color="#1f77b4",
        )
        self.rebuild_figure_from_datasets()

    def add_dataframe(self, name: str, df: pd.DataFrame) -> None:
        """Register a raw DataFrame into the global repository."""
        if name is None:
            return
        # Sanitize name to be a valid identifier
        safe_name = name.replace(" ", "_").replace("-", "_").replace(".", "_")
        self.data_repository[safe_name] = df
        self._touch()

    def get_dataframe(self, name: str) -> Optional[pd.DataFrame]:
        return self.data_repository.get(name)

    def _touch(self) -> None:
        self.metadata["updated_at"] = datetime.now().isoformat(timespec="seconds")

    def _base_layout(self) -> Dict[str, Any]:
        """Base layout analogous to MATLAB's default figure appearance."""
        layout: Dict[str, Any] = {
            "template": self.current_theme,
            "margin": dict(l=60, r=20, t=50, b=60),
            "hovermode": "closest",
            "legend": dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1.0,
            ),
        }
        return layout

    def update_theme(self, theme: str) -> None:
        self.current_theme = theme or self.current_theme
        if self.figure is not None:
            self.figure.update_layout(template=self.current_theme)
            self._touch()

    def get_figure_dict(self) -> Optional[Dict[str, Any]]:
        return self.figure.to_dict() if self.figure is not None else None

    def update_figure(self, fig: go.Figure) -> None:
        """Replace the current figure (e.g. after interactive edits)."""
        self.figure = fig
        self._touch()

    def add_dataset(
        self,
        key: str,
        name: str,
        df: pd.DataFrame,
        color: str = "#1f77b4",
        line_width: float = 2.5,
        marker_size: float = 6.0,
        visible: bool = True,
        chart_type: str = "scatter",
    ) -> None:
        df = df.reset_index(drop=True)
        dataset = TraceDataset(
            key=key,
            name=name,
            df=df,
            color=color,
            line_width=line_width,
            marker_size=marker_size,
            visible=visible,
            chart_type=chart_type,
        )
        self.datasets[key] = dataset
        if key not in self.dataset_order:
            self.dataset_order.append(key)

    def rebuild_figure_from_datasets(self) -> None:
        """Rebuild the Plotly figure from ``datasets`` & ``dataset_order``."""
        fig = go.Figure()
        for key in self.dataset_order:
            dataset = self.datasets.get(key)
            if not dataset:
                continue
            fig.add_trace(dataset.to_plotly_trace())

        fig.update_layout(**self._base_layout())
        # Preserve existing titles if any
        if self.figure is not None:
            # Use getattr to avoid Pylance strict checks if it's confused
            layout = getattr(self.figure, 'layout', None)
            if layout:
                existing_title = getattr(layout, 'title', None)
                if existing_title and getattr(existing_title, 'text', None):
                    fig.update_layout(title=existing_title)
        self.update_figure(fig)

    def set_trace_visibility(self, visible_keys: List[str]) -> None:
        visible_set = set(visible_keys or [])
        for key, dataset in self.datasets.items():
            dataset.visible = key in visible_set
        self._touch()

    def remove_trace(self, index: int) -> None:
        """Remove a trace by its index in the figure."""
        if self.figure is None:
            return

        # Cast to list to modify, then reassign. 
        # Pylance might complain about tuple vs list, but Plotly accepts list.
        data = list(self.figure.data)
        if 0 <= index < len(data):
            data.pop(index)
            self.figure.data = tuple(data) # Convert back to tuple to satisfy type hints
            self._touch()

        if index < len(self.dataset_order):
            key = self.dataset_order[index]
            if key in self.datasets:
                del self.datasets[key]
            self.dataset_order.pop(index)

    def remove_annotation(self, index: int) -> None:
        """Remove an annotation by its index."""
        if self.figure is None:
            return
            
        # Use cast(Any, ...) to bypass Pylance confusion about 'tuple' vs 'Layout'
        layout = cast(Any, self.figure.layout)
        if not layout or not hasattr(layout, 'annotations'):
            return
        
        annots = list(layout.annotations) if layout.annotations else []
        if 0 <= index < len(annots):
            annots.pop(index)
            layout.annotations = tuple(annots)
            self._touch()

    def remove_shape(self, index: int) -> None:
        """Remove a shape by its index."""
        if self.figure is None:
            return
            
        # Use cast(Any, ...) to bypass Pylance confusion about 'tuple' vs 'Layout'
        layout = cast(Any, self.figure.layout)
        if not layout or not hasattr(layout, 'shapes'):
            return
        
        shapes = list(layout.shapes) if layout.shapes else []
        if 0 <= index < len(shapes):
            shapes.pop(index)
            layout.shapes = tuple(shapes)
            self._touch()

    def remove_image(self, index: int) -> None:
        """Remove an image by its index."""
        if self.figure is None:
            return
            
        # Use cast(Any, ...) to bypass Pylance confusion about 'tuple' vs 'Layout'
        layout = cast(Any, self.figure.layout)
        if not layout or not hasattr(layout, 'images'):
            return
        
        images = list(layout.images) if layout.images else []
        if 0 <= index < len(images):
            images.pop(index)
            layout.images = tuple(images)
            self._touch()

    def remove_points(self, selected_points: List[Dict[str, Any]]) -> bool:
        """Remove the selected points from the underlying datasets."""
        if not selected_points or not self.datasets:
            return False

        by_curve: Dict[int, set[int]] = defaultdict(set)
        for p in selected_points:
            c = p.get("curveNumber")
            idx = p.get("pointIndex")
            if c is None or idx is None:
                continue
            by_curve[int(c)].add(int(idx))

        if not by_curve:
            return False

        changed = False
        for curve_idx, indices in by_curve.items():
            if curve_idx < 0 or curve_idx >= len(self.dataset_order):
                continue
            key = self.dataset_order[curve_idx]
            dataset = self.datasets.get(key)
            if dataset is None or dataset.df.empty:
                continue

            df = dataset.df.reset_index(drop=True)
            mask = ~df.index.isin(indices)
            if mask.all():
                continue

            dataset.df = df.loc[mask].reset_index(drop=True)
            changed = True

        if changed:
            self.rebuild_figure_from_datasets()
        return changed

    def serialize_session(self) -> Dict[str, Any]:
        """Return a JSON‑serialisable snapshot of the current session."""
        datasets_payload: Dict[str, Any] = {}
        for key, ds in self.datasets.items():
            datasets_payload[key] = {
                "name": ds.name,
                "color": ds.color,
                "line_width": ds.line_width,
                "marker_size": ds.marker_size,
                "visible": ds.visible,
                "chart_type": ds.chart_type,
                "df": ds.df.to_dict(orient="list"),
            }

        return {
            "metadata": copy.deepcopy(self.metadata),
            "current_theme": self.current_theme,
            "datasets": datasets_payload,
            "dataset_order": list(self.dataset_order),
            "figure": self.figure.to_dict() if self.figure is not None else None,
            "version": "1.0.0",
        }

    def load_session(self, payload: Dict[str, Any]) -> None:
        """Load a session previously returned by :meth:`serialize_session`."""
        if not payload:
            return

        self.current_theme = payload.get("current_theme", self.current_theme)
        self.metadata.update(payload.get("metadata", {}))

        self.datasets.clear()
        self.dataset_order = list(payload.get("dataset_order", []))

        for key, item in payload.get("datasets", {}).items():
            df_dict = item.get("df", {})
            df = pd.DataFrame(df_dict).reset_index(drop=True)
            self.add_dataset(
                key=key,
                name=item.get("name", key),
                df=df,
                color=item.get("color", "#1f77b4"),
                line_width=float(item.get("line_width", 2.5)),
                marker_size=float(item.get("marker_size", 6.0)),
                visible=bool(item.get("visible", True)),
                chart_type=item.get("chart_type", "scatter"),
            )

        fig_dict = payload.get("figure")
        if fig_dict is not None:
            self.figure = go.Figure(fig_dict)
            self.figure.update_layout(template=self.current_theme)
            self._touch()
        else:
            self.rebuild_figure_from_datasets()

# ====================================================================
# 4. History and Logs
# ====================================================================

class HistoryStack:
    """Classic undo/redo stack for figure dictionaries."""

    def __init__(self, max_size: int = 50) -> None:
        self.max_size = max_size
        self.undo_stack: List[Dict[str, Any]] = []
        self.redo_stack: List[Dict[str, Any]] = []

    def push(self, fig_dict: Optional[Dict[str, Any]]) -> None:
        if fig_dict is None:
            return
        snapshot = copy.deepcopy(fig_dict)
        
        # Avoid duplicate states (simple check)
        if self.undo_stack:
            try:
                # Use json dump to compare structure, ignoring memory addresses
                last_state = json.dumps(self.undo_stack[-1], sort_keys=True, default=str)
                new_state = json.dumps(snapshot, sort_keys=True, default=str)
                if last_state == new_state:
                    return
            except Exception:
                pass
                
        self.undo_stack.append(snapshot)
        if len(self.undo_stack) > self.max_size:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def can_undo(self) -> bool:
        return len(self.undo_stack) > 1

    def can_redo(self) -> bool:
        return bool(self.redo_stack)

    def undo(self) -> Optional[Dict[str, Any]]:
        if not self.can_undo():
            return None
        current = self.undo_stack.pop()
        self.redo_stack.append(current)
        return copy.deepcopy(self.undo_stack[-1])

    def redo(self) -> Optional[Dict[str, Any]]:
        if not self.redo_stack:
            return None
        state = self.redo_stack.pop()
        self.undo_stack.append(copy.deepcopy(state))
        return copy.deepcopy(state)


class ActionLog:
    """Append‑only log of high‑level user actions."""

    def __init__(self, max_actions: int = 500) -> None:
        self.max_actions = max_actions
        self.actions: List[Dict[str, Any]] = []

    def record(self, action_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        entry = {
            "type": action_type,
            "payload": copy.deepcopy(payload or {}),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self.actions.append(entry)
        if len(self.actions) > self.max_actions:
            self.actions.pop(0)

# ====================================================================
# 5. Code Generator
# ====================================================================

class CodeGenerator:
    """Turn the current figure into runnable Python code."""

    def generate_code(self, store: FigureStore) -> str:
        if store.figure is None:
            return "# No figure available yet. Interact with the editor first."

        fig_json = store.figure.to_json()

        lines: List[str] = []
        lines.append("# Auto‑generated by Python Interactive Figure Editor")
        lines.append("# Recreate the current figure exactly as seen in the UI.")
        lines.append("import json")
        lines.append("import plotly.graph_objects as go  # import plotly")
        lines.append("")
        lines.append(f"fig_dict = json.loads({fig_json!r})")
        lines.append("fig = go.Figure(fig_dict)")
        lines.append("")
        lines.append("# Show the figure in an interactive window")
        lines.append("fig.show()")
        lines.append("")
        lines.append("# Tip: you can now modify `fig` programmatically, e.g.:")
        lines.append("# fig.update_layout(title='My Edited Figure')")
        return "\n".join(lines)

    def generate_smart_plot_code(self, df_name: str, plot_type: str, df: pd.DataFrame) -> str:
        """Generate Plotly code with smart column selection."""
        
        # --- Smart Column Selection Logic ---
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        all_cols = df.columns.tolist()
        
        if not all_cols:
            return "# Error: Dataset is empty."

        # Helper to safely get index
        def safe_get(lst, idx): return lst[idx] if len(lst) > idx else (all_cols[0] if all_cols else None)
        def safe_get_num(idx): return num_cols[idx] if len(num_cols) > idx else (num_cols[0] if num_cols else (all_cols[0] if all_cols else None))
        
        cols = {}
        
        if plot_type in ['scatter', 'line', 'area', 'bubble', 'scatter3d', 'line3d', 'ternary']:
            cols['x'] = safe_get_num(0)
            cols['y'] = safe_get_num(1)
            cols['z'] = safe_get_num(2)
            cols['size'] = safe_get_num(3)
            cols['color'] = safe_get(cat_cols, 0) # Color by category if possible
            
        elif plot_type in ['bar', 'pie', 'sunburst', 'treemap', 'funnel']:
            cols['x'] = safe_get(cat_cols, 0) or safe_get(num_cols, 0)
            cols['y'] = safe_get_num(0) if cols['x'] != safe_get_num(0) else safe_get_num(1)
            cols['names'] = cols['x']
            cols['values'] = cols['y']
            
        elif plot_type in ['hist', 'box', 'violin', 'heatmap']:
            cols['x'] = safe_get_num(0)
            if plot_type == 'hist' and cols['x'] is None: cols['x'] = safe_get(all_cols, 0)
            cols['y'] = safe_get_num(1)
            cols['color'] = safe_get(cat_cols, 0)
            
        elif plot_type == 'surface':
            cols['x'], cols['y'], cols['z'] = safe_get_num(0), safe_get_num(1), safe_get_num(2)
            
        elif plot_type == 'contour':
            cols['x'], cols['y'] = safe_get_num(0), safe_get_num(1)
            
        elif plot_type == 'polar':
            cols['x'], cols['y'] = safe_get_num(0), safe_get_num(1) # r, theta
            
        elif plot_type == 'waterfall':
             cols['x'] = safe_get(cat_cols, 0) or safe_get(all_cols, 0)
             cols['y'] = safe_get_num(0)
             
        elif plot_type == 'scatmat':
             cols['dimensions'] = num_cols[:5] if len(num_cols) >= 2 else all_cols[:5]
             cols['color'] = safe_get(cat_cols, 0) or safe_get(num_cols, 0)

        elif plot_type == 'parcoords':
             cols['dimensions'] = num_cols[:5] if len(num_cols) >= 2 else all_cols[:5]
             cols['color'] = safe_get(num_cols, 0)
            
        elif plot_type == 'candle':
            lower_cols = [c.lower() for c in all_cols]
            try: cols['x'] = all_cols[lower_cols.index('time')] 
            except: cols['x'] = safe_get(all_cols, 0)
            try: cols['open'] = all_cols[lower_cols.index('open')]
            except: cols['open'] = safe_get_num(1)
            try: cols['high'] = all_cols[lower_cols.index('high')]
            except: cols['high'] = safe_get_num(2)
            try: cols['low'] = all_cols[lower_cols.index('low')]
            except: cols['low'] = safe_get_num(3)
            try: cols['close'] = all_cols[lower_cols.index('close')]
            except: cols['close'] = safe_get_num(4)

        elif plot_type in ['scatgeo', 'globe']:
            lower_cols = [c.lower() for c in all_cols]
            try: cols['lat'] = all_cols[lower_cols.index('lat')]
            except: cols['lat'] = safe_get_num(0)
            try: cols['lon'] = all_cols[lower_cols.index('lon')]
            except: cols['lon'] = safe_get_num(1)
            cols['color'] = safe_get(cat_cols, 0)

        elif plot_type == 'choropleth':
            lower_cols = [c.lower() for c in all_cols]
            try: cols['locations'] = all_cols[lower_cols.index('iso_alpha')]
            except: 
                try: cols['locations'] = all_cols[lower_cols.index('country')]
                except: cols['locations'] = safe_get(cat_cols, 0)
            cols['color'] = safe_get_num(0)

        # --- Code Generation ---
        cmd = f"# Generate {plot_type} plot from {df_name}\n"
        cmd += "import plotly.express as px\n"
        cmd += "import plotly.graph_objects as go\n\n"
        
        # Helper to format None
        def fmt_col(key): return f"'{cols.get(key)}'" if cols.get(key) else "None"
        def fmt_col_check(key): return f"'{cols.get(key)}' if '{cols.get(key)}' != 'None' else None"

        if plot_type == 'scatter':
            cmd += f"fig = px.scatter({df_name}, x={fmt_col('x')}, y={fmt_col('y')}, color={fmt_col_check('color')})"
        elif plot_type == 'line':
            cmd += f"fig = px.line({df_name}, x={fmt_col('x')}, y={fmt_col('y')}, color={fmt_col_check('color')})"
        elif plot_type == 'bar':
            cmd += f"fig = px.bar({df_name}, x={fmt_col('x')}, y={fmt_col('y')}, color={fmt_col('x')})"
        elif plot_type == 'area':
            cmd += f"fig = px.area({df_name}, x={fmt_col('x')}, y={fmt_col('y')})"
        elif plot_type == 'bubble':
            cmd += f"fig = px.scatter({df_name}, x={fmt_col('x')}, y={fmt_col('y')}, size={df_name}[{fmt_col('size')}].abs(), color={fmt_col_check('color')})"
        elif plot_type == 'pie':
            cmd += f"fig = px.pie({df_name}, names={fmt_col('names')}, values={fmt_col('values')})"
        elif plot_type == 'sunburst':
            cmd += f"fig = px.sunburst({df_name}, path=[{fmt_col('names')}], values={fmt_col('values')})"
        elif plot_type == 'treemap':
            cmd += f"fig = px.treemap({df_name}, path=[{fmt_col('names')}], values={fmt_col('values')})"
        elif plot_type == 'funnel':
            cmd += f"fig = px.funnel({df_name}, x={fmt_col('values')}, y={fmt_col('names')})"
        elif plot_type == 'hist':
            cmd += f"fig = px.histogram({df_name}, x={fmt_col('x')}, color={fmt_col_check('color')})"
        elif plot_type == 'heatmap':
            cmd += f"fig = px.density_heatmap({df_name}, x={fmt_col('x')}, y={fmt_col('y')})"
        elif plot_type == 'box':
            cmd += f"fig = px.box({df_name}, x={fmt_col_check('color')}, y={fmt_col('y')})"
        elif plot_type == 'violin':
            cmd += f"fig = px.violin({df_name}, x={fmt_col_check('color')}, y={fmt_col('y')})"
        elif plot_type == 'scatter3d':
            cmd += f"fig = px.scatter_3d({df_name}, x={fmt_col('x')}, y={fmt_col('y')}, z={fmt_col('z')}, color={fmt_col_check('color')})"
        elif plot_type == 'line3d':
            cmd += f"fig = px.line_3d({df_name}, x={fmt_col('x')}, y={fmt_col('y')}, z={fmt_col('z')})"
        elif plot_type == 'surface':
            cmd += f"fig = go.Figure(data=[go.Mesh3d(x={df_name}[{fmt_col('x')}], y={df_name}[{fmt_col('y')}], z={df_name}[{fmt_col('z')}], opacity=0.8)])"
        elif plot_type == 'contour':
            cmd += f"fig = px.density_contour({df_name}, x={fmt_col('x')}, y={fmt_col('y')})"
        elif plot_type == 'polar':
            cmd += f"fig = px.scatter_polar({df_name}, r={fmt_col('x')}, theta={fmt_col('y')})"
        elif plot_type == 'ternary':
            cmd += f"fig = px.scatter_ternary({df_name}, a={fmt_col('x')}, b={fmt_col('y')}, c={fmt_col('z')})"
        elif plot_type == 'waterfall':
            cmd += f"fig = go.Figure(go.Waterfall(name='Waterfall', orientation='v', measure=['relative']*len({df_name}), x={df_name}[{fmt_col('x')}], y={df_name}[{fmt_col('y')}], connector={{'mode':'between', 'line':{{'width':4, 'color':'rgb(0, 0, 0)', 'dash':'solid'}}}}))"
        elif plot_type == 'scatmat':
            cmd += f"fig = px.scatter_matrix({df_name}, dimensions={cols.get('dimensions')}, color={fmt_col_check('color')})"
        elif plot_type == 'parcoords':
            cmd += f"fig = px.parallel_coordinates({df_name}, dimensions={cols.get('dimensions')}, color={fmt_col_check('color')})"
        elif plot_type == 'scatgeo':
            cmd += f"fig = px.scatter_geo({df_name}, lat={fmt_col('lat')}, lon={fmt_col('lon')}, color={fmt_col_check('color')})"
        elif plot_type == 'globe':
            cmd += f"fig = px.scatter_geo({df_name}, lat={fmt_col('lat')}, lon={fmt_col('lon')}, color={fmt_col_check('color')}, projection='orthographic')"
        elif plot_type == 'choropleth':
            cmd += f"fig = px.choropleth({df_name}, locations={fmt_col('locations')}, color={fmt_col('color')}, locationmode='ISO-3')"
        elif plot_type == 'candle':
            cmd += f"fig = go.Figure(data=[go.Candlestick(x={df_name}[{fmt_col('x')}], open={df_name}[{fmt_col('open')}], high={df_name}[{fmt_col('high')}], low={df_name}[{fmt_col('low')}], close={df_name}[{fmt_col('close')}])])"
        
        cmd += "\n\n# Update layout for better view\n"
        cmd += "fig.update_layout(template='plotly_white', title='Generated Plot')\n"
        return cmd

# ====================================================================
# 6. Initialization Singleton
# ====================================================================

figure_store = FigureStore()
history_stack = HistoryStack(max_size=50)
action_log = ActionLog(max_actions=1000)
code_generator = CodeGenerator()

def create_initial_figure() -> go.Figure:
    if figure_store.figure is None:
        figure_store._init_default_figure()
    return cast(go.Figure, figure_store.figure)

print("⚙️  Checking core engine status...")
print(f"   - Active theme: {figure_store.current_theme}")
print(f"   - Dataset count: {len(figure_store.datasets)}")
print(f"   - Data Repository: {list(figure_store.data_repository.keys())}")

# Push current figure to stack, ensuring initial state is available for undo/redo
if figure_store.figure:
    history_stack.push(figure_store.get_figure_dict())

print("✅ Core engine ready.")

# ====================================================================
# 7. UI Components: Top Ribbon
# ====================================================================

ribbon = dbc.Card([
    dbc.CardHeader(
        dbc.Tabs([
            dbc.Tab(label="HOME", tab_id="tab-home", label_style={"fontWeight": "bold", "fontSize": "13px"}),
            dbc.Tab(label="DATA", tab_id="tab-data", label_style={"fontWeight": "bold", "fontSize": "13px"}),
            dbc.Tab(label="PLOTS", tab_id="tab-plots", label_style={"fontWeight": "bold", "fontSize": "13px"}),
            dbc.Tab(label="ANNOTATE", tab_id="tab-annotate", label_style={"fontWeight": "bold", "fontSize": "13px"}),
            dbc.Tab(label="VIEW", tab_id="tab-view", label_style={"fontWeight": "bold", "fontSize": "13px"}),
        ], id="ribbon-tabs", active_tab="tab-home", className="nav-tabs-bottom"),
        className="pb-0 border-bottom-0 bg-light pt-1"
    ),
    dbc.CardBody([
        # --- HOME TAB ---
        html.Div(id="ribbon-content-home", children=[
            dbc.Row([
                dbc.Col([
                    html.Div("File", className="text-muted small fw-bold mb-1 text-center", style={"fontSize": "10px"}),
                    dbc.ButtonGroup([
                        dcc.Upload(
                            dbc.Button([html.Div("📂", className="h4 mb-0"), "Open Session"], color="light", size="sm", className="d-flex flex-column align-items-center px-3 border-0"),
                            id="upload-session", multiple=False, style={"display": "inline-block"}
                        ),
                        dbc.Button([html.Div("💾", className="h4 mb-0"), "Save Session"], id="btn-save-session", color="light", size="sm", className="d-flex flex-column align-items-center px-3 border-0"),
                    ], className="me-2")
                ], width="auto", className="border-end pe-2"),
                dbc.Col([
                    html.Div("History", className="text-muted small fw-bold mb-1 text-center", style={"fontSize": "10px"}),
                    dbc.ButtonGroup([
                        dbc.Button("↶ Undo", id="btn-undo", color="link", size="sm", disabled=True, className="text-decoration-none text-dark"),
                        dbc.Button("↷ Redo", id="btn-redo", color="link", size="sm", disabled=True, className="text-decoration-none text-dark"),
                    ], vertical=True)
                ], width="auto"),
            ], align="center")
        ], style={"display": "block"}),

        # --- DATA TAB ---
        html.Div(id="ribbon-content-data", children=[
            dbc.Row([
                # 1. Data Source (Input)
                dbc.Col([
                    html.Div("1. Data Source", className="text-muted small fw-bold mb-1 text-center", style={"fontSize": "10px"}),
                    dbc.ButtonGroup([
                        dcc.Upload(
                            dbc.Button([html.Div("📂", className="h5 mb-0"), "Import CSV"], color="light", size="sm", className="d-flex flex-column align-items-center px-3 border-0"),
                            id="upload-csv", multiple=False
                        ),
                        dbc.Button([html.Div("🎲", className="h5 mb-0"), "Load Demo"], id="btn-gen-demo", color="light", size="sm", className="d-flex flex-column align-items-center px-3 border-0"),
                    ], className="me-2")
                ], width="auto", className="border-end pe-2"),

                # 2. Data Manager (Selection & Cleaning)
                dbc.Col([
                    html.Div("2. Active Dataset", className="text-muted small fw-bold mb-1 text-center", style={"fontSize": "10px"}),
                    dbc.Row([
                        dbc.Col(dcc.Dropdown(id='dd-dataframe-select', options=[], placeholder="Select Data...", className="small", style={"width": "180px"}), className="pe-1"),
                        dbc.Col(dbc.Button("🗑️", id="btn-delete-data", color="danger", outline=True, size="sm"), width="auto", className="ps-0"),
                    ], align="center", className="g-0"),
                    html.Div(id="data-info-label", className="small text-muted mt-1 text-center", style={"fontSize": "9px"}, children="No data loaded")
                ], width="auto", className="border-end pe-2"),

                # 3. Data Inspection (View)
                dbc.Col([
                    html.Div("3. Inspection", className="text-muted small fw-bold mb-1 text-center", style={"fontSize": "10px"}),
                    dbc.ButtonGroup([
                        dbc.Button([html.Div("📋", className="h5 mb-0"), "Raw Table"], id="btn-view-table", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                        dbc.Button([html.Div("📊", className="h5 mb-0"), "Summary"], id="btn-view-stats", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                        dbc.Button([html.Div("ℹ️", className="h5 mb-0"), "Col Types"], id="btn-view-types", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                    ], className="me-2")
                ], width="auto", className="border-end pe-2"),
                
                # 4. Pre-processing (Simple Operations)
                dbc.Col([
                    html.Div("4. Pre-processing", className="text-muted small fw-bold mb-1 text-center", style={"fontSize": "10px"}),
                    dbc.ButtonGroup([
                        dbc.Button([html.Div("🧹", className="h5 mb-0"), "Clean NA"], id="btn-clean-na", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                        dbc.Button([html.Div("✂️", className="h5 mb-0"), "Remove Sel"], id="btn-remove-selected", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                        dbc.Button([html.Div("🔄", className="h5 mb-0"), "Reset"], id="btn-reset-data", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                    ])
                ], width="auto"),
            ], align="center")
        ], style={"display": "none"}),

        # --- PLOTS TAB ---
        html.Div(id="ribbon-content-plots", children=[
            dbc.Row([
                # 1. Basic 2D
                dbc.Col([
                    html.Div("Basic 2D", className="text-muted small fw-bold mb-1 text-center", style={"fontSize": "10px"}),
                    dbc.ButtonGroup([
                        dbc.Button([html.Div("📈", className="h5 mb-0"), "Scatter"], id="btn-plot-scatter", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                        dbc.Button([html.Div("📉", className="h5 mb-0"), "Line"], id="btn-plot-line", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                        dbc.Button([html.Div("📊", className="h5 mb-0"), "Bar"], id="btn-plot-bar", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                        dbc.Button([html.Div("⛰️", className="h5 mb-0"), "Area"], id="btn-plot-area", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                        dbc.Button([html.Div("🫧", className="h5 mb-0"), "Bubble"], id="btn-plot-bubble", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                    ], className="me-1")
                ], width="auto", className="border-end pe-1"),
                
                # 2. Distribution & Part-of-Whole
                dbc.Col([
                    html.Div("Distribution", className="text-muted small fw-bold mb-1 text-center", style={"fontSize": "10px"}),
                    dbc.ButtonGroup([
                        dbc.Button([html.Div("📶", className="h5 mb-0"), "Hist"], id="btn-plot-hist", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                        dbc.Button([html.Div("📦", className="h5 mb-0"), "Box"], id="btn-plot-box", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                        dbc.Button([html.Div("🎻", className="h5 mb-0"), "Violin"], id="btn-plot-violin", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                        dbc.Button([html.Div("🔥", className="h5 mb-0"), "Heatmap"], id="btn-plot-heatmap", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                        dbc.Button([html.Div("🥧", className="h5 mb-0"), "Pie"], id="btn-plot-pie", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                        dbc.Button([html.Div("🍩", className="h5 mb-0"), "Sunburst"], id="btn-plot-sunburst", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                        dbc.Button([html.Div("🌳", className="h5 mb-0"), "Treemap"], id="btn-plot-treemap", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                    ], className="me-1")
                ], width="auto", className="border-end pe-1"),

                # 3. 3D & Contour
                dbc.Col([
                    html.Div("3D & Contour", className="text-muted small fw-bold mb-1 text-center", style={"fontSize": "10px"}),
                    dbc.ButtonGroup([
                        dbc.Button([html.Div("🧊", className="h5 mb-0"), "Scatter3D"], id="btn-plot-scatter3d", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                        dbc.Button([html.Div("🌀", className="h5 mb-0"), "Line3D"], id="btn-plot-line3d", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                        dbc.Button([html.Div("🌐", className="h5 mb-0"), "Surface"], id="btn-plot-surface", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                        dbc.Button([html.Div("🗺️", className="h5 mb-0"), "Contour"], id="btn-plot-contour", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                    ], className="me-1")
                ], width="auto", className="border-end pe-1"),

                # 4. Specialized (Polar, Ternary, etc.)
                dbc.Col([
                    html.Div("Specialized", className="text-muted small fw-bold mb-1 text-center", style={"fontSize": "10px"}),
                    dbc.ButtonGroup([
                        dbc.Button([html.Div("🕸️", className="h5 mb-0"), "Polar"], id="btn-plot-polar", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                        dbc.Button([html.Div("🔺", className="h5 mb-0"), "Ternary"], id="btn-plot-ternary", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                        dbc.Button([html.Div("🌪️", className="h5 mb-0"), "Funnel"], id="btn-plot-funnel", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                        dbc.Button([html.Div("🕯️", className="h5 mb-0"), "Candle"], id="btn-plot-candle", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                        dbc.Button([html.Div("🌊", className="h5 mb-0"), "Waterfall"], id="btn-plot-waterfall", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                        dbc.Button([html.Div("🔢", className="h5 mb-0"), "ScatMat"], id="btn-plot-scatmat", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                        dbc.Button([html.Div("||", className="h5 mb-0"), "ParCoords"], id="btn-plot-parcoords", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                    ], className="me-1")
                ], width="auto", className="border-end pe-1"),

                # 5. Maps & Geo
                dbc.Col([
                    html.Div("Maps & Geo", className="text-muted small fw-bold mb-1 text-center", style={"fontSize": "10px"}),
                    dbc.ButtonGroup([
                        dbc.Button([html.Div("🌍", className="h5 mb-0"), "ScatGeo"], id="btn-plot-scatgeo", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                        dbc.Button([html.Div("🗺️", className="h5 mb-0"), "Choro"], id="btn-plot-choropleth", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                        dbc.Button([html.Div("🌐", className="h5 mb-0"), "Globe"], id="btn-plot-globe", color="light", size="sm", className="d-flex flex-column align-items-center px-2 border-0"),
                    ], className="me-1")
                ], width="auto", className="border-end pe-1"),
            ], align="center", className="flex-nowrap", style={"overflowX": "auto"})
        ], style={"display": "none"}),

        # --- ANNOTATE TAB ---
        html.Div(id="ribbon-content-annotate", children=[
            dbc.Row([
                dbc.Col([
                    html.Div("Shapes", className="text-muted small fw-bold mb-1 text-center", style={"fontSize": "10px"}),
                    dbc.ButtonGroup([
                        dbc.Button([html.Span("📏"), " Line"], id="btn-draw-line", outline=True, color="dark", size="sm", className="border-0"),
                        dbc.Button([html.Span("⬜"), " Rect"], id="btn-draw-rect", outline=True, color="dark", size="sm", className="border-0"),
                        dbc.Button([html.Span("⭕"), " Circle"], id="btn-draw-circle", outline=True, color="dark", size="sm", className="border-0"),
                        dbc.Button([html.Span("✏️"), " Free"], id="btn-draw-free", outline=True, color="dark", size="sm", className="border-0"),
                        dbc.Button([html.Span("⬡"), " Poly"], id="btn-draw-poly", outline=True, color="dark", size="sm", className="border-0"),
                    ])
                ], width="auto", className="border-end pe-2"),
                dbc.Col([
                    html.Div("Text / Arrow", className="text-muted small fw-bold mb-1 text-center", style={"fontSize": "10px"}),
                    dbc.Button([html.Div("📝", className="h4 mb-0"), "Add Annotation"], id="btn-add-text", color="light", size="sm", className="d-flex flex-column align-items-center px-3 border-0"),
                ], width="auto", className="border-end pe-2"),
                dbc.Col([
                    html.Div("Media", className="text-muted small fw-bold mb-1 text-center", style={"fontSize": "10px"}),
                    dcc.Upload(
                        dbc.Button([html.Div("🖼️", className="h4 mb-0"), "Add Image"], color="light", size="sm", className="d-flex flex-column align-items-center px-3 border-0"),
                        id="upload-image", multiple=False
                    ),
                ], width="auto"),
            ], align="center")
        ], style={"display": "none"}),

        # --- VIEW TAB ---
        html.Div(id="ribbon-content-view", children=[
            dbc.Row([
                dbc.Col([
                    html.Div("Navigation", className="text-muted small fw-bold mb-1 text-center", style={"fontSize": "10px"}),
                    dbc.ButtonGroup([
                        dbc.Button([html.Div("🔍", className="h5 mb-0"), "Zoom"], id="btn-tool-zoom", color="light", size="sm", className="d-flex flex-column align-items-center px-3 border-0"),
                        dbc.Button([html.Div("✋", className="h5 mb-0"), "Pan"], id="btn-tool-pan", color="light", size="sm", className="d-flex flex-column align-items-center px-3 border-0"),
                        dbc.Button([html.Div("🏠", className="h5 mb-0"), "Reset"], id="btn-tool-reset", color="light", size="sm", className="d-flex flex-column align-items-center px-3 border-0"),
                    ], className="me-2")
                ], width="auto", className="border-end pe-2"),
                dbc.Col([
                    html.Div("Panels", className="text-muted small fw-bold mb-1 text-center", style={"fontSize": "10px"}),
                    dbc.Checklist(
                        options=[{"label": "Inspector", "value": "show"}],
                        value=["show"],
                        id="chk-inspector-toggle",
                        switch=True,
                        className="mt-2 ms-2"
                    )
                ], width="auto"),
            ], align="center")
        ], style={"display": "none"}),

    ], className="py-1 px-2 bg-white border-top-0", style={"height": "90px"})
], className="mb-1 shadow-sm rounded-0")

# ====================================================================
# 8. UI Components: Workspace & Inspector
# ====================================================================

# --- Workspace Panel (Left) ---
workspace_panel = dbc.Card([
    dbc.CardHeader(
        dbc.Tabs([
            dbc.Tab(label="Command Window", tab_id="tab-cmd", label_style={"fontSize": "12px"}),
            dbc.Tab(label="Data View", tab_id="tab-dataview", label_style={"fontSize": "12px"}),
        ], id="workspace-tabs", active_tab="tab-cmd", className="nav-tabs-bottom"),
        className="py-1 px-2 bg-light"
    ),
    dbc.CardBody([
        # Command Window Content
        html.Div(id="workspace-content-cmd", children=[
            dcc.Textarea(
                id='code-editor',
                value="# Python Command Window\n# Select a dataset and click a plot button...",
                style={'width': '100%', 'height': '300px', 'fontFamily': 'Consolas, monospace', 'fontSize': '13px', 'backgroundColor': '#f8f9fa', 'border': '1px solid #dee2e6'},
                className="mb-2"
            ),
            dbc.Button("▶ Run Code", id="btn-run-custom-code", color="success", size="sm", className="w-100 mb-2"),
            html.Div("Console Output:", className="small fw-bold text-muted"),
            html.Div(
                id="console-output",
                style={'width': '100%', 'height': '150px', 'fontFamily': 'Consolas, monospace', 'fontSize': '12px', 'backgroundColor': '#212529', 'color': '#00ff00', 'padding': '5px', 'overflowY': 'auto'},
                children=">>> Ready."
            )
        ], style={"display": "block"}),
        
        # Data View Content
        html.Div(id="workspace-content-dataview", children=[
            html.Div(id="data-table-container", children="No data loaded.")
        ], style={"display": "none"}),
    ], className="p-2")
], className="h-100 border-0")

# --- Property Inspector (Right) ---
property_inspector = dbc.Card([
    dbc.CardHeader("Property Inspector", className="bg-light py-1 px-2 small fw-bold"),
    dbc.CardBody([
        # 1. Element Selector
        html.Div([
            html.Label("Select Element:", className="small fw-bold"),
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id='dd-element-select', 
                    placeholder="Select element...", 
                    className="small mb-2",
                    clearable=False,
                    style={"fontSize": "12px"}
                ), width=9, className="pe-1"),
                dbc.Col(dbc.Button("🔦", id="btn-highlight", color="warning", size="sm", outline=True, title="Highlight Selected"), width=3, className="ps-0"),
            ], className="g-0"),
        ], className="mb-3 border-bottom pb-2"),

        # 2. Dynamic Properties Container
        html.Div(id="inspector-controls", children=[
            html.Div("Select an element above to edit its properties.", className="text-muted small text-center mt-5")
        ])
        
    ], style={"overflowY": "auto", "padding": "10px"})
], className="h-100 border-start rounded-0")

# --- Annotation Modal ---
annotation_modal = dbc.Modal([
    dbc.ModalHeader("Add Annotation / Arrow"),
    dbc.ModalBody([
        dbc.Label("Text Content:"),
        dbc.Input(id="annot-text", placeholder="Enter text...", className="mb-3"),
        dbc.Row([
            dbc.Col([dbc.Label("X Coordinate:"), dbc.Input(id="annot-x", type="number", placeholder="Auto (Center)")], width=6),
            dbc.Col([dbc.Label("Y Coordinate:"), dbc.Input(id="annot-y", type="number", placeholder="Auto (Center)")], width=6),
        ], className="mb-3"),
        dbc.Checklist(
            options=[{"label": "Show Arrow", "value": "arrow"}],
            value=["arrow"],
            id="annot-arrow",
            switch=True
        )
    ]),
    dbc.ModalFooter(
        dbc.Button("Add Annotation", id="btn-confirm-annot", color="primary", n_clicks=0)
    )
], id="modal-annotation", is_open=False)

# --- Main Layout ---
app.layout = html.Div([
    dcc.Store(id='figure-store-client', data=figure_store.get_figure_dict()),
    dcc.Store(id='active-dataframe-name', data="demo_signal"),
    dcc.Store(id='trigger-run-signal', data=0), # Signal to auto-run code
    dcc.Store(id='data-update-signal', data=0), # Signal to refresh data view
    dcc.Download(id="download-component"),
    
    # Top Ribbon
    ribbon,
    
    # Main Workspace Area
    dbc.Container([
        dbc.Row([
            # Left: Workspace / Command Window
            dbc.Col(workspace_panel, width=3, className="pe-0 border-end", style={"height": "calc(100vh - 100px)"}),
            
            # Center: Canvas
            dbc.Col([
                dcc.Graph(
                    id='main-graph',
                    figure=create_initial_figure(),
                    config={
                        'editable': True, 
                        'scrollZoom': True,
                        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']
                    },
                    style={'height': '100%', 'width': '100%'}
                )
            ], width=6, className="p-0", style={"height": "calc(100vh - 100px)"}),
            
            # Right: Inspector
            dbc.Col(property_inspector, width=3, className="ps-0", id="col-inspector", style={"height": "calc(100vh - 100px)"}),
        ], className="g-0 h-100")
    ], fluid=True, className="px-0 h-100"),
    
    annotation_modal,
    
], style={"height": "100vh", "overflow": "hidden", "backgroundColor": "#f4f6f8"})

# ====================================================================
# 9. Callbacks: UI Interaction
# ====================================================================

# --- Ribbon & Workspace Tab Switching ---
@app.callback(
    Output("ribbon-content-home", "style"),
    Output("ribbon-content-data", "style"),
    Output("ribbon-content-plots", "style"),
    Output("ribbon-content-annotate", "style"),
    Output("ribbon-content-view", "style"),
    Input("ribbon-tabs", "active_tab"),
)
def toggle_ribbon(active_tab):
    show = {"display": "block"}
    hide = {"display": "none"}
    return (
        show if active_tab == "tab-home" else hide,
        show if active_tab == "tab-data" else hide,
        show if active_tab == "tab-plots" else hide,
        show if active_tab == "tab-annotate" else hide,
        show if active_tab == "tab-view" else hide,
    )

@app.callback(
    Output("workspace-content-cmd", "style"),
    Output("workspace-content-dataview", "style"),
    Input("workspace-tabs", "active_tab"),
)
def toggle_workspace(active_tab):
    show = {"display": "block"}
    hide = {"display": "none"}
    return (show if active_tab == "tab-cmd" else hide, show if active_tab == "tab-dataview" else hide)

@app.callback(
    Output("col-inspector", "style"),
    Input("chk-inspector-toggle", "value"),
)
def toggle_inspector(value):
    if not value:
        return {"display": "none"}
    return {"display": "block", "height": "calc(100vh - 100px)"}

# ====================================================================
# 10a. Callbacks: Data Management
# ====================================================================

@app.callback(
    Output("dd-dataframe-select", "options"), # Added missing output
    Output("dd-dataframe-select", "value"),
    Output("data-info-label", "children"),
    Output("data-update-signal", "data"), # Primary owner
    Output("main-graph", "figure", allow_duplicate=True), # Sync plot with data changes
    Input("upload-csv", "contents"),
    Input("btn-gen-demo", "n_clicks"),
    Input("btn-delete-data", "n_clicks"),
    Input("btn-clean-na", "n_clicks"),
    State("upload-csv", "filename"),
    State("dd-dataframe-select", "value"),
    State("data-update-signal", "data"),
    prevent_initial_call=True
)
def manage_data(upload_content, _n_demo, _n_delete, _n_clean, filename, current_selection, current_signal):
    ctx_id = ctx.triggered_id
    current_signal = current_signal or 0
    fig_update = dash.no_update
    
    # Handle CSV Upload
    if ctx_id == "upload-csv" and upload_content:
        _content_type, content_string = upload_content.split(',')
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            name = filename.split('.')[0]
            figure_store.add_dataframe(name, df)
            current_selection = name.replace(" ", "_").replace("-", "_").replace(".", "_") # Auto-select new data
            current_signal += 1
        except Exception as e:
            print(f"Error parsing CSV: {e}")

    # Handle Demo Data Generation
    if ctx_id == "btn-gen-demo":
        # Generate a richer dataset suitable for ALL plot types
        n_points = 200
        t = np.linspace(0, 10, n_points)
        
        # Categories for Pie/Bar
        categories = np.random.choice(['Category A', 'Category B', 'Category C', 'Category D'], n_points)
        
        # OHLC for Candlestick
        price = 100 + np.cumsum(np.random.randn(n_points))
        high = price + np.random.rand(n_points) * 5
        low = price - np.random.rand(n_points) * 5
        open_p = price + np.random.randn(n_points)
        close_p = price + np.random.randn(n_points)
        
        # Geo Data for Maps
        lat = np.random.uniform(-50, 70, n_points)
        lon = np.random.uniform(-120, 140, n_points)
        countries = np.random.choice(['USA', 'CAN', 'GBR', 'FRA', 'DEU', 'CHN', 'JPN', 'BRA', 'IND', 'AUS'], n_points)
        iso_map = {'USA': 'USA', 'CAN': 'CAN', 'GBR': 'GBR', 'FRA': 'FRA', 'DEU': 'DEU', 'CHN': 'CHN', 'JPN': 'JPN', 'BRA': 'BRA', 'IND': 'IND', 'AUS': 'AUS'}
        iso_codes = [iso_map[c] for c in countries]

        df = pd.DataFrame({
            "time": t,
            "signal": np.sin(t) * 10 + np.random.normal(0, 1, n_points),
            "noise": np.random.randn(n_points),
            "category": categories,
            "x_val": np.random.randn(n_points) * 10,
            "y_val": np.random.randn(n_points) * 10,
            "z_val": np.random.randn(n_points) * 10,  # Explicit Z column for 3D
            "size_val": np.random.randint(5, 20, n_points), # For Bubble
            "open": open_p, "high": high, "low": low, "close": close_p, # For Candle
            "lat": lat, "lon": lon, "country": countries, "iso_alpha": iso_codes # For Maps
        })
        name = f"demo_{uuid.uuid4().hex[:4]}"
        figure_store.add_dataframe(name, df)
        current_selection = name
        current_signal += 1

    # Handle Deletion
    if ctx_id == "btn-delete-data" and current_selection:
        if current_selection in figure_store.data_repository:
            del figure_store.data_repository[current_selection]
            
            # Also remove from datasets (traces) to keep plot in sync
            keys_to_remove = [k for k, d in figure_store.datasets.items() if d.name == current_selection]
            for k in keys_to_remove:
                del figure_store.datasets[k]
                if k in figure_store.dataset_order:
                    figure_store.dataset_order.remove(k)
            
            if keys_to_remove:
                figure_store.rebuild_figure_from_datasets()
                fig_update = figure_store.get_figure_dict()
            
            current_selection = None
            current_signal += 1

    # Handle Cleaning
    if ctx_id == "btn-clean-na" and current_selection:
        df = figure_store.get_dataframe(current_selection)
        if df is not None:
            # Smart Cleaning Logic
            for col in df.columns:
                if df[col].dtype == 'object':
                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                    num_valid = numeric_col.count()
                    total_valid = df[col].count()
                    if total_valid > 0 and (num_valid / total_valid) > 0.5:
                        df[col] = numeric_col
            
            df = df.dropna()
            figure_store.add_dataframe(current_selection, df)
            
            # Update Datasets & Figure to reflect cleaned data
            updated_datasets = False
            for key, dataset in figure_store.datasets.items():
                if dataset.name == current_selection:
                    dataset.df = df
                    updated_datasets = True
            
            if updated_datasets:
                figure_store.rebuild_figure_from_datasets()
                fig_update = figure_store.get_figure_dict()
                
            current_signal += 1

    # Update Options
    options = [{"label": k, "value": k} for k in figure_store.data_repository.keys()]
    
    # Fallback selection
    if current_selection not in figure_store.data_repository:
        current_selection = options[-1]["value"] if options else None
        
    # Info Label
    info_text = "No data loaded"
    if current_selection:
        df = figure_store.get_dataframe(current_selection)
        if df is not None:
            info_text = f"{len(df)} rows × {len(df.columns)} cols"
    
    return options, current_selection, info_text, current_signal, fig_update

@app.callback(
    Output("data-table-container", "children"),
    Output("workspace-tabs", "active_tab"),
    Input("dd-dataframe-select", "value"),
    Input("btn-view-table", "n_clicks"),
    Input("btn-view-stats", "n_clicks"),
    Input("btn-view-types", "n_clicks"),
    Input("data-update-signal", "data"), # Listen to signal
    State("workspace-tabs", "active_tab"),
)
def update_data_view(df_name, _n_table, _n_stats, _n_types, _signal, active_tab):
    ctx_id = ctx.triggered_id
    
    # Auto-switch to Data View tab if buttons clicked OR if new data selected
    if ctx_id in ["btn-view-table", "btn-view-stats", "btn-view-types", "dd-dataframe-select"]:
        active_tab = "tab-dataview"
        
    if not df_name:
        return "No data selected.", active_tab
        
    df = figure_store.get_dataframe(df_name)
    if df is None:
        return "Data not found.", active_tab

    # Determine content type
    if ctx_id == "btn-view-stats":
        # Show Statistics
        stats_df = df.describe().reset_index()
        content = dash_table.DataTable(
            data=cast(List[Dict[str, Any]], stats_df.to_dict('records')),
            columns=[{"name": i, "id": i} for i in stats_df.columns],
            style_table={'overflowX': 'auto'},
            style_cell={'fontSize': '12px', 'fontFamily': 'Arial', 'textAlign': 'left'},
            style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'}
        )
    elif ctx_id == "btn-view-types":
        # Show Column Types
        types_df = pd.DataFrame(df.dtypes, columns=['Dtype']).reset_index().rename(columns={'index': 'Column'})
        types_df['Dtype'] = types_df['Dtype'].astype(str)
        content = dash_table.DataTable(
            data=cast(List[Dict[str, Any]], types_df.to_dict('records')),
            columns=[{"name": i, "id": i} for i in types_df.columns],
            style_cell={'fontSize': '12px', 'fontFamily': 'Arial', 'textAlign': 'left'},
            style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'}
        )
    else:
        # Default: Show Data Table (First 100 rows)
        # ENABLE EDITING HERE
        content = dash_table.DataTable(
            id='interactive-data-table',
            data=cast(List[Dict[str, Any]], df.head(100).to_dict('records')),
            columns=[{"name": i, "id": i, "editable": True} for i in df.columns],
            editable=True,
            row_deletable=True,
            style_table={'overflowX': 'auto'},
            style_cell={'fontSize': '12px', 'fontFamily': 'Arial', 'textAlign': 'left'},
            page_size=20,
            style_header={'fontWeight': 'bold', 'backgroundColor': '#e9ecef'}
        )
        
    return content, active_tab

# New Callback: Sync Edited Table Data back to Repository
@app.callback(
    Output("console-output", "children", allow_duplicate=True),
    Output("main-graph", "figure", allow_duplicate=True), # Add figure output
    Input("interactive-data-table", "data"),
    State("dd-dataframe-select", "value"),
    State("console-output", "children"),
    prevent_initial_call=True
)
def sync_data_from_table(rows, df_name, current_console):
    if not rows or not df_name:
        raise PreventUpdate
        
    # Convert back to DataFrame
    try:
        # 1. Get original DF to preserve types if possible
        original_df = figure_store.get_dataframe(df_name)
        new_df = pd.DataFrame(rows)
        
        # Attempt to restore types
        if original_df is not None:
            for col in new_df.columns:
                if col in original_df.columns:
                    try:
                        # Try to cast to original type
                        if pd.api.types.is_numeric_dtype(original_df[col]):
                            new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
                        elif pd.api.types.is_datetime64_any_dtype(original_df[col]):
                            new_df[col] = pd.to_datetime(new_df[col], errors='coerce')
                        # Add more type checks as needed
                    except:
                        pass # Keep as is if cast fails
        
        # 2. Update Repository
        figure_store.add_dataframe(df_name, new_df)
        
        # 3. Update Datasets (Traces) that use this dataframe
        # This ensures that if we re-plot or interact, the traces are in sync
        updated_count = 0
        for key, dataset in figure_store.datasets.items():
            if dataset.name == df_name:
                dataset.df = new_df
                updated_count += 1
        
        msg = f">>> Data '{df_name}' updated from table ({len(new_df)} rows)."
        fig_update = dash.no_update
        
        if updated_count > 0:
            figure_store.rebuild_figure_from_datasets()
            fig_update = figure_store.get_figure_dict()
            msg += f" Synced {updated_count} active traces."
            
        return f"{current_console}\n{msg}", fig_update
    except Exception as e:
        return f"{current_console}\n>>> Error updating data: {e}", dash.no_update

# ====================================================================
# 10b. Callbacks: Data Interaction (Selection & Removal)
# ====================================================================

@app.callback(
    Output("main-graph", "figure", allow_duplicate=True),
    Output("main-graph", "selectedData"), # Explicitly clear selection
    Output("console-output", "children", allow_duplicate=True),
    Output("data-update-signal", "data", allow_duplicate=True),
    Input("btn-remove-selected", "n_clicks"),
    State("main-graph", "selectedData"),
    State("main-graph", "figure"),
    State("dd-dataframe-select", "value"),
    State("data-update-signal", "data"),
    prevent_initial_call=True
)
def remove_selected_points(n_clicks, selected_data, fig_dict, df_name, current_signal):
    if not n_clicks:
        raise PreventUpdate
        
    if not selected_data:
        return dash.no_update, dash.no_update, ">>> No points selected to remove. (selectedData is None)", dash.no_update
        
    points = selected_data.get("points", [])
    if not points:
        keys = list(selected_data.keys())
        return dash.no_update, dash.no_update, f">>> Selection is empty. Keys found: {keys}", dash.no_update
        
    # Debug info
    first_pt = points[0]
    print(f"DEBUG: Removing {len(points)} points. Sample: {first_pt}")
    
    msg = ""
    current_signal = current_signal or 0
    data_changed = False
    
    # 1. Try to update the underlying Data Repository (The "Real" Data)
    if df_name and df_name in figure_store.data_repository:
        try:
            df = figure_store.data_repository[df_name]
            indices_to_drop = set()
            for pt in points:
                p_idx = pt.get("pointIndex")
                if p_idx is not None and 0 <= p_idx < len(df):
                    indices_to_drop.add(df.index[p_idx])
            
            if indices_to_drop:
                new_df = df.drop(list(indices_to_drop))
                figure_store.add_dataframe(df_name, new_df)
                
                # Sync datasets (traces) to ensure consistency
                updated_count = 0
                for key, dataset in figure_store.datasets.items():
                    if dataset.name == df_name:
                        dataset.df = new_df
                        updated_count += 1
                
                msg += f">>> Removed {len(indices_to_drop)} rows from dataset '{df_name}'. (Synced {updated_count} traces)\n"
                data_changed = True
            else:
                msg += f">>> No matching rows found in '{df_name}' (Index mismatch?).\n"
        except Exception as e:
            msg += f">>> Error updating dataset: {e}\n"
    else:
        msg += ">>> No active dataset selected. Only updating plot.\n"

    # 2. Update the Visual Figure (The "Image")
    try:
        # Group points by curveNumber
        points_by_curve = defaultdict(list)
        for pt in points:
            points_by_curve[pt['curveNumber']].append(pt['pointIndex'])
        
        # We edit the fig_dict directly to preserve exactly what is on screen
        fig = go.Figure(fig_dict)
        any_visual_change = False
        
        for curve_idx, p_indices in points_by_curve.items():
            if curve_idx < len(fig.data):
                trace = fig.data[curve_idx]
                # Only handle traces with x/y arrays
                if hasattr(trace, 'x') and trace.x is not None:
                    x_list = list(trace.x)
                    y_list = list(trace.y) if hasattr(trace, 'y') and trace.y is not None else []
                    
                    # Create a mask for keeping points
                    p_indices_set = set(p_indices)
                    mask = [i not in p_indices_set for i in range(len(x_list))]
                    
                    # Apply mask
                    new_x = [x for i, x in enumerate(x_list) if mask[i]]
                    updates = {'x': new_x}
                    
                    if y_list:
                        new_y = [y for i, y in enumerate(y_list) if mask[i]]
                        updates['y'] = new_y
                    
                    # Also handle marker colors/sizes if they are arrays
                    # Use getattr to safely access marker, and check for color
                    marker = getattr(trace, 'marker', None)
                    if marker:
                        color = getattr(marker, 'color', None)
                        if isinstance(color, (list, tuple, np.ndarray)) and len(color) == len(x_list):
                            new_color = [c for i, c in enumerate(color) if mask[i]]
                            # Use nested update for marker
                            updates['marker'] = dict(color=new_color)
                            
                    trace.update(updates)
                    any_visual_change = True
        
        new_signal = (current_signal + 1) if data_changed else dash.no_update
        
        # Force clear selections in layout just in case
        fig.update_layout(selections=[])
        
        if any_visual_change:
            figure_store.update_figure(fig)
            # Return None for selectedData to clear it on the client side
            return fig, None, msg + " (Visual update applied).", new_signal
        else:
            return dash.no_update, dash.no_update, msg + " (No visual changes possible).", new_signal
            
    except Exception as e:
        return dash.no_update, dash.no_update, msg + f" (Visual update failed: {e})", dash.no_update

# ====================================================================
# 11. Callbacks: Code Generation & Execution
# ====================================================================

@app.callback(
    Output("code-editor", "value"),
    Output("trigger-run-signal", "data"), # Signal to auto-run
    Input("btn-plot-scatter", "n_clicks"),
    Input("btn-plot-line", "n_clicks"),
    Input("btn-plot-bar", "n_clicks"),
    Input("btn-plot-area", "n_clicks"),
    Input("btn-plot-bubble", "n_clicks"),
    Input("btn-plot-pie", "n_clicks"),
    Input("btn-plot-sunburst", "n_clicks"),
    Input("btn-plot-treemap", "n_clicks"),
    Input("btn-plot-heatmap", "n_clicks"),
    Input("btn-plot-scatter3d", "n_clicks"),
    Input("btn-plot-line3d", "n_clicks"),
    Input("btn-plot-surface", "n_clicks"),
    Input("btn-plot-contour", "n_clicks"),
    Input("btn-plot-hist", "n_clicks"),
    Input("btn-plot-box", "n_clicks"),
    Input("btn-plot-violin", "n_clicks"),
    Input("btn-plot-polar", "n_clicks"),
    Input("btn-plot-ternary", "n_clicks"),
    Input("btn-plot-funnel", "n_clicks"),
    Input("btn-plot-candle", "n_clicks"),
    Input("btn-plot-waterfall", "n_clicks"),
    Input("btn-plot-scatmat", "n_clicks"),
    Input("btn-plot-parcoords", "n_clicks"),
    Input("btn-plot-scatgeo", "n_clicks"),
    Input("btn-plot-choropleth", "n_clicks"),
    Input("btn-plot-globe", "n_clicks"),
    State("dd-dataframe-select", "value"),
    State("trigger-run-signal", "data"),
    prevent_initial_call=True
)
def generate_and_trigger_plot(_n_sc, _n_ln, _n_bar, _n_area, _n_bub, _n_pie, _n_sun, _n_tree, _n_heat, _n_3d, _n_line3d, _n_surf, _n_cont, _n_hist, _n_box, _n_violin, _n_pol, _n_ter, _n_fun, _n_can, _n_wat, _n_smat, _n_par, _n_geo, _n_choro, _n_globe, df_name, current_signal):
    if not df_name:
        return "# Please select a dataset first.", dash.no_update
        
    ctx_id = ctx.triggered_id
    if not ctx_id:
        raise PreventUpdate
    plot_type = ctx_id.replace("btn-plot-", "")
    
    df = figure_store.get_dataframe(df_name)
    if df is None:
        return "# Error: Dataset not found.", dash.no_update

    # Use the enhanced CodeGenerator logic
    cmd = code_generator.generate_smart_plot_code(df_name, plot_type, df)
    
    # Increment signal to trigger execution
    new_signal = (current_signal or 0) + 1
    return cmd, new_signal

@app.callback(
    Output("main-graph", "figure"),
    Output("console-output", "children"),
    Input("trigger-run-signal", "data"),
    Input("btn-run-custom-code", "n_clicks"),
    State("code-editor", "value"),
    State("console-output", "children"),
    prevent_initial_call=True
)
def execute_code(signal, n_clicks, code, current_console):
    ctx_id = ctx.triggered_id
    
    # Only run if triggered by signal (auto-run) or button click
    if not code:
        raise PreventUpdate

    try:
        # Create a safe local scope
        local_scope = {
            "pd": pd, 
            "px": px, 
            "go": go, 
            "np": np,
            "figure_store": figure_store # Allow access to store
        }
        
        # Inject all dataframes into scope
        for name, df in figure_store.data_repository.items():
            local_scope[name] = df
            
        # Execute
        exec(code, {}, local_scope)
        
        # Check if 'fig' was created
        if "fig" in local_scope:
            fig = local_scope["fig"]
            if isinstance(fig, go.Figure):
                figure_store.update_figure(fig)
                return fig, f"{current_console}\n>>> Code executed successfully."
            else:
                return dash.no_update, f"{current_console}\n>>> Error: 'fig' is not a plotly Figure."
        else:
            return dash.no_update, f"{current_console}\n>>> Code executed, but no 'fig' variable found."
            
    except Exception as e:
        return dash.no_update, f"{current_console}\n>>> Execution Error: {e}"

# ====================================================================
# 12. Callbacks: Property Editor
# ====================================================================

@app.callback(
    Output("main-graph", "figure", allow_duplicate=True),
    Input("btn-apply-props", "n_clicks"),
    State("dd-element-select", "value"),
    State("input-prop-name", "value"), # Name/Title
    State("input-prop-color", "value"), # Color
    State("input-prop-size", "value"), # Size/Height
    State("input-prop-opacity", "value"), # Opacity
    State("input-prop-symbol", "value"), # Symbol
    State("input-prop-width", "value"), # Line Width/Width
    State("input-prop-dash", "value"), # Line Dash
    State("input-prop-mode", "value"), # Trace Mode
    State("input-prop-font", "value"), # Font Family
    State("input-prop-text", "value"), # Text
    State("input-prop-x", "value"), # X
    State("input-prop-y", "value"), # Y
    State("input-prop-template", "value"), # Figure Template
    State("input-prop-xaxis", "value"), # X Axis Title
    State("input-prop-yaxis", "value"), # Y Axis Title
    State("input-prop-legend", "value"), # Show Legend
    State("input-prop-fill", "value"), # Trace Fill
    State("input-prop-marker_line_color", "value"), # Marker Line Color (Fixed ID)
    State("input-prop-arrow", "value"), # Show Arrow
    State("input-prop-bgcolor", "value"), # Bg Color
    # --- NEW PROPERTIES (ROUND 3) ---
    State("input-prop-hovermode", "value"), # Hover Mode
    State("input-prop-grid_x", "value"), # Grid X (Fixed ID)
    State("input-prop-grid_y", "value"), # Grid Y (Fixed ID)
    State("input-prop-paper_color", "value"), # Paper Bg Color (Fixed ID)
    State("input-prop-line_shape", "value"), # Line Shape (Fixed ID)
    State("input-prop-text_pos", "value"), # Text Position (Fixed ID)
    State("input-prop-text_angle", "value"), # Text Angle (Fixed ID)
    # --- NEW PROPERTIES (ROUND 4) ---
    State("input-prop-legend_orient", "value"), # Legend Orientation (Fixed ID)
    State("input-prop-legend_pos", "value"), # Legend Position (Fixed ID)
    State("input-prop-barmode", "value"), # Bar Mode
    State("input-prop-log_x", "value"), # Log Scale X (Fixed ID)
    State("input-prop-log_y", "value"), # Log Scale Y (Fixed ID)
    State("input-prop-spikes", "value"), # Show Spikes
    State("input-prop-zeroline", "value"), # Show Zero Line
    State("input-prop-global_font_size", "value"), # Global Font Size (Fixed ID)
    
    State("main-graph", "figure"),
    prevent_initial_call=True
)
def apply_property_changes(n_clicks, selected_element, name, color, size, opacity, symbol, width, dash_style, mode, font, text, x, y, 
                           template, xaxis_title, yaxis_title, show_legend, fill, marker_line_color, show_arrow, bgcolor,
                           hovermode, grid_x, grid_y, paper_color, line_shape, text_pos, text_angle,
                           legend_orient, legend_pos, barmode, log_x, log_y, spikes, zeroline, global_font_size,
                           fig_dict):
    if not n_clicks or not selected_element:
        raise PreventUpdate
        
    fig_dict = clean_figure_dict(fig_dict)
    fig = go.Figure(fig_dict)
    
    # Collect valid updates
    props = {}
    if name: props['name'] = name
    if color: props['color'] = color
    if size: props['size'] = size
    if opacity is not None: props['opacity'] = opacity
    if symbol: props['symbol'] = symbol
    if width: props['width'] = width
    if dash_style: props['dash'] = dash_style
    if mode: props['mode'] = mode
    if font: props['font'] = font
    if text: props['text'] = text
    if x is not None: props['x'] = x
    if y is not None: props['y'] = y
    if template: props['template'] = template
    if xaxis_title: props['xaxis_title'] = xaxis_title
    if yaxis_title: props['yaxis_title'] = yaxis_title
    if show_legend: props['showlegend'] = (show_legend == 'Show')
    if fill: props['fill'] = fill
    if marker_line_color: props['marker_line_color'] = marker_line_color
    if show_arrow: props['showarrow'] = (show_arrow == 'Show')
    if bgcolor: props['bgcolor'] = bgcolor
    # New props R3
    if hovermode: props['hovermode'] = hovermode
    if grid_x: props['grid_x'] = (grid_x == 'Show')
    if grid_y: props['grid_y'] = (grid_y == 'Show')
    if paper_color: props['paper_color'] = paper_color
    if line_shape: props['line_shape'] = line_shape
    if text_pos: props['text_pos'] = text_pos
    if text_angle is not None: props['text_angle'] = text_angle
    # New props R4
    if legend_orient: props['legend_orient'] = legend_orient
    if legend_pos: props['legend_pos'] = legend_pos
    if barmode: props['barmode'] = barmode
    if log_x: props['log_x'] = log_x
    if log_y: props['log_y'] = log_y
    if spikes: props['spikes'] = (spikes == 'Show')
    if zeroline: props['zeroline'] = (zeroline == 'Show')
    if global_font_size: props['global_font_size'] = global_font_size
    
    if not props: return dash.no_update

    if selected_element == "figure":
        layout_updates = {}
        if 'name' in props: layout_updates['title'] = dict(text=props['name'])
        if 'width' in props: layout_updates['width'] = int(props['width'])
        if 'size' in props: layout_updates['height'] = int(props['size']) # Map size to height for figure
        if 'color' in props: layout_updates['plot_bgcolor'] = props['color']
        if 'paper_color' in props: layout_updates['paper_bgcolor'] = props['paper_color']
        if 'font' in props: 
            if 'font' not in layout_updates: layout_updates['font'] = {}
            layout_updates['font']['family'] = props['font']
        if 'global_font_size' in props:
            if 'font' not in layout_updates: layout_updates['font'] = {}
            layout_updates['font']['size'] = int(props['global_font_size'])
            
        if 'template' in props: layout_updates['template'] = props['template']
        if 'showlegend' in props: layout_updates['showlegend'] = props['showlegend']
        if 'hovermode' in props: layout_updates['hovermode'] = props['hovermode']
        if 'barmode' in props: layout_updates['barmode'] = props['barmode']
        
        # Legend Updates
        legend_updates = {}
        if 'legend_orient' in props: legend_updates['orientation'] = props['legend_orient']
        if 'legend_pos' in props:
            pos = props['legend_pos']
            if pos == 'tr': legend_updates.update(x=1.02, y=1, xanchor='left', yanchor='top')
            elif pos == 'tl': legend_updates.update(x=0, y=1, xanchor='left', yanchor='top')
            elif pos == 'br': legend_updates.update(x=1.02, y=0, xanchor='left', yanchor='bottom')
            elif pos == 'bl': legend_updates.update(x=0, y=0, xanchor='left', yanchor='bottom')
        if legend_updates: layout_updates['legend'] = legend_updates
        
        # Axis updates need careful merging
        xaxis_opts = {}
        if 'xaxis_title' in props: xaxis_opts['title'] = props['xaxis_title']
        if 'grid_x' in props: xaxis_opts['showgrid'] = props['grid_x']
        if 'log_x' in props: xaxis_opts['type'] = props['log_x']
        if 'spikes' in props: xaxis_opts['showspikes'] = props['spikes']
        if 'zeroline' in props: xaxis_opts['zeroline'] = props['zeroline']
        if xaxis_opts: 
            fig.update_xaxes(**xaxis_opts)
            
        yaxis_opts = {}
        if 'yaxis_title' in props: yaxis_opts['title'] = props['yaxis_title']
        if 'grid_y' in props: yaxis_opts['showgrid'] = props['grid_y']
        if 'log_y' in props: yaxis_opts['type'] = props['log_y']
        if 'spikes' in props: yaxis_opts['showspikes'] = props['spikes']
        if 'zeroline' in props: yaxis_opts['zeroline'] = props['zeroline']
        if yaxis_opts:
            fig.update_yaxes(**yaxis_opts)
            
        fig.update_layout(**layout_updates)

    elif selected_element.startswith("trace_"):
        idx = int(selected_element.split("_")[1])
        if idx < len(fig.data):
            trace = fig.data[idx]
            updates = {}
            
            if 'name' in props: updates['name'] = props['name']
            if 'opacity' in props: updates['opacity'] = props['opacity']
            if 'mode' in props: updates['mode'] = props['mode']
            if 'fill' in props: updates['fill'] = props['fill'] if props['fill'] != 'none' else None
            if 'text_pos' in props: updates['textposition'] = props['text_pos']
            
            # Marker updates
            marker_updates = {}
            if 'color' in props: marker_updates['color'] = props['color']
            if 'size' in props: marker_updates['size'] = props['size']
            if 'symbol' in props: marker_updates['symbol'] = props['symbol']
            if 'width' in props: marker_updates['line'] = dict(width=props['width']) 
            if 'marker_line_color' in props: 
                if 'line' not in marker_updates: marker_updates['line'] = {}
                marker_updates['line']['color'] = props['marker_line_color']
            if marker_updates: updates['marker'] = marker_updates
            
            # Line updates
            line_updates = {}
            if 'color' in props: line_updates['color'] = props['color']
            if 'width' in props: line_updates['width'] = props['width']
            if 'dash' in props: line_updates['dash'] = props['dash']
            if 'line_shape' in props: line_updates['shape'] = props['line_shape']
            if line_updates: updates['line'] = line_updates
            
            trace.update(updates)

    elif selected_element.startswith("annot_"):
        idx = int(selected_element.split("_")[1])
        if fig.layout.annotations and idx < len(fig.layout.annotations):
            annot = fig.layout.annotations[idx]
            updates = {}
            if 'text' in props: updates['text'] = props['text']
            if 'text_angle' in props: updates['textangle'] = props['text_angle']
            
            font_updates = {}
            if 'color' in props: font_updates['color'] = props['color']
            if 'size' in props: font_updates['size'] = props['size']
            if 'font' in props: font_updates['family'] = props['font']
            if font_updates: updates['font'] = font_updates
            
            if 'x' in props: updates['x'] = props['x']
            if 'y' in props: updates['y'] = props['y']
            if 'showarrow' in props: updates['showarrow'] = props['showarrow']
            if 'bgcolor' in props: updates['bgcolor'] = props['bgcolor']
            
            annot.update(updates)
            
    elif selected_element.startswith("shape_"):
        idx = int(selected_element.split("_")[1])
        if fig.layout.shapes and idx < len(fig.layout.shapes):
            shape = fig.layout.shapes[idx]
            updates = {}
            line_updates = {}
            if 'color' in props: line_updates['color'] = props['color']
            if 'dash' in props: line_updates['dash'] = props['dash']
            if 'width' in props: line_updates['width'] = props['width']
            if line_updates: updates['line'] = line_updates
            
            if 'color' in props: updates['fillcolor'] = props['color'] 
            if 'opacity' in props: updates['opacity'] = props['opacity']
            if 'bgcolor' in props: updates['fillcolor'] = props['bgcolor']
            
            shape.update(updates)

    elif selected_element.startswith("image_"):
        idx = int(selected_element.split("_")[1])
        if fig.layout.images and idx < len(fig.layout.images):
            img = fig.layout.images[idx]
            updates = {}
            if 'opacity' in props: updates['opacity'] = props['opacity']
            if 'size' in props: updates['sizex'] = props['size']; updates['sizey'] = props['size']
            if 'x' in props: updates['x'] = props['x']
            if 'y' in props: updates['y'] = props['y']
            img.update(updates)

    figure_store.update_figure(fig)
    return fig

# New Callback: Delete Selected Element
@app.callback(
    Output("main-graph", "figure", allow_duplicate=True),
    Output("dd-element-select", "value", allow_duplicate=True),
    Input("btn-delete-element", "n_clicks"),
    State("dd-element-select", "value"),
    prevent_initial_call=True
)
def delete_element(n_clicks, selected_element):
    if not n_clicks or not selected_element:
        raise PreventUpdate
    
    if selected_element == "figure":
        return dash.no_update, dash.no_update

    if selected_element.startswith("trace_"):
        idx = int(selected_element.split("_")[1])
        figure_store.remove_trace(idx)
    elif selected_element.startswith("shape_"):
        idx = int(selected_element.split("_")[1])
        figure_store.remove_shape(idx)
    elif selected_element.startswith("annot_"):
        idx = int(selected_element.split("_")[1])
        figure_store.remove_annotation(idx)
    elif selected_element.startswith("image_"):
        idx = int(selected_element.split("_")[1])
        figure_store.remove_image(idx)
    
    # Return updated figure and reset selection to None
    return figure_store.get_figure_dict(), None

# New Callback: Highlight Selected Element (Auto-trigger on selection)
@app.callback(
    Output("main-graph", "figure", allow_duplicate=True),
    Input("btn-highlight", "n_clicks"),
    Input("dd-element-select", "value"), # Auto-highlight on selection
    State("main-graph", "figure"),
    prevent_initial_call=True
)
def highlight_element(n_clicks, selected_element, fig_dict):
    if not selected_element or not fig_dict:
        raise PreventUpdate
        
    fig_dict = clean_figure_dict(fig_dict)
    fig = go.Figure(fig_dict)
    
    # Preserve UI state (zoom/pan)
    fig.update_layout(uirevision=True)
    
    # RELOAD FROM STORE TO GET CLEAN STATE
    clean_fig = figure_store.get_figure_dict()
    if clean_fig:
        fig = go.Figure(clean_fig)
        fig.update_layout(uirevision=True) # Keep zoom
    
    if selected_element == "figure":
        return fig # Just return clean figure if "Figure" is selected

    if selected_element.startswith("trace_"):
        idx = int(selected_element.split("_")[1])
        # Dim all other traces
        for i, trace in enumerate(fig.data):
            if i != idx:
                trace.update(opacity=0.2)
            else:
                trace.update(opacity=1.0)
                # Add a marker border to highlight
                # Check if marker exists or just try to update it
                trace.update(marker=dict(line=dict(width=2, color='red')))
                    
    elif selected_element.startswith("shape_"):
        idx = int(selected_element.split("_")[1])
        # Highlight shape with thick red border
        if fig.layout.shapes and idx < len(fig.layout.shapes):
            shape = fig.layout.shapes[idx]
            shape.update(line=dict(width=4, color="red"), opacity=1.0)
        
    elif selected_element.startswith("annot_"):
        idx = int(selected_element.split("_")[1])
        if fig.layout.annotations and idx < len(fig.layout.annotations):
            annot = fig.layout.annotations[idx]
            annot.update(bordercolor="red", borderwidth=2, bgcolor="rgba(255, 255, 0, 0.3)")

    return fig

# ====================================================================
# NEW: Inspector UI Generation Callbacks (Restored)
# ====================================================================

@app.callback(
    Output("dd-element-select", "options"),
    Input("main-graph", "figure"),
)
def update_element_options(fig_dict):
    if not fig_dict:
        return []
    
    options = [{"label": "Figure Settings", "value": "figure"}]
    
    # Traces
    if "data" in fig_dict:
        for i, trace in enumerate(fig_dict["data"]):
            name = trace.get("name", f"Trace {i}")
            options.append({"label": f"Trace {i}: {name}", "value": f"trace_{i}"})
            
    # Layout items
    layout = fig_dict.get("layout", {})
    
    # Shapes
    if "shapes" in layout:
        for i, shape in enumerate(layout["shapes"]):
            type_ = shape.get("type", "shape")
            options.append({"label": f"Shape {i}: {type_}", "value": f"shape_{i}"})
            
    # Annotations
    if "annotations" in layout:
        for i, annot in enumerate(layout["annotations"]):
            text = annot.get("text", "Annotation")[:20]
            options.append({"label": f"Annot {i}: {text}", "value": f"annot_{i}"})
            
    # Images
    if "images" in layout:
        for i, img in enumerate(layout["images"]):
            options.append({"label": f"Image {i}", "value": f"image_{i}"})
            
    return options

@app.callback(
    Output("inspector-controls", "children"),
    Input("dd-element-select", "value"),
    State("main-graph", "figure"),
)
def update_inspector_controls(selected_element, fig_dict):
    if not selected_element:
        return html.Div("Select an element above to edit its properties.", className="text-muted small text-center mt-5")
    
    fig_dict = clean_figure_dict(fig_dict)
    fig = go.Figure(fig_dict)
    
    controls = []
    
    # Options Lists
    COLORS = ['black', 'white', 'red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'grey', 'brown', 'pink', 'gold', 'teal', 'navy', 'transparent']
    SYMBOLS = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'star', 'hexagram', 'pentagon']
    DASHES = ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
    MODES = ['lines', 'markers', 'lines+markers']
    FONTS = ['Arial', 'Verdana', 'Times New Roman', 'Courier New', 'Georgia', 'Comic Sans MS', 'Impact']
    TEMPLATES = ['plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn', 'simple_white', 'none']
    FILLS = ['none', 'tozeroy', 'tonexty', 'toself']
    LEGEND_OPTS = ['Show', 'Hide']
    ARROW_OPTS = ['Show', 'Hide']
    HOVER_MODES = ['closest', 'x', 'y', 'x unified', 'y unified']
    SHOW_HIDE = ['Show', 'Hide']
    LINE_SHAPES = ['linear', 'spline', 'hv', 'vh', 'hvh', 'vhv']
    TEXT_POSITIONS = ['top left', 'top center', 'top right', 'middle left', 'middle center', 'middle right', 'bottom left', 'bottom center', 'bottom right']
    LEGEND_POSITIONS = [
        {'label': 'Top Right', 'value': 'tr'},
        {'label': 'Top Left', 'value': 'tl'},
        {'label': 'Bottom Right', 'value': 'br'},
        {'label': 'Bottom Left', 'value': 'bl'},
    ]
    BAR_MODES = ['group', 'stack', 'overlay', 'relative']
    AXIS_TYPES = [{'label': 'Linear', 'value': 'linear'}, {'label': 'Log', 'value': 'log'}]
    LEGEND_ORIENTS = [{'label': 'Vertical', 'value': 'v'}, {'label': 'Horizontal', 'value': 'h'}]

    # Helper to create input row
    def make_row(label, id_suffix, input_type="text", value=None, placeholder="", visible=True, options=None):
        style = {} if visible else {"display": "none"}
        
        if input_type == "select" and options:
            input_component = dbc.Select(
                id=f"input-prop-{id_suffix}",
                options=[{"label": o.title(), "value": o} for o in options] if isinstance(options[0], str) else options,
                value=value,
                size="sm"
            )
        else:
            input_component = dbc.Input(
                id=f"input-prop-{id_suffix}", 
                type=input_type, 
                value=value, 
                placeholder=placeholder, 
                size="sm"
            )

        return dbc.Row([
            dbc.Col(dbc.Label(label, className="small mb-0"), width=4, className="d-flex align-items-center"),
            dbc.Col(input_component, width=8)
        ], className="mb-2", style=style)

    # Default configuration for all inputs
    config = {
        "name":    {"visible": False, "label": "Name", "value": None, "type": "text"},
        "color":   {"visible": False, "label": "Color", "value": None, "type": "select", "options": COLORS},
        "size":    {"visible": False, "label": "Size", "value": None, "type": "number"},
        "opacity": {"visible": False, "label": "Opacity", "value": None, "type": "number"},
        "symbol":  {"visible": False, "label": "Symbol", "value": None, "type": "select", "options": SYMBOLS},
        "width":   {"visible": False, "label": "Width", "value": None, "type": "number"},
        "dash":    {"visible": False, "label": "Dash", "value": None, "type": "select", "options": DASHES},
        "mode":    {"visible": False, "label": "Mode", "value": None, "type": "select", "options": MODES},
        "font":    {"visible": False, "label": "Font", "value": None, "type": "select", "options": FONTS},
        "text":    {"visible": False, "label": "Text", "value": None, "type": "text"},
        "x":       {"visible": False, "label": "X", "value": None, "type": "number"},
        "y":       {"visible": False, "label": "Y", "value": None, "type": "number"},
        "template": {"visible": False, "label": "Theme", "value": None, "type": "select", "options": TEMPLATES},
        "xaxis":    {"visible": False, "label": "X Title", "value": None, "type": "text"},
        "yaxis":    {"visible": False, "label": "Y Title", "value": None, "type": "text"},
        "legend":   {"visible": False, "label": "Legend", "value": None, "type": "select", "options": LEGEND_OPTS},
        "fill":     {"visible": False, "label": "Fill", "value": None, "type": "select", "options": FILLS},
        "marker_line_color": {"visible": False, "label": "Border Color", "value": None, "type": "select", "options": COLORS},
        "arrow":    {"visible": False, "label": "Arrow", "value": None, "type": "select", "options": ARROW_OPTS},
        "bgcolor":  {"visible": False, "label": "Bg Color", "value": None, "type": "select", "options": COLORS},
        # New Configs R3
        "hovermode": {"visible": False, "label": "Hover Mode", "value": None, "type": "select", "options": HOVER_MODES},
        "grid_x":    {"visible": False, "label": "Grid X", "value": None, "type": "select", "options": SHOW_HIDE},
        "grid_y":    {"visible": False, "label": "Grid Y", "value": None, "type": "select", "options": SHOW_HIDE},
        "paper_color": {"visible": False, "label": "Paper Color", "value": None, "type": "select", "options": COLORS},
        "line_shape": {"visible": False, "label": "Smoothing", "value": None, "type": "select", "options": LINE_SHAPES},
        "text_pos":   {"visible": False, "label": "Text Pos", "value": None, "type": "select", "options": TEXT_POSITIONS},
        "text_angle": {"visible": False, "label": "Angle", "value": None, "type": "number"},
        # New Configs R4
        "legend_orient": {"visible": False, "label": "Legend Dir", "value": None, "type": "select", "options": LEGEND_ORIENTS},
        "legend_pos":    {"visible": False, "label": "Legend Pos", "value": None, "type": "select", "options": LEGEND_POSITIONS},
        "barmode":       {"visible": False, "label": "Bar Mode", "value": None, "type": "select", "options": BAR_MODES},
        "log_x":         {"visible": False, "label": "X Scale", "value": None, "type": "select", "options": AXIS_TYPES},
        "log_y":         {"visible": False, "label": "Y Scale", "value": None, "type": "select", "options": AXIS_TYPES},
        "spikes":        {"visible": False, "label": "Spikes", "value": None, "type": "select", "options": SHOW_HIDE},
        "zeroline":      {"visible": False, "label": "Zero Line", "value": None, "type": "select", "options": SHOW_HIDE},
        "global_font_size": {"visible": False, "label": "Font Size", "value": None, "type": "number"},
    }

    header_text = "Properties"

    if selected_element == "figure":
        layout = fig.layout
        header_text = "Figure Settings"
        config["name"].update({"visible": True, "label": "Title", "value": layout.title.text if layout.title else ""})
        config["width"].update({"visible": True, "label": "Width", "value": layout.width})
        config["size"].update({"visible": True, "label": "Height", "value": layout.height})
        config["color"].update({"visible": True, "label": "Plot Color", "value": layout.plot_bgcolor})
        config["paper_color"].update({"visible": True, "value": layout.paper_bgcolor})
        config["font"].update({"visible": True, "value": layout.font.family if layout.font else None})
        config["global_font_size"].update({"visible": True, "value": layout.font.size if layout.font else None})
        config["template"].update({"visible": True, "value": layout.template.layout.template if hasattr(layout, 'template') and layout.template else None})
        config["xaxis"].update({"visible": True, "value": layout.xaxis.title.text if layout.xaxis and layout.xaxis.title else ""})
        config["yaxis"].update({"visible": True, "value": layout.yaxis.title.text if layout.yaxis and layout.yaxis.title else ""})
        config["legend"].update({"visible": True, "value": "Show" if layout.showlegend is not False else "Hide"})
        config["hovermode"].update({"visible": True, "value": layout.hovermode})
        config["grid_x"].update({"visible": True, "value": "Show" if getattr(layout.xaxis, 'showgrid', True) else "Hide"})
        config["grid_y"].update({"visible": True, "value": "Show" if getattr(layout.yaxis, 'showgrid', True) else "Hide"})
        config["barmode"].update({"visible": True, "value": layout.barmode})
        config["log_x"].update({"visible": True, "value": layout.xaxis.type if layout.xaxis else 'linear'})
        config["log_y"].update({"visible": True, "value": layout.yaxis.type if layout.yaxis else 'linear'})
        config["spikes"].update({"visible": True, "value": "Show" if getattr(layout.xaxis, 'showspikes', False) else "Hide"})
        config["zeroline"].update({"visible": True, "value": "Show" if getattr(layout.xaxis, 'zeroline', True) else "Hide"})
        
        # Legend settings
        if layout.legend:
            config["legend_orient"].update({"visible": True, "value": layout.legend.orientation})
            # Infer position (rough approximation)
            # This is tricky because x/y can be anything. We just leave it blank if it doesn't match exactly, or let user overwrite.
            config["legend_pos"].update({"visible": True})

    elif selected_element.startswith("trace_"):
        idx = int(selected_element.split("_")[1])
        if idx < len(fig.data):
            trace = fig.data[idx]
            header_text = f"Trace {idx} Properties"
            config["name"].update({"visible": True, "value": trace.name})
            config["opacity"].update({"visible": True, "value": trace.opacity})
            config["mode"].update({"visible": True, "value": trace.mode})
            config["fill"].update({"visible": True, "value": trace.fill})
            config["text_pos"].update({"visible": True, "value": trace.textposition})
            
            # Color
            color = None
            if hasattr(trace, 'marker') and trace.marker: color = trace.marker.color
            if not color and hasattr(trace, 'line') and trace.line: color = trace.line.color
            if isinstance(color, (list, np.ndarray)): color = None 
            config["color"].update({"visible": True, "value": color})
            
            # Marker Line Color
            mlc = None
            if hasattr(trace, 'marker') and trace.marker and trace.marker.line: mlc = trace.marker.line.color
            config["marker_line_color"].update({"visible": True, "value": mlc})
            
            # Size
            if hasattr(trace, 'marker'):
                size = trace.marker.size if trace.marker else None
                if isinstance(size, (list, np.ndarray)): size = None
                config["size"].update({"visible": True, "value": size})
                
            # Width
            if hasattr(trace, 'line'):
                width = trace.line.width if trace.line else None
                config["width"].update({"visible": True, "label": "Line Width", "value": width})
                
            # Symbol
            if hasattr(trace, 'marker'):
                symbol = trace.marker.symbol if trace.marker else None
                if isinstance(symbol, (list, np.ndarray)): symbol = None
                config["symbol"].update({"visible": True, "value": symbol})
                
            # Dash
            if hasattr(trace, 'line'):
                dash_style = trace.line.dash if trace.line else None
                config["dash"].update({"visible": True, "value": dash_style})
                
            # Line Shape
            if hasattr(trace, 'line'):
                shape = trace.line.shape if trace.line else None
                config["line_shape"].update({"visible": True, "value": shape})

    elif selected_element.startswith("annot_"):
        idx = int(selected_element.split("_")[1])
        if fig.layout.annotations and idx < len(fig.layout.annotations):
            annot = fig.layout.annotations[idx]
            header_text = f"Annotation {idx}"
            config["text"].update({"visible": True, "value": annot.text})
            config["color"].update({"visible": True, "value": annot.font.color if annot.font else None})
            config["size"].update({"visible": True, "value": annot.font.size if annot.font else None})
            config["font"].update({"visible": True, "value": annot.font.family if annot.font else None})
            config["x"].update({"visible": True, "value": annot.x})
            config["y"].update({"visible": True, "value": annot.y})
            config["arrow"].update({"visible": True, "value": "Show" if annot.showarrow else "Hide"})
            config["bgcolor"].update({"visible": True, "value": annot.bgcolor})
            config["text_angle"].update({"visible": True, "value": annot.textangle})

    elif selected_element.startswith("shape_"):
        idx = int(selected_element.split("_")[1])
        if fig.layout.shapes and idx < len(fig.layout.shapes):
            shape = fig.layout.shapes[idx]
            header_text = f"Shape {idx} ({shape.type})"
            config["color"].update({"visible": True, "label": "Line Color", "value": shape.line.color if shape.line else None})
            config["width"].update({"visible": True, "label": "Line Width", "value": shape.line.width if shape.line else None})
            config["opacity"].update({"visible": True, "value": shape.opacity})
            config["dash"].update({"visible": True, "value": shape.line.dash if shape.line else None})
            config["bgcolor"].update({"visible": True, "label": "Fill Color", "value": shape.fillcolor})

    # Build controls
    controls.append(html.H6(header_text, className="border-bottom pb-1 mb-3"))
    
    # Order of controls
    keys_order = [
        "template", "name", "xaxis", "yaxis", "legend", "legend_orient", "legend_pos", # Figure General
        "hovermode", "grid_x", "grid_y", "spikes", "zeroline", "log_x", "log_y", "barmode", # Figure Axes/Layout
        "text", "mode", "line_shape", "fill", "text_pos", "text_angle", # Trace/Annot
        "color", "paper_color", "bgcolor", "marker_line_color", # Colors
        "opacity", "size", "width", "global_font_size", # Numerics
        "symbol", "dash", "font", "arrow", # Styles
        "x", "y" # Coords
    ]
    
    for key in keys_order:
        cfg = config[key]
        controls.append(make_row(cfg["label"], key, input_type=cfg["type"], value=cfg["value"], visible=cfg["visible"], options=cfg.get("options")))

    # Add Apply & Delete Buttons
    controls.append(html.Hr())
    controls.append(dbc.Row([
        dbc.Col(dbc.Button("Apply Changes", id="btn-apply-props", color="primary", size="sm", className="w-100"), width=8),
        dbc.Col(dbc.Button("🗑️", id="btn-delete-element", color="danger", size="sm", className="w-100", title="Delete Element"), width=4),
    ], className="g-1"))
    
    return controls

# ====================================================================
# 13. Callbacks: Drawing Tools & Annotations
# ====================================================================

@app.callback(
    Output("main-graph", "figure", allow_duplicate=True),
    Input("btn-draw-line", "n_clicks"),
    Input("btn-draw-rect", "n_clicks"),
    Input("btn-draw-circle", "n_clicks"),
    Input("btn-draw-free", "n_clicks"),
    Input("btn-draw-poly", "n_clicks"),
    State("main-graph", "figure"),
    prevent_initial_call=True
)
def set_shape_draw_mode(n_line, n_rect, n_circle, n_free, n_poly, fig_dict):
    if not fig_dict: raise PreventUpdate
    ctx_id = ctx.triggered_id
    
    fig_dict = clean_figure_dict(fig_dict)
    fig = go.Figure(fig_dict)
    
    # Set dragmode to enable drawing
    if ctx_id == "btn-draw-line":
        fig.update_layout(dragmode="drawline")
    elif ctx_id == "btn-draw-rect":
        fig.update_layout(dragmode="drawrect")
    elif ctx_id == "btn-draw-circle":
        fig.update_layout(dragmode="drawcircle")
    elif ctx_id == "btn-draw-free":
        fig.update_layout(dragmode="drawopenpath")
    elif ctx_id == "btn-draw-poly":
        fig.update_layout(dragmode="drawclosedpath")
        
    # Ensure newshape properties are set for visibility
    fig.update_layout(newshape=dict(line=dict(color="black", width=2), opacity=1))
        
    figure_store.update_figure(fig)
    return fig

# Sync drawn shapes from graph to store AND update graph to reflect changes immediately
@app.callback(
    Output("figure-store-client", "data", allow_duplicate=True),
    Output("main-graph", "figure", allow_duplicate=True),
    Input("main-graph", "relayoutData"),
    State("main-graph", "figure"),
    prevent_initial_call=True
)
def sync_drawn_shapes(relayout_data, fig_dict):
    if not relayout_data or not fig_dict:
        raise PreventUpdate
        
    # Check if shapes were added/modified
    if 'shapes' in relayout_data:
        # This happens when a shape is drawn or modified
        fig_dict = clean_figure_dict(fig_dict)
        fig = go.Figure(fig_dict)
        fig.layout.shapes = relayout_data['shapes']
        
        # Update store
        figure_store.update_figure(fig)
        
        # Return updated figure to graph to ensure consistency and trigger other callbacks
        return fig.to_dict(), fig
        
    return dash.no_update, dash.no_update

@app.callback(
    Output("modal-annotation", "is_open"),
    Input("btn-add-text", "n_clicks"),
    Input("btn-confirm-annot", "n_clicks"),
    State("modal-annotation", "is_open"),
    prevent_initial_call=True
)
def toggle_annot_modal(n1, n2, is_open):
    if n1 or n2: return not is_open
    return is_open

@app.callback(
    Output("main-graph", "figure", allow_duplicate=True),
    Input("btn-confirm-annot", "n_clicks"),
    State("annot-text", "value"),
    State("annot-x", "value"),
    State("annot-y", "value"),
    State("annot-arrow", "value"),
    State("main-graph", "figure"),
    prevent_initial_call=True
)
def add_text_annotation(n_clicks, text, x, y, show_arrow, fig_dict):
    if not n_clicks or not text: raise PreventUpdate
    
    fig_dict = clean_figure_dict(fig_dict)
    fig = go.Figure(fig_dict)
    
    # Push to history
    history_stack.push(clean_figure_dict(fig.to_dict()))
    
    # Default to center if no coords
    xref, yref = "paper", "paper"
    x_val, y_val = (x, y) if (x is not None and y is not None) else (0.5, 0.5)
    
    fig.add_annotation(
        text=text,
        x=x_val, y=y_val,
        xref=xref, yref=yref,
        showarrow=bool(show_arrow),
        arrowhead=2 if show_arrow else 0,
        font=dict(size=14, color="black"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    )
    
    figure_store.update_figure(fig)
    return fig

# ====================================================================
# 13b. Callbacks: Media Tools (Image Upload)
# ====================================================================

@app.callback(
    Output("main-graph", "figure", allow_duplicate=True),
    Input("upload-image", "contents"),
    State("main-graph", "figure"),
    prevent_initial_call=True
)
def add_background_image(content, fig_dict):
    if not content:
        raise PreventUpdate
        
    fig_dict = clean_figure_dict(fig_dict)
    fig = go.Figure(fig_dict)
    
    # Add image to layout
    # content is already "data:image/png;base64,..."
    
    new_image = dict(
        source=content,
        xref="paper", yref="paper",
        x=0, y=1,
        sizex=1, sizey=1,
        sizing="stretch",
        opacity=0.5,
        layer="below"
    )
    
    fig.add_layout_image(new_image)
    
    figure_store.update_figure(fig)
    return fig

# ====================================================================
# 14. Callbacks: History & Session
# ====================================================================

@app.callback(
    Output("main-graph", "figure", allow_duplicate=True),
    Output("btn-undo", "disabled"),
    Output("btn-redo", "disabled"),
    Input("btn-undo", "n_clicks"),
    Input("btn-redo", "n_clicks"),
    Input("main-graph", "figure"), # Listen to graph updates to update button state
    State("main-graph", "figure"),
    prevent_initial_call=True
)
def manage_history(n_undo, n_redo, fig_trigger, current_fig_dict):
    ctx_id = ctx.triggered_id
    
    # If triggered by graph update, just update buttons
    if ctx_id == "main-graph":
        return dash.no_update, not history_stack.can_undo(), not history_stack.can_redo()
        
    # Handle Undo/Redo
    new_fig = None
    if ctx_id == "btn-undo":
        new_fig = history_stack.undo()
    elif ctx_id == "btn-redo":
        new_fig = history_stack.redo()
        
    if new_fig:
        new_fig = clean_figure_dict(new_fig)
        fig = go.Figure(new_fig)
        figure_store.update_figure(fig)
        return fig, not history_stack.can_undo(), not history_stack.can_redo()
        
    return dash.no_update, not history_stack.can_undo(), not history_stack.can_redo()

@app.callback(
    Output("download-component", "data"),
    Input("btn-save-session", "n_clicks"),
    State("main-graph", "figure"),
    prevent_initial_call=True
)
def save_session(n_clicks, fig_dict):
    if not fig_dict: raise PreventUpdate
    fig_dict = clean_figure_dict(fig_dict)
    return dict(content=json.dumps(fig_dict, indent=2), filename="session.json")

@app.callback(
    Output("main-graph", "figure", allow_duplicate=True),
    Input("upload-session", "contents"),
    prevent_initial_call=True
)
def load_session(content):
    if not content: raise PreventUpdate
    try:
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        fig_dict = json.loads(decoded.decode('utf-8'))
        fig_dict = clean_figure_dict(fig_dict)
        fig = go.Figure(fig_dict)
        figure_store.update_figure(fig)
        history_stack.push(fig_dict)
        return fig
    except Exception as e:
        print(f"Error loading session: {e}")
        raise PreventUpdate

# ====================================================================
# 15. Callbacks: View Tools
# ====================================================================

@app.callback(
    Output("main-graph", "figure", allow_duplicate=True),
    Input("btn-tool-zoom", "n_clicks"),
    Input("btn-tool-pan", "n_clicks"),
    Input("btn-tool-reset", "n_clicks"),
    State("main-graph", "figure"),
    prevent_initial_call=True
)
def view_tools(n_zoom, n_pan, n_reset, fig_dict):
    ctx_id = ctx.triggered_id
    if not fig_dict: raise PreventUpdate
    
    fig_dict = clean_figure_dict(fig_dict)
    fig = go.Figure(fig_dict)
    
    if ctx_id == "btn-tool-zoom":
        fig.update_layout(dragmode="zoom")
    elif ctx_id == "btn-tool-pan":
        fig.update_layout(dragmode="pan")
    elif ctx_id == "btn-tool-reset":
        fig.update_xaxes(autorange=True)
        fig.update_yaxes(autorange=True)
        if 'scene' in fig.layout:
            fig.update_scenes(xaxis_autorange=True, yaxis_autorange=True, zaxis_autorange=True)
            
    figure_store.update_figure(fig)
    return fig

# ====================================================================
# 15b. Callbacks: Real-time Statistics
# ====================================================================

@app.callback(
    Output("console-output", "children", allow_duplicate=True),
    Input("main-graph", "selectedData"),
    State("console-output", "children"),
    prevent_initial_call=True
)
def show_selection_stats(selected_data, current_console):
    if not selected_data:
        raise PreventUpdate
    
    points = selected_data.get("points", [])
    if not points:
        return dash.no_update
        
    msg = f">>> Selected {len(points)} points."
    
    # Calculate simple stats from the selection data directly
    ys = [p.get('y') for p in points if 'y' in p and isinstance(p.get('y'), (int, float))]
    if ys:
        mean_y = sum(ys) / len(ys)
        min_y = min(ys)
        max_y = max(ys)
        msg += f" | Y-Stats: Mean={mean_y:.2f}, Min={min_y:.2f}, Max={max_y:.2f}"
        
    # Keep console history reasonable
    if len(current_console) > 1000:
        current_console = current_console[-500:]
        
    return f"{current_console}\n{msg}"

# ====================================================================
# 16. Launch Application
# ====================================================================

if __name__ == '__main__':
    print("\n" + "="*72)
    print("🚀 Python Interactive Figure Editor - Starting...")
    print("="*72)
    print("📍 URL: http://localhost:8051")
    print("💡 Tip: Use Ctrl+Click to open in browser")
    print("⚡ Feature Highlights:")
    print("   - Dash-powered canvas with MATLAB-style figure editing")
    print("   - Drawing tools (line/rect/circle/freehand) + undo/redo stack")
    print("   - Trace styling, theme presets, and live property inspector")
    print("   - Lasso statistics & outlier removal from datasets")
    print("   - Hybrid canvas: overlay images with adjustable opacity")
    print("   - Layer manager with visibility toggles and summaries")
    print("   - Code generator + session export/restore + PNG output")
    print("="*72 + "\n")
    app.run(debug=True, jupyter_mode='inline', port=8051)
