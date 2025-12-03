# PyFigureEditor: A Python-Based Interactive Visualization GUI with MATLAB-Style Editing Capabilities

<div align="center">

## MATH 4710 - Final Project Report

---

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Dash](https://img.shields.io/badge/Dash-2.17+-00BC8C?style=for-the-badge&logo=plotly&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.20+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.5+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.22+-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production-success?style=for-the-badge)
![Deployed](https://img.shields.io/badge/Live_Demo-Available-brightgreen?style=for-the-badge)

---

> MATH 4710 – Data Visualization
> Kean University · Department of Mathematics Science
> Instructor: Dr. Hamza Puneet Rana 


---

### 🌐 Live Production Deployment

## **[https://zye.pythonanywhere.com/](https://zye.pythonanywhere.com/)**

*Use the link above to experience the fully functional application without installation, **and the one below for the Colab experiment version**.*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1235357/PyFigureEditor/blob/main/Final_Project_Implementation.ipynb)


## 💻 Live Learning in Chinese [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1235357/PyFigureEditor/blob/main/LEARNING_Project_Implementation.ipynb)

</div>

---

# 🎯 Quick Guide (Complete Code Architecture Analysis)


## 📋 Quick Navigation: Guide ↔ Report Mapping

| Quick Guide Section | Click to Jump to Detailed Report |
|---------------------|----------------------------------|
| [Part 1: Project Overview](#-part-1-project-overview-what--why) | → [Section 1: Introduction (Full Details)](#1-introduction) |
| [Part 2: Code Architecture](#-part-2-code-architecture-overview) | → [Section 3: System Architecture](#3-system-architecture-and-design) |
| [Part 3: Four Core Classes](#-part-3-four-core-classes-explained) | → [Section 3.2: Core Class Design](#32-core-class-design) |
| [Part 4: Callback System](#-part-4-callback-system-deep-dive) | → [Section 4.3: Callback Implementation](#43-callback-implementation---the-brain-of-the-application) |
| [Part 5: UI Layout](#-part-5-ui-layout-architecture) | → [Section 4.2: Layout Implementation](#42-layout-implementation---building-the-user-interface) |
| [Part 6: Data Flow Example](#-part-6-complete-data-flow-example) | → [Section 3.5: Data Flow Architecture](#35-data-flow-architecture) |
| [Part 7: Technical Points](#-part-7-key-technical-points) | → [Section 2.4: Reactive Programming](#24-reactive-programming-paradigm) |
| [Part 8: Demo Workflow](#-part-8-suggested-demo-workflow) | → [Section 5: Feature Documentation](#5-complete-feature-documentation---your-user-manual) |
| [Part 9: Q&A](#-part-9-common-questions-qa) | → [Section 2: Literature Review](#2-literature-review-and-technical-foundation) |
| [Part 10: Summary](#-part-10-technical-achievement-summary) | → [Section 1.4: Scope and Deliverables](#14-scope-and-deliverables) |

---

<img width="3839" height="1944" alt="image" src="https://github.com/user-attachments/assets/b61cf9d3-1e66-4f95-aabd-dbf28150d110" />


## 📌 Part 1: Project Overview (What & Why)

> 📖 **Want more details?** See [Section 1: Introduction](#1-introduction) in the Full Report below.

### 🤔 What Problem Does This Project Solve?

**The Pain Point of Traditional Python Plotting:**

```python
# Traditional way: Every modification requires code changes and re-running
import matplotlib.pyplot as plt
plt.plot([1,2,3], [4,5,6], color='blue')  # Want to change to red?
plt.show()

# To change the color, you must:
# 1. Go back to the code
# 2. Change color='red'  
# 3. Re-run the entire script
# 4. View result... not satisfied? Repeat the above steps 😫
```

**My Solution: PyFigureEditor**

```
No code changes needed! Click on chart → Select element → Modify properties → Instant effect!
As simple as editing images in PowerPoint ✨
```

### 🎯 Core Features Overview

| Feature | User Action | Technical Implementation Behind | 📖 Learn More |
|---------|-------------|--------------------------------|---------------|
| **Create Charts** | Select chart type → Click button | `px.scatter()`, `go.Figure()` generates Plotly chart | [5.1 Plot Types](#51-plot-types-and-chart-creation---your-26-chart-arsenal) |
| **Edit Properties** | Select element → Change color/size → Apply | Callback listens → Modifies `fig.data[i]` → Returns new chart | [5.2 Property Editing](#52-property-editing-system---fine-tune-everything) |
| **Draw Shapes** | Click "Draw Rectangle" → Drag on chart | `fig.update_layout(dragmode="drawrect")` | [5.3 Drawing Tools](#53-drawing-and-annotation-tools---make-your-charts-talk) |
| **Undo/Redo** | Click Undo/Redo | `HistoryStack` class stores history states | [5.7 Undo/Redo](#57-undoredo-system---never-lose-your-work) |
| **Save Session** | Click Save → Download JSON | `fig.to_dict()` serializes to JSON | [5.5 Session Management](#55-session-management---save-and-share-your-work) |
| **Generate Code** | Auto-generate Python code | `CodeGenerator` class reverse-generates code | [5.6 Code Export](#56-code-export---take-your-work-anywhere) |

---

## 📌 Part 2: Code Architecture Overview

> 📖 **Want more details?** See [Section 3: System Architecture and Design](#3-system-architecture-and-design) for complete architecture diagrams and design decisions.

### 🗂️ File Structure

```
Final Project/
├── app.py                              # Main application (2696 lines)
├── Final_Project_Implementation.ipynb  # Jupyter version (same logic)
└── README.md                           # This document
```

### 🏗️ `app.py` Code Organization (By Line Numbers)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  SECTION 0: Auto Dependency Installation (Lines 1-155)                  │
│  ├── AUTO_DEPENDENCY_MAP: Defines required packages                     │
│  └── _auto_install_dependencies(): Auto pip install                     │
├─────────────────────────────────────────────────────────────────────────┤
│  SECTION 1-2: Library Imports & Core Data Model (Lines 156-270)         │
│  ├── TraceDataset: Data container for a single chart layer              │
│  └── Initialize Dash App                                                │
├─────────────────────────────────────────────────────────────────────────┤
│  SECTION 3: FigureStore State Management (Lines 271-565)                │
│  └── Core class: Stores current chart, datasets, metadata               │
├─────────────────────────────────────────────────────────────────────────┤
│  SECTION 4: History & Logs (Lines 566-640)                              │
│  ├── HistoryStack: Undo/Redo implementation                             │
│  └── ActionLog: User action logging                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  SECTION 5: CodeGenerator (Lines 641-817)                               │
│  └── Generates Python code from current chart state                     │
├─────────────────────────────────────────────────────────────────────────┤
│  SECTION 6: Singleton Initialization (Lines 818-845)                    │
│  └── figure_store, history_stack, code_generator global instances       │
├─────────────────────────────────────────────────────────────────────────┤
│  SECTION 7-8: UI Components (Lines 846-1187)                            │
│  ├── ribbon: Top toolbar (HOME/DATA/PLOTS/ANNOTATE/VIEW tabs)           │
│  ├── workspace_panel: Left panel (code editor + data table)             │
│  ├── property_inspector: Right property inspector                       │
│  └── Modals: Pop-ups (add annotation, about)                            │
├─────────────────────────────────────────────────────────────────────────┤
│  SECTION 9-15: Callbacks (Lines 1188-2676)                              │
│  ├── 9: UI Interaction (tab switching)                                  │
│  ├── 10a: Data Management (CSV upload, delete, clean)                   │
│  ├── 10b: Data Interaction (select points, delete points)               │
│  ├── 11: Code Generation & Execution                                    │
│  ├── 12: Property Editor (CORE!)                                        │
│  ├── 13: Drawing Tools & Annotations                                    │
│  ├── 14: History & Session Management                                   │
│  └── 15: View Tools (Zoom/Pan/Reset)                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  SECTION 16: Launch Application (Lines 2677-2696)                       │
│  └── app.run(debug=True, port=8051)                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📌 Part 3: Four Core Classes Explained

> 📖 **Want more details?** See [Section 3.2: Core Class Design](#32-core-class-design) for complete class diagrams, all methods, and design rationale.

### 🔷 Class 1: `TraceDataset` (Data Container)

**Location:** Lines 210-266  
**Purpose:** Encapsulates all information for a single chart layer (trace)

```python
@dataclass
class TraceDataset:
    """Data container for one chart layer"""
    key: str              # Unique identifier, e.g., "trace_1"
    name: str             # Display name, e.g., "Demo Signal"
    df: pd.DataFrame      # Actual data (x, y, z columns)
    color: str = "#1f77b4"      # Color
    line_width: float = 2.5     # Line width
    marker_size: float = 6.0    # Marker size
    visible: bool = True        # Visibility
    chart_type: str = "scatter" # Chart type

    def to_plotly_trace(self):
        """Convert to Plotly trace object"""
        if self.chart_type == "bar":
            trace = go.Bar(x=self.df['x'], y=self.df['y'], name=self.name)
        elif self.chart_type == "scatter":
            trace = go.Scatter(x=self.df['x'], y=self.df['y'], 
                              mode="lines+markers", name=self.name)
        # ... other types
        
        # Apply styling
        trace.update(marker=dict(size=self.marker_size, color=self.color))
        return trace
```

**Why Do We Need This Class?**
- Binds "data" and "styling" together
- Easy to serialize for save/load
- Can generate multiple chart types from one dataset

---

### 🔷 Class 2: `FigureStore` (Core State Management)

**Location:** Lines 271-565  
**Purpose:** The "brain" of the entire application, manages all state

```python
class FigureStore:
    """Manages current Plotly chart and all datasets"""
    
    def __init__(self, theme: str = "plotly_white"):
        self.current_theme: str = theme          # Current theme
        self.figure: go.Figure = None            # Current chart object ⭐
        self.datasets: Dict[str, TraceDataset] = {}  # All datasets
        self.dataset_order: List[str] = []       # Dataset order
        self.data_repository: Dict[str, pd.DataFrame] = {}  # Raw data warehouse
        self.metadata: Dict = {...}              # Metadata (creation time, etc.)
```

**Key Methods Explained:**

```python
# 1. Add Dataset
def add_dataset(self, key, name, df, color, ...):
    dataset = TraceDataset(key=key, name=name, df=df, ...)
    self.datasets[key] = dataset
    self.dataset_order.append(key)

# 2. Rebuild Figure from Datasets
def rebuild_figure_from_datasets(self):
    fig = go.Figure()
    for key in self.dataset_order:
        dataset = self.datasets[key]
        fig.add_trace(dataset.to_plotly_trace())  # Add each trace
    fig.update_layout(**self._base_layout())
    self.figure = fig

# 3. Serialize Session (Save as JSON)
def serialize_session(self) -> Dict:
    return {
        "metadata": self.metadata,
        "datasets": {k: ds.to_dict() for k, ds in self.datasets.items()},
        "figure": self.figure.to_dict(),
        "version": "1.0.0"
    }

# 4. Load Session
def load_session(self, payload: Dict):
    self.datasets.clear()
    for key, item in payload["datasets"].items():
        df = pd.DataFrame(item["df"])
        self.add_dataset(key=key, df=df, ...)
    self.figure = go.Figure(payload["figure"])
```

**State Flow Diagram:**
```
User Action → Callback Invoked → FigureStore Method → Update self.figure → Return to UI
```

---

### 🔷 Class 3: `HistoryStack` (Undo/Redo)

**Location:** Lines 570-636  
**Purpose:** Classic Undo/Redo stack implementation

```python
class HistoryStack:
    """Undo/Redo stack for chart states"""
    
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.undo_stack: List[Dict] = []  # History states
        self.redo_stack: List[Dict] = []  # Undone states

    def push(self, fig_dict: Dict):
        """Save current state to history"""
        snapshot = copy.deepcopy(fig_dict)
        self.undo_stack.append(snapshot)
        if len(self.undo_stack) > self.max_size:
            self.undo_stack.pop(0)  # Remove oldest
        self.redo_stack.clear()     # New action clears redo

    def undo(self) -> Optional[Dict]:
        """Undo: Go back to previous state"""
        if len(self.undo_stack) <= 1:
            return None
        current = self.undo_stack.pop()      # Pop current
        self.redo_stack.append(current)       # Store in redo
        return self.undo_stack[-1]            # Return previous state

    def redo(self) -> Optional[Dict]:
        """Redo: Restore undone state"""
        if not self.redo_stack:
            return None
        state = self.redo_stack.pop()
        self.undo_stack.append(state)
        return state
```

**Visual Understanding:**
```
Action sequence: A → B → C → D
                           ↑ Current
Undo Stack: [A, B, C, D]
Redo Stack: []

--- User clicks Undo ---
Undo Stack: [A, B, C]  ← Back to C
Redo Stack: [D]

--- User clicks Redo ---
Undo Stack: [A, B, C, D]  ← Restored to D
Redo Stack: []
```

---

### 🔷 Class 4: `CodeGenerator` (Code Generator)

**Location:** Lines 641-817  
**Purpose:** Reverse-generate Python code from current chart state

```python
class CodeGenerator:
    """Convert current chart to runnable Python code"""

    def generate_code(self, store: FigureStore) -> str:
        """Generate complete Python code"""
        fig_json = store.figure.to_json()
        
        code = [
            "# Auto-generated by PyFigureEditor",
            "import json",
            "import plotly.graph_objects as go",
            "",
            f"fig_dict = json.loads({fig_json!r})",
            "fig = go.Figure(fig_dict)",
            "fig.show()"
        ]
        return "\n".join(code)

    def generate_smart_plot_code(self, df_name, plot_type, df):
        """Smart code generation: Auto-infer column names"""
        
        # 1. Analyze column data types
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        
        # 2. Smart column selection based on chart type
        if plot_type == 'scatter':
            x_col, y_col = num_cols[0], num_cols[1]
            return f"fig = px.scatter({df_name}, x='{x_col}', y='{y_col}')"
            
        elif plot_type == 'bar':
            x_col = cat_cols[0] if cat_cols else num_cols[0]
            y_col = num_cols[0]
            return f"fig = px.bar({df_name}, x='{x_col}', y='{y_col}')"
        # ... 26+ chart types
```

---

## 📌 Part 4: Callback System Deep Dive

> 📖 **Want more details?** See [Section 4.3: Callback Implementation](#43-callback-implementation---the-brain-of-the-application) for complete callback code and [Section 2.4: Reactive Programming](#24-reactive-programming-paradigm) for the theory.

### 🔄 What is a Callback? (Core Mechanism)

**Callback = Reactive Programming**

```
Traditional Programming:     Reactive Programming (Dash):
while True:                  @app.callback(...)
  if button_clicked:         def handle_click():
    do_something()               do_something()
  sleep(0.1)                 # Auto-triggered, no polling needed!
```

**Three Components of a Dash Callback:**

```python
@app.callback(
    Output('main-graph', 'figure'),     # 1. OUTPUT: What to update
    Input('btn-apply', 'n_clicks'),     # 2. INPUT: What triggers this function
    State('dd-color', 'value')          # 3. STATE: Data to read but not trigger
)
def update_graph(n_clicks, color):
    # 4. FUNCTION: The actual logic
    fig = go.Figure(...)
    return fig  # Return value automatically updates Output
```

**Input vs State Difference:**
| | Input | State |
|---|---|---|
| Triggers Callback on change | ✅ Yes | ❌ No |
| Can read current value | ✅ Yes | ✅ Yes |
| Use case | Trigger (button click) | Additional data (dropdown current value) |

---

### 🔧 Key Callback Analysis

#### Callback 1: Property Editor (Most Complex Callback)

**Location:** Lines 2006-2090  
**Function:** When user clicks "Apply", update the selected element's properties

```python
@app.callback(
    Output("main-graph", "figure", allow_duplicate=True),
    Input("btn-apply-props", "n_clicks"),           # Trigger: Apply button
    State("dd-element-select", "value"),            # Selected element
    State("input-prop-name", "value"),              # Name input
    State("input-prop-color", "value"),             # Color picker
    State("input-prop-size", "value"),              # Size input
    # ... 35+ property States
    State("main-graph", "figure"),                  # Current chart
    prevent_initial_call=True
)
def apply_property_changes(n_clicks, selected_element, name, color, size, ...):
    # 1. Get current chart
    fig = go.Figure(fig_dict)
    
    # 2. Update different properties based on selected element type
    if selected_element == "figure":
        # Update entire chart properties
        fig.update_layout(title=name, plot_bgcolor=color, ...)
        
    elif selected_element.startswith("trace_"):
        # Update a specific trace's properties
        idx = int(selected_element.split("_")[1])  # "trace_0" → 0
        trace = fig.data[idx]
        trace.update(marker=dict(color=color, size=size))
        
    elif selected_element.startswith("annot_"):
        # Update annotation
        idx = int(selected_element.split("_")[1])
        fig.layout.annotations[idx].update(text=name, font=dict(color=color))
        
    elif selected_element.startswith("shape_"):
        # Update shape
        idx = int(selected_element.split("_")[1])
        fig.layout.shapes[idx].update(line=dict(color=color, width=width))
    
    # 3. Save and return
    figure_store.update_figure(fig)
    return fig
```

**Data Flow Diagram:**
```
User changes color → Clicks Apply → Callback triggered
                                        ↓
                              Read all State values
                                        ↓
                              Determine selected element type
                                        ↓
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
               trace_0              figure              annot_0
                    ↓                   ↓                   ↓
            fig.data[0]       fig.update_layout    fig.layout.annotations[0]
              .update()                                 .update()
                    └───────────────────┼───────────────────┘
                                        ▼
                              return fig → Update UI
```

---

#### Callback 2: Dynamic Property Panel Generation

**Location:** Lines 1767-2005  
**Function:** Display different property editing options based on selected element type

```python
@app.callback(
    Output("inspector-controls", "children"),    # UI container to update
    Input("dd-element-select", "value"),         # Trigger: Dropdown selection change
    State("main-graph", "figure"),               # Current chart
)
def update_inspector_controls(selected_element, fig_dict):
    """Dynamically generate property editing panel"""
    
    fig = go.Figure(fig_dict)
    controls = []  # List of UI components to display
    
    if selected_element == "figure":
        # Show chart-level properties
        controls.append(make_input("Title", fig.layout.title.text))
        controls.append(make_input("Width", fig.layout.width))
        controls.append(make_input("Height", fig.layout.height))
        controls.append(make_dropdown("Theme", ["plotly", "plotly_dark", ...]))
        controls.append(make_input("X Axis Title", ...))
        controls.append(make_input("Y Axis Title", ...))
        
    elif selected_element.startswith("trace_"):
        # Show trace-level properties
        idx = int(selected_element.split("_")[1])
        trace = fig.data[idx]
        controls.append(make_input("Name", trace.name))
        controls.append(make_color_picker("Color", trace.marker.color))
        controls.append(make_input("Size", trace.marker.size))
        controls.append(make_dropdown("Symbol", ["circle", "square", ...]))
        controls.append(make_dropdown("Line Style", ["solid", "dash", ...]))
        
    elif selected_element.startswith("annot_"):
        # Show annotation properties
        controls.append(make_input("Text", ...))
        controls.append(make_input("X Position", ...))
        controls.append(make_input("Y Position", ...))
        controls.append(make_checkbox("Show Arrow", ...))
    
    # Add Apply and Delete buttons
    controls.append(make_button("Apply Changes"))
    controls.append(make_button("Delete Element"))
    
    return controls  # Return dynamically generated UI
```

**Why This Matters:**
- **Traditional way:** Show all 100+ properties, users can't find what they need
- **My approach:** Only show 10-20 properties relevant to selected element, clear and concise!

---

#### Callback 3: Drawing Tools Implementation

**Location:** Lines 2379-2455  
**Function:** Allow users to draw shapes directly on the chart

```python
# Part A: Set Drawing Mode
@app.callback(
    Output("main-graph", "figure", allow_duplicate=True),
    Input("btn-draw-line", "n_clicks"),
    Input("btn-draw-rect", "n_clicks"),
    Input("btn-draw-circle", "n_clicks"),
    # ...
)
def set_shape_draw_mode(n_line, n_rect, n_circle, ...):
    ctx_id = ctx.triggered_id  # Which button was clicked
    
    if ctx_id == "btn-draw-line":
        fig.update_layout(dragmode="drawline")      # Plotly built-in mode
    elif ctx_id == "btn-draw-rect":
        fig.update_layout(dragmode="drawrect")
    elif ctx_id == "btn-draw-circle":
        fig.update_layout(dragmode="drawcircle")
    
    return fig

# Part B: Capture Drawn Shapes
@app.callback(
    Output("figure-store-client", "data"),
    Input("main-graph", "relayoutData"),  # Plotly auto-sends drawing data
)
def sync_drawn_shapes(relayout_data, fig_dict):
    if 'shapes' in relayout_data:
        # User just finished drawing, Plotly tells us the coordinates
        fig.layout.shapes = relayout_data['shapes']
        figure_store.update_figure(fig)
    return fig.to_dict()
```

**Drawing Workflow:**
```
1. User clicks "Draw Rectangle" button
   ↓
2. Callback executes: fig.update_layout(dragmode="drawrect")
   ↓
3. Plotly.js enters drawing mode, cursor becomes crosshair
   ↓
4. User drags to draw on the chart
   ↓
5. Plotly.js auto-sends relayoutData = {shapes: [...]}
   ↓
6. Callback captures and saves to FigureStore
```

---

## 📌 Part 5: UI Layout Architecture

> 📖 **Want more details?** See [Section 4.2: Layout Implementation](#42-layout-implementation---building-the-user-interface) for actual layout code and component details.

### 🖥️ Overall Layout (Three-Column Design)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  RIBBON (Top Toolbar)                                                   │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌──────────┐ ┌─────┐                          │
│  │HOME │ │DATA │ │PLOTS│ │ ANNOTATE │ │VIEW │                          │
│  └─────┘ └─────┘ └─────┘ └──────────┘ └─────┘                          │
│  [Open][Save]  [Import CSV] [Scatter][Bar][3D]  [Draw][Text]  [Zoom]   │
├─────────────┬───────────────────────────────────┬───────────────────────┤
│  LEFT PANEL │         CENTER CANVAS             │     RIGHT PANEL       │
│  (width=3)  │           (width=6)               │       (width=3)       │
│             │                                   │                       │
│ ┌─────────┐ │                                   │ ┌───────────────────┐ │
│ │Command  │ │                                   │ │Property Inspector │ │
│ │Window   │ │         ┌─────────────┐           │ ├───────────────────┤ │
│ │         │ │         │             │           │ │ Select Element:   │ │
│ │ # Python│ │         │   PLOTLY    │           │ │ [▼ Trace 0     ]  │ │
│ │ code... │ │         │   GRAPH     │           │ │                   │ │
│ │         │ │         │             │           │ │ Name: [Demo     ] │ │
│ └─────────┘ │         │             │           │ │ Color: [● Blue  ] │ │
│             │         │             │           │ │ Size:  [6       ] │ │
│ ┌─────────┐ │         └─────────────┘           │ │ ...               │ │
│ │Data View│ │                                   │ │                   │ │
│ │ (Table) │ │                                   │ │ [Apply] [Delete]  │ │
│ └─────────┘ │                                   │ └───────────────────┘ │
└─────────────┴───────────────────────────────────┴───────────────────────┘
```

### 📋 Ribbon (Top Tab Bar) Details

**HOME Tab:**
```python
# File Operations + History
[📂 Open Session] [💾 Save Session] [↶ Undo] [↷ Redo] [ℹ️ About]
```

**DATA Tab:**
```python
# Data Management Workflow: Import → Select → View → Clean
[📂 Import CSV] [🎲 Load Demo]  # 1. Data Source
[▼ Select Data] [🗑️ Delete]     # 2. Select Dataset
[📋 Raw Table] [📊 Summary]     # 3. View Data
[🧹 Clean NA] [✂️ Remove Sel]   # 4. Pre-processing
```

**PLOTS Tab:**
```python
# 26+ Chart Types, Grouped Display
Basic 2D:     [Scatter] [Line] [Bar] [Area] [Bubble]
Distribution: [Hist] [Box] [Violin] [Heatmap] [Pie] [Sunburst] [Treemap]
3D & Contour: [Scatter3D] [Line3D] [Surface] [Contour]
Specialized:  [Polar] [Ternary] [Funnel] [Candle] [Waterfall] [ScatMat] [ParCoords]
Maps & Geo:   [ScatGeo] [Choropleth] [Globe]
```

**ANNOTATE Tab:**
```python
# Drawing and Annotation Tools
Shapes:    [📏 Line] [⬜ Rect] [⭕ Circle] [✏️ Free] [⬡ Polygon]
Text:      [📝 Add Annotation]
Media:     [🖼️ Add Image]
```

**VIEW Tab:**
```python
# View Controls
Navigation: [🔍 Zoom] [✋ Pan] [🏠 Reset]
Panels:     [☑️ Inspector Toggle]
```

---

## 📌 Part 6: Complete Data Flow Example

> 📖 **Want more details?** See [Section 3.5: Data Flow Architecture](#35-data-flow-architecture) for complete data flow diagrams.

### 📊 Example: From Creating a Scatter Plot to Changing Color

```
Step 1: User clicks "Scatter" button
        ↓
Step 2: Callback `generate_and_trigger_plot` triggered
        ↓
        Generate code: "fig = px.scatter(df, x='x', y='y')"
        ↓
Step 3: Code auto-executes (Callback `execute_code`)
        ↓
        exec(code) → Create fig object
        ↓
Step 4: figure_store.update_figure(fig)
        ↓
        History recorded: history_stack.push(fig.to_dict())
        ↓
Step 5: Return fig → Update main-graph → User sees the chart!
        ↓
        ════════════════════════════════════════════
        ↓
Step 6: User selects "Trace 0" from dropdown
        ↓
Step 7: Callback `update_inspector_controls` triggered
        ↓
        Read current properties of fig.data[0]
        ↓
        Dynamically generate property panel: [Name][Color][Size][Symbol]...
        ↓
Step 8: User changes Color to "red", clicks Apply
        ↓
Step 9: Callback `apply_property_changes` triggered
        ↓
        fig.data[0].update(marker=dict(color="red"))
        ↓
Step 10: Return fig → Chart color changes to red instantly!
```

---

## 📌 Part 7: Key Technical Points

> 📖 **Want more details?** See [Section 2.4: Reactive Programming](#24-reactive-programming-paradigm) for complete technical explanations.

### 🔑 Technical Point 1: `allow_duplicate=True`

**Problem:** Dash by default doesn't allow multiple Callbacks to update the same Output

**Solution:**
```python
# Callback 1: Edit Properties
@app.callback(
    Output("main-graph", "figure", allow_duplicate=True),  # ← Allow duplicate
    Input("btn-apply-props", "n_clicks"),
    ...
)

# Callback 2: Drawing Tools
@app.callback(
    Output("main-graph", "figure", allow_duplicate=True),  # ← Same Output
    Input("btn-draw-rect", "n_clicks"),
    ...
)

# Callback 3: Undo/Redo
@app.callback(
    Output("main-graph", "figure", allow_duplicate=True),  # ← All can update
    Input("btn-undo", "n_clicks"),
    ...
)
```

### 🔑 Technical Point 2: `ctx.triggered_id` to Identify Trigger Source

**Problem:** A Callback has multiple Inputs, how to know which one triggered it?

```python
@app.callback(
    Output(...),
    Input("btn-zoom", "n_clicks"),
    Input("btn-pan", "n_clicks"),
    Input("btn-reset", "n_clicks"),
)
def view_tools(n_zoom, n_pan, n_reset):
    ctx_id = ctx.triggered_id  # Get the triggered component ID
    
    if ctx_id == "btn-zoom":
        fig.update_layout(dragmode="zoom")
    elif ctx_id == "btn-pan":
        fig.update_layout(dragmode="pan")
    elif ctx_id == "btn-reset":
        fig.update_xaxes(autorange=True)
```

### 🔑 Technical Point 3: Plotly Figure JSON Structure

```python
fig.to_dict()  # Returns:
{
    "data": [
        {
            "type": "scatter",
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "name": "Trace 0",
            "marker": {"color": "blue", "size": 10}
        },
        {...}  # More traces
    ],
    "layout": {
        "title": {"text": "My Chart"},
        "template": "plotly_white",
        "shapes": [...],       # Drawn shapes
        "annotations": [...]   # Text annotations
    }
}
```

**This is why we can:**
- Save session as JSON
- Load session from JSON
- Modify any property

---

## 📌 Part 8: Suggested Demo Workflow

> 📖 **Want more details?** See [Section 5: Complete Feature Documentation](#5-complete-feature-documentation---your-user-manual) for step-by-step tutorials on each feature.

### 🎬 Live Demo Steps (5 Minutes)

| Time | Action | Purpose | 📖 Learn More |
|------|--------|---------|---------------|
| 0:00 | Open [zye.pythonanywhere.com](https://zye.pythonanywhere.com/) | Show successful deployment | [Live Demo](https://zye.pythonanywhere.com/) |
| 0:30 | DATA → Load Demo → Generate demo data | Show data management | [5.4 Data Management](#54-data-management---load-edit-clean) |
| 1:00 | PLOTS → Click Scatter → Auto-generate chart | Show one-click creation | [5.1 Plot Types](#51-plot-types-and-chart-creation---your-26-chart-arsenal) |
| 1:30 | Select Trace 0 → Change color to red → Apply | Show property editing | [5.2 Property Editing](#52-property-editing-system---fine-tune-everything) |
| 2:00 | ANNOTATE → Draw Rect → Draw rectangle on chart | Show drawing tools | [5.3 Drawing Tools](#53-drawing-and-annotation-tools---make-your-charts-talk) |
| 2:30 | Add Annotation → Enter text → Add | Show annotation feature | [5.3.2 Text Annotations](#532-adding-text-annotations---step-by-step) |
| 3:00 | Undo → Redo → Show history feature | Show undo/redo | [5.7 Undo/Redo System](#57-undoredo-system---never-lose-your-work) |
| 3:30 | HOME → Save Session → Download JSON | Show session saving | [5.5 Session Management](#55-session-management---save-and-share-your-work) |
| 4:00 | View Command Window generated code | Show code generation | [5.6 Code Export](#56-code-export---take-your-work-anywhere) |
| 4:30 | Switch to 3D/Map chart types | Show diversity | [5.1.3 3D Charts](#513-3d-charts---when-two-dimensions-arent-enough) |

---

## 📌 Part 9: Common Questions Q&A

> 📖 **Want more details?** See [Section 2: Literature Review](#2-literature-review-and-technical-foundation) for complete library comparisons and technical justifications.

| Question | Answer | 📖 Learn More |
|----------|--------|---------------|
| **Why choose Plotly over Matplotlib?** | Matplotlib is static; Plotly natively supports interaction (zoom, pan, hover), and has `config={'editable': True}` for on-chart editing | [2.1 Visualization Libraries](#21-overview-of-python-visualization-libraries) |
| **Why use Dash instead of Flask + JavaScript?** | Dash is Plotly's official framework with native integration; no JS needed, entire app in Python | [2.3 Web Frameworks](#23-web-based-gui-frameworks-for-python) |
| **Will too many Callbacks be slow?** | No. Dash only updates changed parts (virtual DOM diffing), and Callbacks are event-driven | [2.4 Reactive Programming](#24-reactive-programming-paradigm) |
| **How does it handle large data?** | Plotly uses WebGL rendering, tested with 10,000+ data points without lag |
| **What does Session save?** | Complete `fig.to_dict()`, including data, styling, layout, shapes, annotations |
| **Is the code generation accurate?** | Generated code is complete JSON reconstruction, guarantees 100% reproduction of current chart |

---

## 📌 Part 10: Technical Achievement Summary

> 📖 **Want more details?** See [Section 1.4: Scope and Deliverables](#14-scope-and-deliverables) for complete project metrics and code statistics.

| Metric | Value | 📖 See More |
|--------|-------|-------------|
| Total Lines of Code | 2,696 lines | [Code Statistics](#144-code-statistics-summary) |
| Supported Chart Types | 26+ types | [Section 5.1: Plot Types](#51-plot-types-and-chart-creation---your-26-chart-arsenal) |
| Editable Properties | 35+ properties | [Section 5.2: Property Editing](#52-property-editing-system---fine-tune-everything) |
| Number of Callbacks | 24+ callbacks | [Section 4.5: Callback Reference](#45-complete-callback-reference-table) |
| Core Classes | 4 classes | [Section 3.2: Core Class Design](#32-core-class-design) |
| Deployment Status | ✅ Running in production | [Live Demo](https://zye.pythonanywhere.com/) |

---

## 🔗 Cross-Reference: Quick Guide → Detailed Report

> **How to use this document:** The Quick Guide above gives you a fast overview. Click any link to jump to the corresponding detailed section in the Full Technical Report below.

| 🚀 Quick Guide Topic | 📚 Detailed Report Section | What You'll Learn |
|---------------------|---------------------------|-------------------|
| **Part 1:** What & Why | **[1. Introduction](#1-introduction)** | Problem statement, motivation, objectives |
| **Part 2:** File Structure | **[3. System Architecture](#3-system-architecture-and-design)** | Complete architecture diagrams |
| **Part 3:** 4 Core Classes | **[3.2 Core Class Design](#32-core-class-design)** | Full class implementations with code |
| **Part 4:** Callbacks | **[4.3 Callback Implementation](#43-callback-implementation---the-brain-of-the-application)** | All 24 callbacks explained |
| **Part 5:** UI Layout | **[4.2 Layout Implementation](#42-layout-implementation---building-the-user-interface)** | Component code and structure |
| **Part 6:** Data Flow | **[3.5 Data Flow Architecture](#35-data-flow-architecture)** | Complete flow diagrams |
| **Part 7:** Technical Points | **[2.4 Reactive Programming](#24-reactive-programming-paradigm)** | Theory and patterns |
| **Part 8:** Demo Steps | **[5. Feature Documentation](#5-complete-feature-documentation---your-user-manual)** | Step-by-step user manual |
| **Part 9:** Q&A | **[2. Literature Review](#2-literature-review-and-technical-foundation)** | Technology comparisons |
| **26+ Chart Types** | **[5.1 Plot Types](#51-plot-types-and-chart-creation---your-26-chart-arsenal)** | All chart types with examples |
| **Property Editing** | **[5.2 Property Editing](#52-property-editing-system---fine-tune-everything)** | 35+ properties documented |
| **Drawing Tools** | **[5.3 Drawing Tools](#53-drawing-and-annotation-tools---make-your-charts-talk)** | Annotation tutorial |
| **Data Management** | **[5.4 Data Management](#54-data-management---load-edit-clean)** | CSV import, editing, cleaning |
| **Session Save/Load** | **[5.5 Session Management](#55-session-management---save-and-share-your-work)** | Save workflow explained |
| **Code Export** | **[5.6 Code Export](#56-code-export---take-your-work-anywhere)** | Generated code examples |
| **Undo/Redo** | **[5.7 Undo/Redo System](#57-undoredo-system---never-lose-your-work)** | History stack explained |
| **Templates** | **[5.8 Template System](#58-template-system---instant-professional-styling)** | 10 templates compared |

---

> 📖 **Below is the full technical report with more implementation details, code explanations, and architecture diagrams.**

---

## Abstract

This report presents **PyFigureEditor**, a web-based interactive visualization GUI developed in Python that replicates and extends the functionality of MATLAB's Figure Tool. The project addresses a fundamental limitation in traditional Python plotting workflows: the inability to interactively edit and modify visualizations after their initial creation.

The application provides a graphical user interface (GUI) featuring **26+ chart types**, **real-time property editing**, **drawing and annotation tools**, **session management**, and **automatic Python code generation**. Built upon the Dash framework with Plotly.js as the visualization engine, the system implements a sophisticated **reactive callback architecture** for seamless user interaction.

Two implementation versions are provided: (1) a **Jupyter Notebook version** (`Final_Project_Implementation.ipynb`) optimized for educational and development environments, and (2) a **standalone server application** (`app.py`) suitable for production deployment. The application has been successfully deployed to PythonAnywhere and is accessible at **https://zye.pythonanywhere.com/**.

Key technical contributions include: a centralized **FigureStore state management system**, a **dynamic UI generation mechanism** for context-sensitive property editing, a **smart code generator** with automatic column type inference, and a comprehensive **undo/redo history stack**. The system demonstrates that Python can provide an interactive data visualization experience comparable to like MATLAB.

---

## Table of Contents

- [PyFigureEditor: A Python-Based Interactive Scientific Visualization Platform with MATLAB-Style Editing Capabilities](#pyfigureeditor-a-python-based-interactive-scientific-visualization-platform-with-matlab-style-editing-capabilities)
  - [MATH 4710 - Final Project Report](#math-4710---final-project-report)
    - [A Comprehensive Technical Documentation and Project Report](#a-comprehensive-technical-documentation-and-project-report)
    - [🌐 Live Production Deployment](#-live-production-deployment)
  - [**https://zye.pythonanywhere.com/**](#httpszyepythonanywherecom)
- [🎯 Quick Guide (Complete Code Architecture Analysis)](#-quick-guide-complete-code-architecture-analysis)
  - [📋 Quick Navigation: Guide ↔ Report Mapping](#-quick-navigation-guide--report-mapping)
  - [📌 Part 1: Project Overview (What \& Why)](#-part-1-project-overview-what--why)
    - [🤔 What Problem Does This Project Solve?](#-what-problem-does-this-project-solve)
    - [🎯 Core Features Overview](#-core-features-overview)
  - [📌 Part 2: Code Architecture Overview](#-part-2-code-architecture-overview)
    - [🗂️ File Structure](#️-file-structure)
    - [🏗️ `app.py` Code Organization (By Line Numbers)](#️-apppy-code-organization-by-line-numbers)
  - [📌 Part 3: Four Core Classes Explained](#-part-3-four-core-classes-explained)
    - [🔷 Class 1: `TraceDataset` (Data Container)](#-class-1-tracedataset-data-container)
    - [🔷 Class 2: `FigureStore` (Core State Management)](#-class-2-figurestore-core-state-management)
    - [🔷 Class 3: `HistoryStack` (Undo/Redo)](#-class-3-historystack-undoredo)
    - [🔷 Class 4: `CodeGenerator` (Code Generator)](#-class-4-codegenerator-code-generator)
  - [📌 Part 4: Callback System Deep Dive](#-part-4-callback-system-deep-dive)
    - [🔄 What is a Callback? (Core Mechanism)](#-what-is-a-callback-core-mechanism)
    - [🔧 Key Callback Analysis](#-key-callback-analysis)
      - [Callback 1: Property Editor (Most Complex Callback)](#callback-1-property-editor-most-complex-callback)
      - [Callback 2: Dynamic Property Panel Generation](#callback-2-dynamic-property-panel-generation)
      - [Callback 3: Drawing Tools Implementation](#callback-3-drawing-tools-implementation)
  - [📌 Part 5: UI Layout Architecture](#-part-5-ui-layout-architecture)
    - [🖥️ Overall Layout (Three-Column Design)](#️-overall-layout-three-column-design)
    - [📋 Ribbon (Top Tab Bar) Details](#-ribbon-top-tab-bar-details)
  - [📌 Part 6: Complete Data Flow Example](#-part-6-complete-data-flow-example)
    - [📊 Example: From Creating a Scatter Plot to Changing Color](#-example-from-creating-a-scatter-plot-to-changing-color)
  - [📌 Part 7: Key Technical Points](#-part-7-key-technical-points)
    - [🔑 Technical Point 1: `allow_duplicate=True`](#-technical-point-1-allow_duplicatetrue)
    - [🔑 Technical Point 2: `ctx.triggered_id` to Identify Trigger Source](#-technical-point-2-ctxtriggered_id-to-identify-trigger-source)
    - [🔑 Technical Point 3: Plotly Figure JSON Structure](#-technical-point-3-plotly-figure-json-structure)
  - [📌 Part 8: Suggested Demo Workflow](#-part-8-suggested-demo-workflow)
    - [🎬 Live Demo Steps (5 Minutes)](#-live-demo-steps-5-minutes)
  - [📌 Part 9: Common Questions Q\&A](#-part-9-common-questions-qa)
  - [📌 Part 10: Technical Achievement Summary](#-part-10-technical-achievement-summary)
  - [🔗 Cross-Reference: Quick Guide → Detailed Report](#-cross-reference-quick-guide--detailed-report)
  - [Abstract](#abstract)
  - [Table of Contents](#table-of-contents)
  - [1. Introduction](#1-introduction)
    - [1.1 Project Background and Motivation](#11-project-background-and-motivation)
      - [1.1.1 The Inspiration: MATLAB's Figure Tool](#111-the-inspiration-matlabs-figure-tool)
      - [1.1.2 The Gap in Python's Ecosystem](#112-the-gap-in-pythons-ecosystem)
      - [1.1.3 MATLAB Figure Tool Feature Reference](#113-matlab-figure-tool-feature-reference)
    - [1.2 Problem Statement](#12-problem-statement)
      - [1.2.1 The Primary Problem](#121-the-primary-problem)
      - [1.2.2 Secondary Problems (Pain Points)](#122-secondary-problems-pain-points)
      - [1.2.3 Formal Requirements](#123-formal-requirements)
    - [1.3 Project Objectives](#13-project-objectives)
      - [1.3.1 Tier 1 - Core Objectives (Must Have)](#131-tier-1---core-objectives-must-have)
      - [1.3.2 Tier 2 - Enhanced Objectives (Should Have)](#132-tier-2---enhanced-objectives-should-have)
      - [1.3.3 Tier 3 - Advanced Objectives (Nice to Have)](#133-tier-3---advanced-objectives-nice-to-have)
    - [1.4 Scope and Deliverables](#14-scope-and-deliverables)
      - [1.4.1 What's Included (In Scope)](#141-whats-included-in-scope)
      - [1.4.2 What's NOT Included (Out of Scope)](#142-whats-not-included-out-of-scope)
      - [1.4.3 Project Deliverables](#143-project-deliverables)
      - [1.4.4 Code Statistics Summary](#144-code-statistics-summary)
  - [2. Literature Review and Technical Foundation](#2-literature-review-and-technical-foundation)
    - [2.1 Overview of Python Visualization Libraries](#21-overview-of-python-visualization-libraries)
      - [2.1.1 Matplotlib: The Grandfather of Python Plotting](#211-matplotlib-the-grandfather-of-python-plotting)
      - [2.1.2 Seaborn: Beautiful Statistical Charts](#212-seaborn-beautiful-statistical-charts)
      - [2.1.3 Bokeh: Interactive Web Charts](#213-bokeh-interactive-web-charts)
      - [2.1.4 Plotly: The Winner! 🏆](#214-plotly-the-winner-)
      - [2.1.5 The Secret Weapon: Plotly's JSON Architecture](#215-the-secret-weapon-plotlys-json-architecture)
      - [2.1.6 Comparative Analysis Summary](#216-comparative-analysis-summary)
    - [2.2 MATLAB Figure Tool Analysis](#22-matlab-figure-tool-analysis)
      - [2.2.1 MATLAB Figure Tool Architecture](#221-matlab-figure-tool-architecture)
      - [2.2.2 Feature Mapping: MATLAB to PyFigureEditor](#222-feature-mapping-matlab-to-pyfigureeditor)
      - [2.2.3 Interaction Paradigms: Modal System](#223-interaction-paradigms-modal-system)
    - [2.3 Web-Based GUI Frameworks for Python](#23-web-based-gui-frameworks-for-python)
      - [2.3.1 Framework Comparison](#231-framework-comparison)
      - [2.3.2 Why Dash Was Selected](#232-why-dash-was-selected)
      - [2.3.3 Dash Architecture Overview](#233-dash-architecture-overview)
    - [2.4 Reactive Programming Paradigm](#24-reactive-programming-paradigm)
      - [2.4.1 What is Reactive Programming?](#241-what-is-reactive-programming)
      - [2.4.2 Dash's Callback System: Input, Output, State](#242-dashs-callback-system-input-output-state)
      - [2.4.3 Callback Graph and Execution Order](#243-callback-graph-and-execution-order)
      - [2.4.4 Handling Multiple Outputs: The `allow_duplicate` Pattern](#244-handling-multiple-outputs-the-allow_duplicate-pattern)
  - [3. System Architecture and Design](#3-system-architecture-and-design)
    - [3.1 High-Level System Architecture](#31-high-level-system-architecture)
    - [3.2 Core Class Design](#32-core-class-design)
      - [3.2.1 FigureStore Class — The Central Hub](#321-figurestore-class--the-central-hub)
      - [3.2.2 HistoryStack Class — The Time Machine](#322-historystack-class--the-time-machine)
      - [3.2.3 TraceDataset Class — The Data Container](#323-tracedataset-class--the-data-container)
      - [3.2.4 CodeGenerator Class — The Code Writer](#324-codegenerator-class--the-code-writer)
    - [3.3 Component Interaction Diagram](#33-component-interaction-diagram)
    - [3.4 Layout Architecture](#34-layout-architecture)
    - [3.5 Data Flow Architecture](#35-data-flow-architecture)
    - [3.6 Design Decisions and Trade-offs](#36-design-decisions-and-trade-offs)
    - [3.7 Extensibility Points](#37-extensibility-points)
  - [4. Core Implementation Details](#4-core-implementation-details)
    - [4.1 Application Initialization - The "Birth" of Your App](#41-application-initialization---the-birth-of-your-app)
      - [4.1.1 Understanding Auto-Dependency Installation](#411-understanding-auto-dependency-installation)
      - [4.1.2 Import Statements - Loading Your Toolbox](#412-import-statements---loading-your-toolbox)
      - [4.1.3 Application Creation - The "Main Engine"](#413-application-creation---the-main-engine)
    - [4.2 Layout Implementation - Building the User Interface](#42-layout-implementation---building-the-user-interface)
      - [4.2.1 Main Layout Structure - The "Blueprint"](#421-main-layout-structure---the-blueprint)
      - [4.2.2 The Ribbon Tabs - ACTUAL Code from app.py](#422-the-ribbon-tabs---actual-code-from-apppy)
      - [4.2.3 The PLOTS Tab - 26+ Chart Types in One Place](#423-the-plots-tab---26-chart-types-in-one-place)
      - [4.2.4 The Workspace Panel - Command Window \& Data View](#424-the-workspace-panel---command-window--data-view)
      - [4.2.5 The Property Inspector - Dynamic UI Generation](#425-the-property-inspector---dynamic-ui-generation)
    - [4.3 Callback Implementation - The "Brain" of the Application](#43-callback-implementation---the-brain-of-the-application)
      - [4.3.1 Understanding Callback Structure](#431-understanding-callback-structure)
      - [4.3.2 Tab Switching Callback - The Simplest Example](#432-tab-switching-callback---the-simplest-example)
      - [4.3.3 Data Management Callback - A Complex Multi-Input Example](#433-data-management-callback---a-complex-multi-input-example)
      - [4.3.4 Code Generation Callback - Auto-Generate Plotly Code](#434-code-generation-callback---auto-generate-plotly-code)
      - [4.3.5 Code Execution Callback - Running User Code](#435-code-execution-callback---running-user-code)
      - [4.3.6 Property Editor Callback - The "Big One" (35+ Properties)](#436-property-editor-callback---the-big-one-35-properties)
      - [4.3.7 History (Undo/Redo) Callback](#437-history-undoredo-callback)
    - [4.4 Helper Functions - The "Utility Belt"](#44-helper-functions---the-utility-belt)
      - [4.4.1 The clean\_figure\_dict() Function - Data Sanitizer](#441-the-clean_figure_dict-function---data-sanitizer)
      - [4.4.2 The create\_initial\_figure() Function - Default Canvas](#442-the-create_initial_figure-function---default-canvas)
      - [4.4.3 Data Parsing Functions - Understanding User Input](#443-data-parsing-functions---understanding-user-input)
      - [4.4.4 Inspector Control Generation - Dynamic UI Building](#444-inspector-control-generation---dynamic-ui-building)
      - [4.4.5 Application Launch - The Final Piece](#445-application-launch---the-final-piece)
    - [4.5 Complete Callback Reference Table](#45-complete-callback-reference-table)
  - [5. Complete Feature Documentation - Your User Manual](#5-complete-feature-documentation---your-user-manual)
    - [5.1 Plot Types and Chart Creation - Your 26+ Chart Arsenal](#51-plot-types-and-chart-creation---your-26-chart-arsenal)
      - [5.1.1 Basic 2D Charts - Where Most Analysis Starts](#511-basic-2d-charts---where-most-analysis-starts)
      - [5.1.2 Statistical Charts - For Data Scientists](#512-statistical-charts---for-data-scientists)
      - [5.1.3 3D Charts - When Two Dimensions Aren't Enough](#513-3d-charts---when-two-dimensions-arent-enough)
      - [5.1.4 Specialized Charts - For Specific Use Cases](#514-specialized-charts---for-specific-use-cases)
      - [5.1.5 Geographic Charts - Map Your Data](#515-geographic-charts---map-your-data)
    - [5.2 Property Editing System - Fine-Tune Everything](#52-property-editing-system---fine-tune-everything)
      - [5.2.1 How to Use the Property Inspector](#521-how-to-use-the-property-inspector)
      - [5.2.2 Trace Properties - 35+ Customization Options](#522-trace-properties---35-customization-options)
      - [5.2.3 Figure (Layout) Properties](#523-figure-layout-properties)
      - [5.2.4 Annotation Properties](#524-annotation-properties)
      - [5.2.5 Shape Properties](#525-shape-properties)
    - [5.3 Drawing and Annotation Tools - Make Your Charts Talk](#53-drawing-and-annotation-tools---make-your-charts-talk)
      - [5.3.1 The Drawing Toolbar - Your Creative Tools](#531-the-drawing-toolbar---your-creative-tools)
      - [5.3.2 Adding Text Annotations - Step by Step](#532-adding-text-annotations---step-by-step)
      - [5.3.3 Adding Images - Overlay Reference Pictures](#533-adding-images---overlay-reference-pictures)
    - [5.4 Data Management - Load, Edit, Clean](#54-data-management---load-edit-clean)
      - [5.4.1 Loading Data - Three Options](#541-loading-data---three-options)
      - [5.4.2 Viewing Data - Three Inspection Modes](#542-viewing-data---three-inspection-modes)
      - [5.4.3 Editing Data - Direct Manipulation](#543-editing-data---direct-manipulation)
    - [5.5 Session Management - Save and Share Your Work](#55-session-management---save-and-share-your-work)
      - [5.5.1 What Gets Saved in a Session?](#551-what-gets-saved-in-a-session)
      - [5.5.2 Saving Your Work](#552-saving-your-work)
      - [5.5.3 Loading a Previous Session](#553-loading-a-previous-session)
    - [5.6 Code Export - Take Your Work Anywhere](#56-code-export---take-your-work-anywhere)
      - [5.6.1 How Code Export Works](#561-how-code-export-works)
      - [5.6.2 The Generated Code](#562-the-generated-code)
      - [5.6.3 Using Exported Code](#563-using-exported-code)
    - [5.7 Undo/Redo System - Never Lose Your Work](#57-undoredo-system---never-lose-your-work)
      - [5.7.1 What Can Be Undone?](#571-what-can-be-undone)
      - [5.7.2 Using Undo/Redo](#572-using-undoredo)
      - [5.7.3 Understanding the History Stack](#573-understanding-the-history-stack)
    - [5.8 Template System - Instant Professional Styling](#58-template-system---instant-professional-styling)
      - [5.8.1 Available Templates (10 Built-in)](#581-available-templates-10-built-in)
      - [5.8.2 Applying a Template](#582-applying-a-template)
      - [5.8.3 Template Visual Comparison](#583-template-visual-comparison)
    - [5.9 View Tools - Navigate Your Visualization](#59-view-tools---navigate-your-visualization)
      - [5.9.1 Navigation Controls (VIEW Tab)](#591-navigation-controls-view-tab)
      - [5.9.2 Interactive Features (Built into Canvas)](#592-interactive-features-built-into-canvas)
    - [5.10 Selection and Statistics - Interactive Analysis](#510-selection-and-statistics---interactive-analysis)
      - [5.10.1 Lasso Selection](#5101-lasso-selection)
      - [5.10.2 Box Selection](#5102-box-selection)
      - [5.10.3 Remove Selected Points](#5103-remove-selected-points)


---

## 1. Introduction

> 💡 **Tips:** This section explains WHY this project exists, WHAT problem it solves, and WHO benefits from it. If you've ever been frustrated changing a plot color in Python and having to re-run your entire script, this project is for you!

### 1.1 Project Background and Motivation

#### 1.1.1 The Inspiration: MATLAB's Figure Tool

The genesis of this project stems from a direct requirement articulated by a reference video demonstrating MATLAB's Figure Tool capabilities: [YouTube Link](https://www.youtube.com/watch?v=owKwqPyg5bk).

**What is MATLAB's Figure Tool?**

MATLAB is a commercial software widely used in engineering and scientific computing. One of its most beloved features is the **Figure Tool** — a graphical interface that lets you edit charts after they're created, just like editing images in PowerPoint or Photoshop.

Imagine this scenario:
1. You create a line chart showing sales data
2. Your boss says "Can you make the line red instead of blue?"
3. In MATLAB: Click on the line → Change color → Done! ✅
4. In Python: Go back to code → Find the color parameter → Change it → Re-run script → Wait for result 😫

This difference in workflow is what motivated this entire project.

#### 1.1.2 The Gap in Python's Ecosystem

This request highlighted a significant gap in the Python data science ecosystem. While Python has become the de facto standard for data analysis and machine learning, its visualization workflow remains fundamentally different from MATLAB's interactive approach.

**What MATLAB Users Can Do (That Python Users Can't):**

| MATLAB Capability | What It Means | Python Equivalent? |
|-------------------|---------------|-------------------|
| Zoom and pan interactively | Use mouse to explore different parts of a chart | ⚠️ Limited (only some libraries) |
| Click on data points for values | Hover or click to see exact (x, y) coordinates | ⚠️ Partial (Plotly has hover) |
| Add text annotations and arrows | Draw arrows pointing to important data points | ❌ Requires code |
| Modify colors, line styles | Click element → Change its appearance | ❌ Requires code |
| Adjust layout and axis properties | Drag to resize, change titles | ❌ Requires code |
| Save the modified figure | Export your edited work | ⚠️ Partial |

**The Traditional Python Workflow (Before PyFigureEditor):**

```python
# Step 1: Write code
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [4, 5, 6], color='blue')
plt.title('My Chart')
plt.show()

# Step 2: See result... Hmm, want to change color to red?

# Step 3: Go back to code, change 'blue' to 'red'
plt.plot([1, 2, 3], [4, 5, 6], color='red')  # Modified
plt.title('My Chart')
plt.show()

# Step 4: Re-run entire script... Wait again...
# Step 5: Still not right? Repeat steps 3-4 endlessly! 😭
```

This creates friction in the data exploration process and presents a barrier for users transitioning from MATLAB to Python.

#### 1.1.3 MATLAB Figure Tool Feature Reference

The MATLAB Figure Tool reference video demonstrates the following key capabilities that served as design targets for this project:

| Timestamp | Feature | Description | PyFigureEditor Implementation |
|-----------|---------|-------------|-------------------------------|
| 0:00 | Figure Tool Overview | Introduction to the interactive editing interface | ✅ Full GUI with Ribbon interface |
| 0:20 | Save Figure | Exporting the current visualization | ✅ Session save/load as JSON |
| 0:47 | Zoom by Scrolling | Mouse wheel zoom functionality | ✅ `dragmode="zoom"` |
| 0:56 | Restore Home View | Reset to original viewport | ✅ `fig.update_xaxes(autorange=True)` |
| 1:01 | Zoom In and Out | Dedicated zoom controls | ✅ VIEW tab zoom buttons |
| 1:17 | Pan with Hand | Click-and-drag navigation | ✅ `dragmode="pan"` |
| 1:24 | Datatips | Hover to show data point values | ✅ Built-in Plotly feature |
| 1:51 | Adjust Plot Layout | Modify figure dimensions and margins | ✅ Layout property editor |
| 2:34 | Insert Plot Features | Add annotations, shapes, and text | ✅ ANNOTATE tab with 5 shape tools |

### 1.2 Problem Statement

> 💡 **Tips:** This section formally defines the problems we're solving. In academic projects, clearly stating the problem is crucial because it shows you understand what needs to be fixed before jumping into solutions.

#### 1.2.1 The Primary Problem

**Primary Problem:** Python's standard visualization libraries (Matplotlib, Seaborn, Plotly) generate static or semi-interactive outputs that cannot be fully edited through a graphical interface after creation. Users must return to source code to make modifications, breaking the flow of data exploration.

**Let's break this down in simple terms:**

```
Traditional Workflow:          PyFigureEditor Workflow:
                              
   [Write Code]                    [Click Buttons]
        ↓                               ↓
   [Run Script]                    [See Chart Instantly]
        ↓                               ↓
   [See Chart]                     [Click on Element]
        ↓                               ↓
   [Want Changes?]                 [Change Properties]
        ↓                               ↓
   [Edit Code Again] ←──┐          [Apply] → Done! ✅
        ↓               │
   [Re-run Script]      │
        ↓               │
   [Still Wrong?] ──────┘
```

#### 1.2.2 Secondary Problems (Pain Points)

**Problem 1: Steep Learning Curve**

Each visualization library has its own API (Application Programming Interface), requiring users to memorize extensive documentation to make even simple changes.

```python
# To change line color in different libraries:

# Matplotlib:
ax.plot(x, y, color='red')
ax.lines[0].set_color('red')  # After creation

# Plotly:
fig.update_traces(line_color='red')
fig.data[0].line.color = 'red'  # Alternative

# Seaborn:
# Good luck finding where to change it! 😅
```

**Problem 2: No Unified Interface**

Different plot types often require different syntax, making it difficult to switch between visualizations.

```python
# Creating different chart types requires completely different code:

# Scatter plot
px.scatter(df, x='col1', y='col2')

# Bar chart
px.bar(df, x='category', y='value')

# 3D surface
go.Figure(data=[go.Surface(z=z_data)])

# Candlestick
go.Figure(data=[go.Candlestick(x=dates, open=open, high=high, low=low, close=close)])

# Each one is completely different syntax! 😵
```

**Problem 3: Limited Persistence**

There is no standard way to save an "editing session" that preserves all modifications made to a figure.

```python
# You can save the image, but not the editable state!
fig.write_image("output.png")  # Static image - can't edit later
fig.write_html("output.html")  # Can view, but can't continue editing

# What if you want to save your work and continue tomorrow?
# Answer: You can't... until PyFigureEditor! 🎉
```

**Problem 4: Code-Visualization Disconnect**

Changes made through limited interactive features (like Plotly's built-in zoom) are not reflected back to code, making reproducibility challenging.

```python
# You zoom in on a chart using the mouse...
# But how do you recreate that exact view in code?
# You have to manually figure out the axis ranges!

# PyFigureEditor solution: Auto-generates code for ANY state!
```

#### 1.2.3 Formal Requirements

The solution must provide:

| Requirement ID | Description | Priority | How We Solve It |
|----------------|-------------|----------|-----------------|
| R1 | Web-based GUI accessible through a browser | Critical | Dash framework → runs in any browser |
| R2 | Support for multiple chart types | Critical | 26+ chart types implemented |
| R3 | Real-time property editing without code changes | Critical | Inspector panel with Apply button |
| R4 | Drawing and annotation capabilities | High | ANNOTATE tab with 5 shape tools |
| R5 | Undo/redo functionality | High | `HistoryStack` class with 50 states |
| R6 | Session save/load functionality | High | JSON serialization via `FigureStore` |
| R7 | Automatic Python code generation | Medium | `CodeGenerator` class |
| R8 | Data import/export capabilities | Medium | CSV upload + demo data generator |
| R9 | Production deployment capability | Medium | Deployed to PythonAnywhere |

### 1.3 Project Objectives

> 💡 **Tips:** Objectives are organized into "must have," "should have," and "nice to have" tiers. This is a common software engineering practice called **MoSCoW prioritization** (Must, Should, Could, Won't). It helps you focus on critical features first.

#### 1.3.1 Tier 1 - Core Objectives (Must Have)

These are the features without which the project would be considered a failure:

| Objective | Status | Implementation Location |
|-----------|--------|-------------------------|
| 1. Develop a functional web-based GUI | ✅ Done | `app.py` lines 846-1187 (Layout) |
| 2. Implement a property inspector | ✅ Done | `app.py` lines 1767-2090 (Inspector callbacks) |
| 3. Provide at least 15 chart types | ✅ Done | 26+ types in PLOTS tab |
| 4. Enable zoom, pan, reset controls | ✅ Done | VIEW tab implementation |
| 5. Deploy to public server | ✅ Done | [zye.pythonanywhere.com](https://zye.pythonanywhere.com/) |

#### 1.3.2 Tier 2 - Enhanced Objectives (Should Have)

These significantly improve the user experience:

| Objective | Status | Implementation Location |
|-----------|--------|-------------------------|
| 1. Drawing tools for shapes | ✅ Done | ANNOTATE tab, `dragmode` callbacks |
| 2. Text annotation with arrows | ✅ Done | `add_annotation()` implementation |
| 3. Undo/redo system | ✅ Done | `HistoryStack` class (lines 570-636) |
| 4. Session save/load | ✅ Done | `serialize_session()` / `load_session()` |
| 5. CSV import interface | ✅ Done | DATA tab with `dcc.Upload` |

#### 1.3.3 Tier 3 - Advanced Objectives (Nice to Have)

These are bonus features that make the project stand out:

| Objective | Status | Notes |
|-----------|--------|-------|
| 1. Auto code generation | ✅ Done | `CodeGenerator` class generates reproducible Python code |
| 2. Smart column inference | ✅ Done | `generate_smart_plot_code()` analyzes DataFrame columns |
| 3. DataTable integration | ✅ Done | Interactive data editing in DATA tab |
| 4. Background image overlay | ✅ Done | Image upload in ANNOTATE tab |
| 5. Multiple theme presets | ✅ Done | Theme dropdown in Layout properties |

### 1.4 Scope and Deliverables

> 💡 **Tips:** Scope defines what's IN and OUT of your project. This is critical for setting expectations and avoiding "scope creep" (when a project keeps growing beyond original plans).

#### 1.4.1 What's Included (In Scope)

| Category | Specific Features |
|----------|-------------------|
| **Framework** | Web-based application using Dash/Plotly |
| **2D Charts** | Scatter, Line, Bar, Area, Bubble, Histogram, Box, Violin, Pie, Heatmap |
| **3D Charts** | Scatter3D, Line3D, Surface, Mesh3D |
| **Geographic** | Scatter Geo, Choropleth, Globe projection |
| **Specialized** | Candlestick, Waterfall, Funnel, Treemap, Sunburst, Polar, Ternary |
| **Statistical** | Scatter Matrix, Parallel Coordinates |
| **Editing** | Property inspector with 35+ editable properties |
| **Drawing** | Line, Rectangle, Circle, Freeform, Polygon |
| **Annotation** | Text labels, arrows, images |
| **Persistence** | Session save/load as JSON |
| **Deployment** | Live on PythonAnywhere |

#### 1.4.2 What's NOT Included (Out of Scope)

| Feature | Why It's Out of Scope |
|---------|----------------------|
| Real-time collaborative editing | Would require WebSocket/real-time sync infrastructure |
| Mobile-native applications | Dash is web-based; mobile apps need React Native/Flutter |
| Offline desktop executables | Would need PyInstaller/Electron packaging |
| External database integration | Focus is on visualization, not data storage |
| Animation/video export | Complex feature requiring additional libraries |
| Machine learning integration | Outside the scope of a visualization tool |

#### 1.4.3 Project Deliverables

| ID | Deliverable | Description | Location |
|----|-------------|-------------|----------|
| D1 | **Jupyter Notebook** | Educational version with step-by-step explanations | `Final_Project_Implementation.ipynb` |
| D2 | **Standalone App** | Production-ready 2,696-line Python application | `app.py` |
| D3 | **Documentation** | This comprehensive technical report | `README.md` |
| D4 | **Live Demo** | Publicly accessible deployment | [zye.pythonanywhere.com](https://zye.pythonanywhere.com/) |
| D5 | **Source Code** | Complete codebase with version control | [GitHub Repository](https://github.com/1235357/PyFigureEditor) |

#### 1.4.4 Code Statistics Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    PyFigureEditor Stats                     │
├─────────────────────────────────────────────────────────────┤
│  Total Lines of Code        │  2,696 lines                  │
│  Core Classes               │  4 (TraceDataset, FigureStore,│
│                             │     HistoryStack, CodeGenerator)
│  Callbacks                  │  24+ reactive callbacks        │
│  Chart Types Supported      │  26+ types                     │
│  Editable Properties        │  35+ properties                │
│  UI Sections                │  16 distinct sections          │
│  Deployment Status          │  ✅ Production                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Literature Review and Technical Foundation

> 💡 **Tips:** This section is like a "research phase" documentation. Before building anything, good engineers study existing solutions. We looked at many Python visualization libraries and chose the best one for our needs. This section explains WHY we made the choices we did.

This section provides a comprehensive analysis of the existing technologies, frameworks, and approaches that informed the design and implementation of PyFigureEditor. Understanding this foundation is essential for appreciating the architectural decisions made throughout the project.

### 2.1 Overview of Python Visualization Libraries

> 💡 **Tips:** Python has MANY libraries for creating charts. Each has its own strengths and weaknesses. Choosing the right one is like choosing the right tool for a job — you wouldn't use a hammer to screw in a nail!

Python's ecosystem offers numerous visualization libraries, each with distinct philosophies, capabilities, and limitations. A thorough understanding of these options was necessary to select the most appropriate foundation for this project.

#### 2.1.1 Matplotlib: The Grandfather of Python Plotting

**What is Matplotlib?**

Matplotlib, created by John D. Hunter in 2003, is the foundational visualization library in Python. It was designed to mimic MATLAB's plotting interface (hence the name "Mat-plot-lib").

**Think of it like:** Microsoft Paint for data visualization — powerful, fundamental, but requires effort for fancy results.

```python
# Two ways to use Matplotlib:

# Way 1: Pyplot Interface (MATLAB-like, beginner-friendly)
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title('Simple Plot')
plt.show()

# Way 2: Object-Oriented Interface (more control, professional)
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
ax.set_title('Simple Plot')
plt.show()
```

**Strengths:**
| Strength | What It Means | Example |
|----------|---------------|---------|
| Highly customizable | Control every pixel | Change font size of tick labels |
| Huge community | Lots of StackOverflow answers | Any question already answered |
| NumPy/Pandas integration | Works with your data tools | `ax.plot(df['x'], df['y'])` |
| Publication-quality | Academic papers accept it | Export to PDF/SVG |

**Why We DIDN'T Choose Matplotlib:**
| Limitation | Problem for Our Project |
|------------|------------------------|
| Static output | Charts are images, can't click on them |
| No built-in interactivity | Can't zoom/pan without extra code |
| Web integration is hacky | Need `mpld3` or convert to other formats |
| No property editor | Can't click-and-edit after creation |

```python
# The fundamental problem with Matplotlib for our use case:
plt.plot([1,2,3], [4,5,6], color='blue')
plt.show()  # This shows a static image!

# To change color, you MUST go back to code:
plt.plot([1,2,3], [4,5,6], color='red')  # Edit code
plt.show()  # Run again

# There's no way to click on the blue line and change it to red!
```

#### 2.1.2 Seaborn: Beautiful Statistical Charts

**What is Seaborn?**

Seaborn is built ON TOP of Matplotlib, providing prettier defaults and easier statistical visualizations.

**Think of it like:** Instagram filters for Matplotlib — same underlying engine, but prettier output with less effort.

```python
import seaborn as sns
import pandas as pd

# Load built-in dataset
tips = sns.load_dataset("tips")

# One-liner creates a beautiful box plot!
sns.boxplot(x="day", y="total_bill", data=tips)
```

**Strengths:**
| Strength | What It Means |
|----------|---------------|
| Beautiful defaults | Charts look professional immediately |
| Statistical focus | Built-in mean, confidence intervals, distributions |
| Less code | `sns.boxplot(...)` vs 10 lines of Matplotlib |

**Why We DIDN'T Choose Seaborn:**
| Limitation | Problem for Our Project |
|------------|------------------------|
| Built on Matplotlib | Inherits all of Matplotlib's static limitations |
| Limited chart types | Great for statistics, poor for maps/3D/finance |
| No interactivity | Same static output problem |

#### 2.1.3 Bokeh: Interactive Web Charts

**What is Bokeh?**

Bokeh (pronounced "BO-kay", like the photography term) is designed specifically for creating interactive visualizations in web browsers.

**Think of it like:** A JavaScript charting library that you can use from Python.

```python
from bokeh.plotting import figure, show

# Create figure with interactive tools
p = figure(
    title="Interactive Plot", 
    tools="pan,wheel_zoom,box_zoom,reset,hover"  # Built-in interactivity!
)
p.circle([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], size=15)
show(p)  # Opens in browser with zoom/pan working!
```

**Strengths:**
| Strength | What It Means |
|----------|---------------|
| Web-native | Outputs HTML that works in any browser |
| Good interactivity | Zoom, pan, hover tooltips built-in |
| Bokeh Server | Can create real-time updating dashboards |
| Standalone HTML | Send someone a single .html file |

**Why We DIDN'T Choose Bokeh:**
| Limitation | Problem for Our Project |
|------------|------------------------|
| Fewer chart types | No candlestick, limited 3D, fewer maps |
| Complex callbacks | Writing interactive apps is harder than Dash |
| No property editor | Would need to build from scratch |
| Separate from Plotly | Can't use with Dash framework |

#### 2.1.4 Plotly: The Winner! 🏆

**What is Plotly?**

Plotly is a modern, interactive visualization library that outputs web-based charts. It has both a JavaScript library and Python/R bindings.

**Think of it like:** A professional charting library used by companies like Netflix, Tesla, and NASA.

```python
import plotly.express as px  # High-level, easy API
import plotly.graph_objects as go  # Low-level, full control

# High-level Express API (one-liner!)
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
fig.show()

# Low-level Graph Objects API (full control)
fig = go.Figure(data=[
    go.Scatter(
        x=[1, 2, 3], 
        y=[4, 5, 6],
        mode="lines+markers",
        marker=dict(color="red", size=10),
        line=dict(width=2)
    )
])
fig.update_layout(title="My Plot")
fig.show()
```

**Why Plotly is PERFECT for This Project:**

| Strength | Why It Matters | Code Example |
|----------|----------------|--------------|
| **40+ chart types** | Scatter, Bar, 3D, Maps, Financial — all in one library | `px.scatter_3d()`, `go.Candlestick()` |
| **Native interactivity** | Zoom, pan, hover work automatically | No extra code needed! |
| **JSON-based** | Charts are dictionaries, easy to save/load | `fig.to_dict()`, `fig.to_json()` |
| **Dash integration** | Same company made Dash framework | `dcc.Graph(figure=fig)` |
| **Editable mode** | Can enable click-to-edit on charts | `config={'editable': True}` |

#### 2.1.5 The Secret Weapon: Plotly's JSON Architecture

> 💡 **Tips:** This is the KEY insight that makes PyFigureEditor possible! Plotly stores charts as dictionaries (JSON), not as compiled binary objects. This means we can read and modify ANY property!

**Understanding Plotly's Internal Structure:**

Every Plotly figure is just a Python dictionary with two main keys: `data` and `layout`.

```python
# Create a simple figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=[1,2,3], y=[4,5,6], name="My Line"))
fig.update_layout(title="Example")

# Look inside!
print(fig.to_dict())
```

**Output (simplified):**
```python
{
    "data": [                          # List of all visual elements (traces)
        {
            "type": "scatter",         # What kind of chart
            "x": [1, 2, 3],            # X coordinates
            "y": [4, 5, 6],            # Y coordinates
            "mode": "lines",           # Lines, markers, or both
            "name": "My Line",         # Legend label
            "marker": {                # Point appearance
                "color": "blue",
                "size": 6,
                "symbol": "circle"
            },
            "line": {                  # Line appearance
                "color": "blue", 
                "width": 2,
                "dash": "solid"
            }
        }
    ],
    "layout": {                        # Overall chart settings
        "title": {"text": "Example"},
        "xaxis": {
            "title": {"text": "X Axis"},
            "range": [0, 4]
        },
        "yaxis": {
            "title": {"text": "Y Axis"},
            "range": [3, 7]
        },
        "template": "plotly_white",
        "shapes": [...],               # Drawn rectangles, lines, etc.
        "annotations": [...],          # Text labels
        "images": [...]                # Background images
    }
}
```

**Why This Architecture Enables Our Project:**

```python
# Because it's a dictionary, we can:

# 1. READ any property
current_color = fig.data[0].marker.color  # "blue"

# 2. MODIFY any property
fig.data[0].marker.color = "red"  # Change to red!

# 3. SAVE the entire state
with open("my_session.json", "w") as f:
    json.dump(fig.to_dict(), f)

# 4. LOAD it back later
with open("my_session.json", "r") as f:
    fig = go.Figure(json.load(f))

# 5. GENERATE code to recreate it
code = f"fig = go.Figure({fig.to_dict()})"
```

**This is exactly what MATLAB's Figure Tool does internally!**

#### 2.1.6 Comparative Analysis Summary

| Feature | Matplotlib | Seaborn | Bokeh | **Plotly** |
|---------|------------|---------|-------|------------|
| Web Native | ❌ No | ❌ No | ✅ Yes | ✅ **Yes** |
| Interactivity | ⚠️ Limited | ⚠️ Limited | ✅ Good | ✅ **Excellent** |
| Chart Types | ✅ Many | ⚠️ Statistical | ✅ Many | ✅ **Most (40+)** |
| JSON Serializable | ❌ No | ❌ No | ⚠️ Partial | ✅ **Yes** |
| Dash Integration | ⚠️ Via Plotly | ⚠️ Via Plotly | ❌ Separate | ✅ **Native** |
| 3D Support | ⚠️ Basic | ❌ No | ⚠️ Limited | ✅ **Excellent** |
| Geo/Maps | ⚠️ Basemap | ❌ No | ✅ Good | ✅ **Excellent** |
| Learning Curve | 📈 Steep | 📉 Easy | 📊 Moderate | 📊 **Moderate** |

**Conclusion:** Plotly's comprehensive feature set, JSON-based architecture, and native Dash integration made it the clear choice for implementing a MATLAB-style interactive figure editor.

### 2.2 MATLAB Figure Tool Analysis

> 💡 **Tips:** To replicate something, you must first understand how it works. We carefully studied MATLAB's Figure Tool to identify which features to implement.

#### 2.2.1 MATLAB Figure Tool Architecture

MATLAB uses a hierarchical object model where every visual element is an "object" with properties:

```
Figure (gcf - "get current figure")
├── Axes (gca - "get current axes")
│   ├── Line objects      ← The actual data lines
│   ├── Scatter objects   ← The actual data points
│   ├── Bar objects       ← The actual bars
│   ├── Surface objects   ← 3D surfaces
│   └── Text objects      ← Labels on the chart
├── UI Controls           ← Buttons, sliders on the figure
├── Legends               ← The legend box
└── Colorbars             ← Color scale indicator
```

**In MATLAB, you can access and modify any object:**

```matlab
% Create a plot
h = plot(1:10, rand(1,10));

% h is now a "handle" to the line object
% You can modify it directly!
h.Color = 'red';           % Change color
h.LineWidth = 2;           % Make thicker
h.Marker = 'o';            % Add circle markers
h.MarkerSize = 10;         % Make markers bigger
h.DisplayName = 'My Data'; % Change legend text
```

**PyFigureEditor replicates this with Plotly's similar structure:**

```python
# Create a plot
fig = px.line(x=[1,2,3,4,5], y=[2,4,3,5,4])

# Access the trace (equivalent to MATLAB's handle)
# fig.data[0] is like h in MATLAB

# Modify it!
fig.data[0].line.color = 'red'           # Change color
fig.data[0].line.width = 2               # Make thicker
fig.data[0].mode = 'lines+markers'       # Add markers
fig.data[0].marker.size = 10             # Make markers bigger
fig.data[0].name = 'My Data'             # Change legend text
```

#### 2.2.2 Feature Mapping: MATLAB to PyFigureEditor

| MATLAB Feature | MATLAB Command | PyFigureEditor Implementation | Code Location |
|----------------|----------------|-------------------------------|---------------|
| Property Inspector | Built-in GUI panel | Custom Dash Inspector | `app.py` lines 1767-2090 |
| Zoom | `zoom on` | Plotly native + VIEW tab | `dragmode="zoom"` |
| Pan | `pan on` | Plotly native + VIEW tab | `dragmode="pan"` |
| Data Cursor | `datacursormode on` | Plotly hover tooltips | Built-in |
| Insert Text | `gtext()` function | Annotation modal | ANNOTATE tab |
| Insert Arrow | Arrow annotation | `showarrow=True` | `add_annotation()` |
| Insert Shape | Rectangle, ellipse tools | Drawing mode buttons | `dragmode="drawrect"` |
| Save Figure | `saveas()`, `exportgraphics()` | JSON session export | `serialize_session()` |
| Edit Plot | Property editor GUI | Dynamic property inspector | Inspector callbacks |
| Undo | Ctrl+Z | HistoryStack class | `history_stack.undo()` |

#### 2.2.3 Interaction Paradigms: Modal System

**What is a "Modal" System?**

MATLAB uses a **modal** interaction system where the figure enters different "modes." Only one mode is active at a time.

```
User clicks "Zoom" button
        ↓
Figure enters ZOOM MODE
        ↓
All mouse interactions now zoom (not pan or select)
        ↓
User clicks "Pan" button
        ↓
Figure enters PAN MODE
        ↓
All mouse interactions now pan (not zoom)
```

**PyFigureEditor replicates this with Plotly's `dragmode` property:**

```python
# MATLAB: zoom on
fig.update_layout(dragmode="zoom")
# Now mouse drag = zoom

# MATLAB: pan on  
fig.update_layout(dragmode="pan")
# Now mouse drag = pan

# Drawing modes (no MATLAB equivalent - we added these!)
fig.update_layout(dragmode="drawrect")   # Draw rectangles
fig.update_layout(dragmode="drawline")   # Draw lines
fig.update_layout(dragmode="drawcircle") # Draw circles
fig.update_layout(dragmode="drawopenpath")  # Freeform drawing
fig.update_layout(dragmode="drawclosedpath") # Polygon drawing
```

**Our VIEW tab buttons simply change this `dragmode` property:**

```python
@app.callback(
    Output("main-graph", "figure"),
    Input("btn-zoom", "n_clicks"),
    Input("btn-pan", "n_clicks"),
    Input("btn-reset", "n_clicks"),
    State("main-graph", "figure"),
)
def handle_view_tools(n_zoom, n_pan, n_reset, current_fig):
    trigger = ctx.triggered_id
    fig = go.Figure(current_fig)
    
    if trigger == "btn-zoom":
        fig.update_layout(dragmode="zoom")
    elif trigger == "btn-pan":
        fig.update_layout(dragmode="pan")
    elif trigger == "btn-reset":
        fig.update_xaxes(autorange=True)
        fig.update_yaxes(autorange=True)
    
    return fig
```

### 2.3 Web-Based GUI Frameworks for Python

> 💡 **Tips:** Once we chose Plotly for charts, we needed a way to build the buttons, dropdowns, and panels around it. This is called a "GUI framework." We compared several options.

#### 2.3.1 Framework Comparison

| Framework | Type | Learning Curve | Plotly Integration | Real-time Updates | Our Assessment |
|-----------|------|----------------|-------------------|-------------------|----------------|
| Flask + JS | Traditional | 📈 High | Manual | WebSocket | Too much JavaScript |
| Django + JS | Traditional | 📈 High | Manual | Channels | Overkill for this project |
| Streamlit | Declarative | 📉 Low | Good | Automatic | Not enough control |
| Gradio | ML-focused | 📉 Low | Limited | Automatic | Wrong use case |
| Panel | HoloViews | 📊 Moderate | Good | Good | Less mature |
| **Dash** | Reactive | 📊 Moderate | **Native** | **Callbacks** | **Winner!** 🏆 |

#### 2.3.2 Why Dash Was Selected

**Dash** (by Plotly) was selected for these key reasons:

**Reason 1: Native Plotly Integration**

Dash is built by the same team that created Plotly. The `dcc.Graph` component accepts Plotly figures directly — no conversion needed!

```python
import dash
from dash import dcc, html
import plotly.express as px

app = dash.Dash(__name__)

# Create a Plotly figure
fig = px.scatter(x=[1,2,3], y=[4,5,6])

# Put it directly in Dash!
app.layout = html.Div([
    dcc.Graph(figure=fig)  # That's it! No conversion needed
])
```

**Reason 2: Reactive Programming Model**

Dash's callback system handles user interactions WITHOUT writing JavaScript:

```python
# When slider changes, chart automatically updates
@app.callback(
    Output('my-chart', 'figure'),
    Input('my-slider', 'value')
)
def update_chart(slider_value):
    fig = px.scatter(x=[1,2,3], y=[v*slider_value for v in [1,2,3]])
    return fig
```

**Reason 3: Bootstrap Components**

Dash Bootstrap Components (dbc) provides professional-looking UI elements:

```python
import dash_bootstrap_components as dbc

# Professional button group
dbc.ButtonGroup([
    dbc.Button("Save", color="primary"),
    dbc.Button("Load", color="secondary"),
    dbc.Button("Delete", color="danger"),
])

# Clean tabs
dbc.Tabs([
    dbc.Tab(label="HOME", children=[...]),
    dbc.Tab(label="DATA", children=[...]),
    dbc.Tab(label="PLOTS", children=[...]),
])
```

**Reason 4: Production Ready**

Dash applications can be deployed to standard WSGI servers:

```python
# app.py
app = dash.Dash(__name__)
application = app.server  # This is a standard Flask server!

# Can deploy to:
# - PythonAnywhere (what we use)
# - Heroku
# - AWS
# - Google Cloud
# - Any WSGI-compatible host
```

**Reason 5: No JavaScript Required**

The ENTIRE PyFigureEditor application (2,696 lines) is written in Python. Zero JavaScript!

#### 2.3.3 Dash Architecture Overview

> 💡 **Tips:** Dash apps have two parts: **Layout** (what the user sees) and **Callbacks** (how the app responds to user actions). This is similar to HTML (structure) and JavaScript (behavior).

**Part 1: Layout (Declarative UI)**

Layout defines WHAT components appear on the page:

```python
app.layout = html.Div([
    # Header
    html.H1("PyFigureEditor"),
    
    # The chart
    dcc.Graph(id='main-graph', figure=initial_figure),
    
    # A slider
    dcc.Slider(id='opacity-slider', min=0, max=1, value=0.5),
    
    # A dropdown
    dcc.Dropdown(
        id='color-dropdown',
        options=[
            {'label': 'Red', 'value': 'red'},
            {'label': 'Blue', 'value': 'blue'},
            {'label': 'Green', 'value': 'green'},
        ],
        value='blue'
    ),
    
    # Output area
    html.Div(id='status-message')
])
```

**Part 2: Callbacks (Reactive Logic)**

Callbacks define HOW the app responds to user interactions:

```python
@app.callback(
    Output('main-graph', 'figure'),      # What to UPDATE
    Output('status-message', 'children'), # Can have multiple outputs!
    Input('opacity-slider', 'value'),    # What TRIGGERS the update
    Input('color-dropdown', 'value'),    # Another trigger
    State('main-graph', 'figure'),       # Data to READ (doesn't trigger)
)
def update_chart(opacity, color, current_figure):
    # This function runs AUTOMATICALLY when slider or dropdown changes!
    fig = go.Figure(current_figure)
    fig.update_traces(opacity=opacity, marker_color=color)
    return fig, f"Updated: opacity={opacity}, color={color}"
```

**The MVC Pattern Mapping:**

This separation of concerns maps to the classic Model-View-Controller (MVC) pattern:

| MVC Component | Dash Equivalent | PyFigureEditor Example |
|---------------|-----------------|------------------------|
| **Model** (Data) | Python classes | `FigureStore`, `HistoryStack`, `TraceDataset` |
| **View** (UI) | Layout components | `html.Div`, `dcc.Graph`, `dbc.Button` |
| **Controller** (Logic) | Callback functions | `@app.callback` decorated functions |

### 2.4 Reactive Programming Paradigm

> 💡 **Tips:** This section explains the "magic" that makes PyFigureEditor feel responsive. Instead of constantly checking "did the user click anything?", we tell the computer "when this happens, do that."

#### 2.4.1 What is Reactive Programming?

**Traditional (Imperative) Programming — The Old Way:**

```python
# You have to manually check for changes in a loop
while True:
    if button_was_clicked():
        update_display()
    if slider_was_moved():
        recalculate()
    if dropdown_changed():
        refresh_data()
    time.sleep(0.1)  # Check every 100ms (polling)
```

**Problems with this approach:**
- Wastes CPU cycles checking things that didn't change
- Code becomes a mess of if-statements
- Easy to miss events between checks

**Reactive (Declarative) Programming — The Modern Way:**

```python
# Declare what should happen when things change
@when(button.clicked)
def on_button_click():
    update_display()

@when(slider.changed)
def on_slider_change():
    recalculate()

@when(dropdown.changed)
def on_dropdown_change():
    refresh_data()

# No loop! The system handles it automatically
```

**Benefits:**
- Only runs code when something actually changes
- Clear cause-and-effect relationships
- Easier to understand and maintain

#### 2.4.2 Dash's Callback System: Input, Output, State

Dash implements reactivity through **callbacks** with three types of dependencies:

| Dependency | Symbol | Purpose | Triggers Callback? |
|------------|--------|---------|-------------------|
| `Output` | ← | What component to UPDATE | N/A (result) |
| `Input` | → | What TRIGGERS the callback | ✅ Yes |
| `State` | 📋 | What to READ (without triggering) | ❌ No |

**Example with All Three:**

```python
@app.callback(
    Output('result-display', 'children'),    # OUTPUT: Update this div's text
    Input('submit-button', 'n_clicks'),      # INPUT: Triggered when clicked
    State('name-input', 'value'),            # STATE: Read the name
    State('email-input', 'value'),           # STATE: Read the email
)
def submit_form(n_clicks, name, email):
    # This function ONLY runs when submit-button is clicked
    # It does NOT run when name or email changes (they're State, not Input)
    
    if n_clicks is None:
        return "Please fill the form"  # Initial state
    
    return f"Submitted: {name} ({email})"
```

**Why State is Important:**

Without `State`, you'd need to either:
1. Make name/email trigger submission (bad UX — submits while typing!)
2. Use global variables (bad practice)

```python
# BAD: Using Input for everything
@app.callback(
    Output('result', 'children'),
    Input('submit-btn', 'n_clicks'),
    Input('name-input', 'value'),  # This triggers on EVERY keystroke!
)
def submit(n, name):
    return f"Submitted: {name}"  # Runs as user types — not what we want!

# GOOD: Using State for non-triggering values
@app.callback(
    Output('result', 'children'),
    Input('submit-btn', 'n_clicks'),  # Only this triggers
    State('name-input', 'value'),     # Read but don't trigger
)
def submit(n, name):
    return f"Submitted: {name}"  # Only runs on button click!
```

#### 2.4.3 Callback Graph and Execution Order

> 💡 **Tips:** Dash automatically figures out the order to run callbacks. If Callback A updates something that Callback B depends on, Dash runs A first, then B. This is called a "dependency graph."

**Example: The Chain Reaction in PyFigureEditor**

When you click "Apply" after changing a color:

```
Step 1: User clicks "Apply Properties" button
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  CALLBACK: apply_property_changes                        │
│  ─────────────────────────────────────────────────────  │
│  Input: btn-apply-props.n_clicks (button was clicked)   │
│  State: color-input.value, size-input.value, etc.       │
│  Output: main-graph.figure (the chart)                  │
│                                                          │
│  Action: Read new color → Update fig.data[0].color      │
│          → Return modified figure                        │
└─────────────────────────────────────────────────────────┘
        │
        │ (figure was updated, triggers next callback!)
        ▼
┌─────────────────────────────────────────────────────────┐
│  CALLBACK: update_element_options                        │
│  ─────────────────────────────────────────────────────  │
│  Input: main-graph.figure (figure changed!)             │
│  Output: dd-element-select.options (dropdown items)     │
│                                                          │
│  Action: Read all traces from new figure                │
│          → Generate dropdown options                     │
│          → Return ["Trace 0", "Trace 1", ...]           │
└─────────────────────────────────────────────────────────┘
        │
        │ (dropdown options changed, triggers next!)
        ▼
┌─────────────────────────────────────────────────────────┐
│  CALLBACK: update_inspector_controls                     │
│  ─────────────────────────────────────────────────────  │
│  Input: dd-element-select.value (selected item)         │
│  State: main-graph.figure                               │
│  Output: inspector-controls.children (property panel)   │
│                                                          │
│  Action: Read selected trace's properties               │
│          → Generate color/size/name input fields        │
│          → Return the property editor UI                │
└─────────────────────────────────────────────────────────┘
        │
        ▼
     DONE! ✅
     (User sees updated chart with new inspector values)
```

**This automatic chaining is fundamental to PyFigureEditor:**
- User changes ONE thing (color)
- System automatically updates EVERYTHING related (chart, dropdown, inspector)
- No manual coordination needed!

#### 2.4.4 Handling Multiple Outputs: The `allow_duplicate` Pattern

> 💡 **Tips:** By default, Dash says "only ONE callback can update each output." But in our app, MANY things update the chart! We use `allow_duplicate=True` to allow this.

**The Problem:**

```python
# Callback 1: Property editor updates the chart
@app.callback(Output("main-graph", "figure"), Input("btn-apply", "n_clicks"))
def apply_properties(...): 
    return updated_fig

# Callback 2: Drawing tool updates the chart
@app.callback(Output("main-graph", "figure"), Input("btn-draw-rect", "n_clicks"))
def draw_rectangle(...):
    return updated_fig

# Callback 3: Undo updates the chart
@app.callback(Output("main-graph", "figure"), Input("btn-undo", "n_clicks"))
def undo(...):
    return previous_fig

# ❌ ERROR! Dash says: "Multiple callbacks update main-graph.figure!"
```

**The Solution: `allow_duplicate=True`**

```python
# Callback 1: Property editor updates the chart
@app.callback(
    Output("main-graph", "figure", allow_duplicate=True),  # ← Allow duplicate!
    Input("btn-apply", "n_clicks"),
    prevent_initial_call=True
)
def apply_properties(...): 
    return updated_fig

# Callback 2: Drawing tool updates the chart
@app.callback(
    Output("main-graph", "figure", allow_duplicate=True),  # ← Same output allowed!
    Input("btn-draw-rect", "n_clicks"),
    prevent_initial_call=True
)
def draw_rectangle(...):
    return updated_fig

# Callback 3: Undo updates the chart
@app.callback(
    Output("main-graph", "figure", allow_duplicate=True),  # ← Also allowed!
    Input("btn-undo", "n_clicks"),
    prevent_initial_call=True
)
def undo(...):
    return previous_fig

# ✅ SUCCESS! All three can update the same chart
```

**This pattern is used throughout PyFigureEditor:**

```python
# In app.py, we have 10+ callbacks that all update main-graph.figure:
# - Plot creation callbacks (scatter, bar, line, etc.)
# - Property editing callbacks
# - Drawing tool callbacks
# - Annotation callbacks
# - Undo/Redo callbacks
# - Session load callbacks
# - Theme change callbacks
# All using allow_duplicate=True!
```

---

## 3. System Architecture and Design

> 💡 **Tips:** This chapter is like the "blueprint" of a building. Before constructing anything, architects draw detailed plans showing how everything fits together. This section explains HOW PyFigureEditor is organized internally.

This chapter presents the architectural blueprint of PyFigureEditor, explaining how different components interact to create a cohesive, maintainable, and extensible system.

### 3.1 High-Level System Architecture

> 💡 **Tips:** Software architecture is often organized in "layers," like a cake. Each layer has a specific job, and they communicate only with adjacent layers. This makes the code easier to understand, test, and modify.

PyFigureEditor follows a **layered architecture** pattern that separates concerns and promotes modularity:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PRESENTATION LAYER                                  │
│          (What the User Sees - HTML/CSS/JavaScript via Dash)                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Ribbon    │  │   Graph     │  │  Property   │  │   Modal Dialogs     │ │
│  │   Toolbar   │  │   Display   │  │  Inspector  │  │ (Save, About, etc.) │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│                                                                              │
│  Components: dbc.Tabs, dcc.Graph, dbc.Card, dbc.Modal, html.Div, etc.       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ User clicks button
                                       │ or changes input
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CALLBACK LAYER (Controller)                             │
│              (The "Brain" - Responds to User Actions)                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  20+ Callback Functions, each handling specific interactions:         │   │
│  │                                                                        │   │
│  │  @app.callback(Output(...), Input(...))                               │   │
│  │  def handle_user_action(...):                                         │   │
│  │      # Process input → Update state → Return new output               │   │
│  │                                                                        │   │
│  │  Examples:                                                             │   │
│  │  • apply_property_changes()  - When user clicks "Apply"               │   │
│  │  • generate_and_trigger_plot() - When user clicks a chart button      │   │
│  │  • handle_undo_redo()        - When user clicks Undo/Redo             │   │
│  │  • update_inspector_controls() - When selected element changes        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Callbacks use these classes
                                       │ to manage data
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     BUSINESS LOGIC LAYER (Model)                             │
│              (The "Data Experts" - Python Classes)                           │
│                                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ FigureStore │  │HistoryStack │  │TraceDataset │  │   CodeGenerator     │ │
│  │             │  │             │  │             │  │                     │ │
│  │ Manages the │  │ Undo/Redo   │  │ Holds data  │  │ Creates Python      │ │
│  │ Plotly      │  │ history     │  │ for each    │  │ code from the       │ │
│  │ figure      │  │ (50 states) │  │ trace       │  │ current figure      │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│                                                                              │
│  Location in app.py: Lines 210-817                                          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Classes operate on this data
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                                          │
│              (The Actual Data Being Stored)                                  │
│                                                                              │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐   │
│  │     Plotly Figure Dictionary    │  │     Session JSON File           │   │
│  │                                 │  │                                 │   │
│  │  {                              │  │  {                              │   │
│  │    "data": [                    │  │    "metadata": {...},           │   │
│  │      {type, x, y, marker...}    │  │    "figure": {...},             │   │
│  │    ],                           │  │    "datasets": {...},           │   │
│  │    "layout": {                  │  │    "dataset_order": [...]       │   │
│  │      title, xaxis, shapes...    │  │  }                              │   │
│  │    }                            │  │                                 │   │
│  │  }                              │  │  Saved to: session.json         │   │
│  └─────────────────────────────────┘  └─────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why Layered Architecture?**

| Benefit | Explanation | Example |
|---------|-------------|---------|
| **Separation of Concerns** | Each layer does ONE thing well | UI layer doesn't know how to save files |
| **Easier Testing** | Test each layer independently | Test `HistoryStack` without running the UI |
| **Easier Maintenance** | Change one layer without breaking others | Update UI colors without touching data logic |
| **Reusability** | Classes can be reused elsewhere | `CodeGenerator` could work in any Plotly app |

### 3.2 Core Class Design

> 💡 **Tips:** Classes are like "blueprints" for creating objects. Each class in PyFigureEditor has ONE specific job (this is called the "Single Responsibility Principle"). Let's examine each one.

The business logic is encapsulated in four primary classes, each with a single responsibility.

#### 3.2.1 FigureStore Class — The Central Hub

**Purpose:** Centralized management of the Plotly figure state. This is the MOST IMPORTANT class — everything revolves around it.

**Design Pattern:** Facade Pattern — provides a simplified interface to the complex Plotly figure structure.

**Think of it like:** A museum curator who manages all the artwork (traces, annotations, shapes) in an exhibition (the figure).

**Location in app.py:** Lines 240-520

**Actual Code from app.py:**

```python
class FigureStore:
    """Owns the current Plotly figure and its logical datasets."""

    def __init__(self, theme: str = "plotly_white") -> None:
        self.current_theme: str = theme
        self.figure: Optional[go.Figure] = None
        self.datasets: Dict[str, TraceDataset] = {}      # Stores trace data
        self.dataset_order: List[str] = []               # Order of traces
        self.data_repository: Dict[str, pd.DataFrame] = {}  # Uploaded DataFrames
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": None,
            "version": "1.0.0",
        }
        self._init_default_figure()  # Create initial demo figure
```

**Key Methods Explained:**

| Method | What It Does | When It's Called | Code Location |
|--------|--------------|------------------|---------------|
| `get_figure_dict()` | Returns the figure as a dictionary | When callbacks need current state | Line 350 |
| `update_figure(fig)` | Replaces the entire figure | After any modification | Line 355 |
| `add_dataset(key, name, df, ...)` | Adds a new trace with data | When user creates a chart | Line 360 |
| `remove_trace(index)` | Deletes a trace by index | When user clicks Delete | Line 405 |
| `remove_annotation(index)` | Deletes an annotation | When user deletes text | Line 420 |
| `remove_shape(index)` | Deletes a shape | When user deletes a rectangle | Line 435 |
| `serialize_session()` | Converts everything to JSON | When user clicks Save Session | Line 480 |
| `load_session(payload)` | Restores from JSON | When user loads a session | Line 500 |

**Detailed Code Analysis — `serialize_session()`:**

```python
def serialize_session(self) -> Dict[str, Any]:
    """Return a JSON-serializable snapshot of the current session.
    
    This method is called when the user clicks "Save Session".
    It creates a complete backup that can be saved to a file
    and loaded later to restore the exact state.
    """
    
    # Step 1: Package all datasets
    datasets_payload: Dict[str, Any] = {}
    for key, ds in self.datasets.items():
        datasets_payload[key] = {
            "name": ds.name,
            "color": ds.color,
            "line_width": ds.line_width,
            "marker_size": ds.marker_size,
            "visible": ds.visible,
            "chart_type": ds.chart_type,
            "df": ds.df.to_dict(orient="list"),  # Convert DataFrame to dict
        }

    # Step 2: Build the complete session object
    return {
        "metadata": copy.deepcopy(self.metadata),
        "current_theme": self.current_theme,
        "datasets": datasets_payload,
        "dataset_order": list(self.dataset_order),
        "figure": self.figure.to_dict() if self.figure is not None else None,
        "version": "1.0.0",
    }
```

**State Diagram — How FigureStore Changes Over Time:**

```
                    ┌─────────────────┐
                    │  INITIALIZED    │
                    │  (Demo figure)  │
                    └────────┬────────┘
                             │ User loads data / creates chart
                             ▼
                    ┌─────────────────┐
          ┌────────│  HAS TRACES     │────────┐
          │        │  (Working state)│        │
          │        └────────┬────────┘        │
          │                 │                 │
    add_annotation()   update_*()        add_shape()
          │                 │                 │
          ▼                 ▼                 ▼
    ┌───────────┐    ┌───────────┐    ┌───────────┐
    │   WITH    │    │  MODIFIED │    │   WITH    │
    │ANNOTATIONS│    │   PROPS   │    │  SHAPES   │
    └───────────┘    └───────────┘    └───────────┘
          │                 │                 │
          └─────────────────┼─────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │    COMPLEX      │
                   │ (Full editing)  │
                   └─────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
       serialize_session()          load_session()
              │                           │
              ▼                           ▼
       ┌─────────────┐            ┌─────────────┐
       │ JSON FILE   │───────────>│  RESTORED   │
       │  (saved)    │            │   STATE     │
       └─────────────┘            └─────────────┘
```

#### 3.2.2 HistoryStack Class — The Time Machine

**Purpose:** Implement undo/redo functionality. Every time you make a change, the previous state is saved. You can go back in time!

**Design Pattern:** Memento Pattern — captures and stores the internal state of an object so it can be restored later.

**Think of it like:** A "Time Machine" that remembers the last 50 versions of your chart.

**Location in app.py:** Lines 525-600

**Actual Code from app.py:**

```python
class HistoryStack:
    """Classic undo/redo stack for figure dictionaries."""

    def __init__(self, max_size: int = 50) -> None:
        self.max_size = max_size
        self.undo_stack: List[Dict[str, Any]] = []  # Past states
        self.redo_stack: List[Dict[str, Any]] = []  # Future states (after undo)

    def push(self, fig_dict: Optional[Dict[str, Any]]) -> None:
        """Save current state before making a change."""
        if fig_dict is None:
            return
        snapshot = copy.deepcopy(fig_dict)  # IMPORTANT: Deep copy!
        
        # Avoid duplicate consecutive states
        if self.undo_stack:
            last_state = json.dumps(self.undo_stack[-1], sort_keys=True, default=str)
            new_state = json.dumps(snapshot, sort_keys=True, default=str)
            if last_state == new_state:
                return  # Same state, don't save again
                
        self.undo_stack.append(snapshot)
        
        # Limit memory usage
        if len(self.undo_stack) > self.max_size:
            self.undo_stack.pop(0)  # Remove oldest
            
        self.redo_stack.clear()  # New action clears redo history

    def can_undo(self) -> bool:
        return len(self.undo_stack) > 1  # Need at least 2 states

    def can_redo(self) -> bool:
        return bool(self.redo_stack)

    def undo(self) -> Optional[Dict[str, Any]]:
        """Go back one step."""
        if not self.can_undo():
            return None
        current = self.undo_stack.pop()      # Remove current state
        self.redo_stack.append(current)      # Save it for redo
        return copy.deepcopy(self.undo_stack[-1])  # Return previous state

    def redo(self) -> Optional[Dict[str, Any]]:
        """Go forward one step (after undo)."""
        if not self.redo_stack:
            return None
        state = self.redo_stack.pop()
        self.undo_stack.append(copy.deepcopy(state))
        return copy.deepcopy(state)
```

**Visual Explanation — How Undo/Redo Works:**

```
SCENARIO: User creates chart, changes color, adds annotation, then wants to undo twice.

STEP 1: Initial state (empty)
┌─────────────────┐     ┌─────────────────┐
│   UNDO STACK    │     │   REDO STACK    │
│   (empty)       │     │   (empty)       │
└─────────────────┘     └─────────────────┘

STEP 2: User creates scatter chart → push(state)
┌─────────────────┐     ┌─────────────────┐
│   UNDO STACK    │     │   REDO STACK    │
│   [S0: scatter] │     │   (empty)       │
└─────────────────┘     └─────────────────┘

STEP 3: User changes color to red → push(state)
┌─────────────────┐     ┌─────────────────┐
│   UNDO STACK    │     │   REDO STACK    │
│   [S0: scatter] │     │   (empty)       │
│   [S1: red]     │     │                 │
└─────────────────┘     └─────────────────┘

STEP 4: User adds annotation → push(state)
┌─────────────────┐     ┌─────────────────┐
│   UNDO STACK    │     │   REDO STACK    │
│   [S0: scatter] │     │   (empty)       │
│   [S1: red]     │     │                 │
│   [S2: +annot]  │     │                 │
└─────────────────┘     └─────────────────┘

STEP 5: User clicks UNDO → undo()
┌─────────────────┐     ┌─────────────────┐
│   UNDO STACK    │     │   REDO STACK    │
│   [S0: scatter] │     │   [S2: +annot]  │  ← S2 moved here
│   [S1: red]     │     │                 │
└─────────────────┘     └─────────────────┘
Result: Chart shows S1 (red, no annotation)

STEP 6: User clicks UNDO again → undo()
┌─────────────────┐     ┌─────────────────┐
│   UNDO STACK    │     │   REDO STACK    │
│   [S0: scatter] │     │   [S2: +annot]  │
│                 │     │   [S1: red]     │  ← S1 moved here
└─────────────────┘     └─────────────────┘
Result: Chart shows S0 (original blue scatter)

STEP 7: User clicks REDO → redo()
┌─────────────────┐     ┌─────────────────┐
│   UNDO STACK    │     │   REDO STACK    │
│   [S0: scatter] │     │   [S2: +annot]  │
│   [S1: red]     │ ←   │                 │  ← S1 moved back
└─────────────────┘     └─────────────────┘
Result: Chart shows S1 (red)

STEP 8: User makes NEW change (adds shape) → push(state)
┌─────────────────┐     ┌─────────────────┐
│   UNDO STACK    │     │   REDO STACK    │
│   [S0: scatter] │     │   (CLEARED!)    │  ← Redo stack cleared!
│   [S1: red]     │     │                 │
│   [S3: +shape]  │     │                 │
└─────────────────┘     └─────────────────┘
Note: S2 is LOST forever — this is standard undo/redo behavior!
```

**Why Deep Copy is Critical:**

```python
# BAD: Without deep copy
self.undo_stack.append(fig_dict)
# Problem: fig_dict is a reference! If the figure changes, 
# the "saved" state changes too — your history is corrupted!

# GOOD: With deep copy
self.undo_stack.append(copy.deepcopy(fig_dict))
# Now we have an independent copy that won't change
```

#### 3.2.3 TraceDataset Class — The Data Container

**Purpose:** Encapsulate data and styling for a single trace (data series). Think of it as a "package" containing everything needed to draw one line/bar/scatter.

**Design Pattern:** Data Transfer Object (DTO) — a simple container for related data.

**Think of it like:** A labeled box containing all the information for one item on your chart.

**Location in app.py:** Lines 180-240

**Actual Code from app.py:**

```python
@dataclass
class TraceDataset:
    """Container for a single logical plot layer."""
    
    key: str                    # Unique identifier (e.g., "trace_1")
    name: str                   # Display name (e.g., "Sales Data")
    df: pd.DataFrame            # The actual data (x, y columns)
    color: str = "#1f77b4"      # Default blue
    line_width: float = 2.5     # Line thickness
    marker_size: float = 6.0    # Point size
    visible: bool = True        # Show or hide
    chart_type: str = "scatter" # scatter, bar, line, etc.

    def to_plotly_trace(self):
        """Convert this dataset into a Plotly trace object."""
        
        # Get x and y data from DataFrame
        x = self.df['x'] if 'x' in self.df.columns else None
        y = self.df['y'] if 'y' in self.df.columns else None
        
        # Create appropriate trace type
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
                x=x, y=y,
                mode="lines+markers",
                name=self.name,
            )

        # Apply styling
        if hasattr(trace, "marker"):
            trace.update(marker=dict(size=self.marker_size, color=self.color))
        if hasattr(trace, "line"):
            trace.update(line=dict(width=self.line_width, color=self.color))
        
        trace.visible = True if self.visible else "legendonly"
        return trace
```

**Example Usage:**

```python
# Create a dataset
dataset = TraceDataset(
    key="sales_2024",
    name="Sales 2024",
    df=pd.DataFrame({"x": [1,2,3,4], "y": [100,150,120,180]}),
    color="#ff6b6b",
    line_width=3,
    marker_size=10,
    chart_type="scatter"
)

# Convert to Plotly trace
trace = dataset.to_plotly_trace()
# Now trace is a go.Scatter object ready to add to a figure!
```

#### 3.2.4 CodeGenerator Class — The Code Writer

**Purpose:** Generate Python code that recreates the current figure. This is the "magic" that lets users export their work as reproducible code.

**Design Pattern:** Template Method Pattern — defines the skeleton of code generation.

**Think of it like:** A robot that watches you edit a chart and writes down the Python code to recreate it.

**Location in app.py:** Lines 640-800

**Actual Code from app.py:**

```python
class CodeGenerator:
    """Turn the current figure into runnable Python code."""

    def generate_code(self, store: FigureStore) -> str:
        """Generate complete Python code to recreate the figure."""
        
        if store.figure is None:
            return "# No figure available yet. Interact with the editor first."

        # Get the complete JSON representation
        fig_json = store.figure.to_json()

        # Build the code
        lines: List[str] = []
        lines.append("# Auto-generated by Python Interactive Figure Editor")
        lines.append("# Recreate the current figure exactly as seen in the UI.")
        lines.append("import json")
        lines.append("import plotly.graph_objects as go")
        lines.append("")
        lines.append(f"fig_dict = json.loads({fig_json!r})")
        lines.append("fig = go.Figure(fig_dict)")
        lines.append("")
        lines.append("# Show the figure")
        lines.append("fig.show()")
        lines.append("")
        lines.append("# Tip: you can now modify `fig` programmatically, e.g.:")
        lines.append("# fig.update_layout(title='My Edited Figure')")
        
        return "\n".join(lines)
```

**Smart Plot Code Generation:**

The `CodeGenerator` also has a `generate_smart_plot_code()` method that generates cleaner code based on the data:

```python
def generate_smart_plot_code(self, df_name: str, plot_type: str, df: pd.DataFrame) -> str:
    """Generate Plotly code with smart column selection.
    
    This method analyzes the DataFrame to pick appropriate columns:
    - Numeric columns for x, y, z, size
    - Categorical columns for color, grouping
    """
    
    # Analyze DataFrame columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    
    # Smart column selection based on plot type
    if plot_type == 'scatter':
        x_col = num_cols[0] if num_cols else df.columns[0]
        y_col = num_cols[1] if len(num_cols) > 1 else num_cols[0]
        color_col = cat_cols[0] if cat_cols else None
        
        code = f"fig = px.scatter({df_name}, x='{x_col}', y='{y_col}'"
        if color_col:
            code += f", color='{color_col}'"
        code += ")"
        
    elif plot_type == 'bar':
        # Use category for x, numeric for y
        x_col = cat_cols[0] if cat_cols else df.columns[0]
        y_col = num_cols[0] if num_cols else df.columns[1]
        code = f"fig = px.bar({df_name}, x='{x_col}', y='{y_col}')"
        
    # ... more plot types ...
    
    return code
```

**Generated Code Example:**

When you create a scatter plot, change the color to red, and add an annotation, the generated code looks like:

```python
# Auto-generated by Python Interactive Figure Editor
import json
import plotly.graph_objects as go

fig_dict = json.loads('{"data":[{"type":"scatter","x":[1,2,3,4,5],"y":[2,4,1,5,3],"mode":"lines+markers","name":"My Data","marker":{"color":"red","size":10},"line":{"color":"red","width":2}}],"layout":{"title":{"text":"My Chart"},"annotations":[{"text":"Important Point","x":3,"y":5,"showarrow":true}],"template":"plotly_white"}}')

fig = go.Figure(fig_dict)

# Show the figure
fig.show()

# Tip: you can now modify `fig` programmatically, e.g.:
# fig.update_layout(title='My Edited Figure')
```

### 3.3 Component Interaction Diagram

> 💡 **Tips:** This diagram shows what happens "behind the scenes" when you click a button. Each arrow represents a function call or data transfer.

**Sequence Diagram: User Changes Trace Color**

```
 USER          BROWSER         CALLBACK         FigureStore      HistoryStack
  │               │               │                  │                │
  │ Click "Apply" │               │                  │                │
  │──────────────>│               │                  │                │
  │               │  HTTP POST    │                  │                │
  │               │──────────────>│                  │                │
  │               │               │                  │                │
  │               │               │  1. Get current figure           │
  │               │               │─────────────────>│                │
  │               │               │<─────────────────│                │
  │               │               │                  │                │
  │               │               │  2. Push to history (backup)     │
  │               │               │─────────────────────────────────>│
  │               │               │<─────────────────────────────────│
  │               │               │                  │                │
  │               │               │  3. Apply color change           │
  │               │               │  fig.data[0].marker.color = "red"│
  │               │               │                  │                │
  │               │               │  4. Update figure                │
  │               │               │─────────────────>│                │
  │               │               │<─────────────────│                │
  │               │               │                  │                │
  │               │  JSON Response│                  │                │
  │               │<──────────────│                  │                │
  │ Updated Chart │               │                  │                │
  │<──────────────│               │                  │                │
  │               │               │                  │                │
  
TIME: ~50-100ms total (feels instant!)
```

### 3.4 Layout Architecture

> 💡 **Tips:** The visual layout uses a "grid system" where the screen is divided into rows and columns. This is the same concept used by Bootstrap CSS framework.

**The PyFigureEditor Screen Layout:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RIBBON TOOLBAR                                  │
│  ┌─────────┬─────────┬─────────┬───────────┬─────────┐                      │
│  │  HOME   │  DATA   │  PLOTS  │ ANNOTATE  │  VIEW   │     [GitHub Link]   │
│  └─────────┴─────────┴─────────┴───────────┴─────────┘                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Tab Content: Buttons change based on selected tab                   │    │
│  │ HOME: [Open] [Save] [Undo] [Redo] [About]                          │    │
│  │ DATA: [Import CSV] [Load Demo] [Select Data] [Clean] [Summary]     │    │
│  │ PLOTS: [Scatter] [Line] [Bar] ... (26+ buttons)                    │    │
│  │ ANNOTATE: [Line] [Rect] [Circle] [Text] [Image]                    │    │
│  │ VIEW: [Zoom] [Pan] [Reset] [Inspector Toggle]                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────┬───────────────────────────────┤
│                                             │                               │
│                                             │      PROPERTY INSPECTOR       │
│                                             │  ┌─────────────────────────┐  │
│                                             │  │ Element: [Dropdown ▼]  │  │
│           MAIN GRAPH AREA                   │  ├─────────────────────────┤  │
│                                             │  │                         │  │
│           (dcc.Graph component)             │  │  Name:  [___________]  │  │
│                                             │  │  Color: [___________]  │  │
│           • Shows the Plotly chart          │  │  Size:  [___________]  │  │
│           • Interactive (zoom, pan, hover)  │  │  Style: [Dropdown ▼]  │  │
│           • Receives figure from callbacks  │  │  Opacity: [___________]│  │
│                                             │  │                         │  │
│           Width: ~70% of screen             │  │  [Apply] [Reset]       │  │
│                                             │  │  [Delete Element]      │  │
│                                             │  └─────────────────────────┘  │
│                                             │                               │
│                                             │      Width: ~30% of screen   │
├─────────────────────────────────────────────┴───────────────────────────────┤
│                           COMMAND WINDOW                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ # Auto-generated code appears here                                   │    │
│  │ fig = px.scatter(df, x='x', y='y')                                  │    │
│  │ fig.update_traces(marker_color='red')                               │    │
│  │                                                           [Copy] 📋  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘

RESPONSIVE DESIGN:
- On wide screens (>1200px): Side-by-side layout as shown
- On narrow screens (<768px): Inspector moves below graph
```

**Code Structure for Layout (Simplified):**

```python
app.layout = dbc.Container([
    # Row 1: Ribbon
    ribbon,  # The tab bar and buttons
    
    # Row 2: Main content (graph + inspector)
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="main-graph", figure=initial_figure, config={...})
        ], lg=8, md=12),  # 8/12 columns on large, full on medium
        
        dbc.Col([
            inspector_panel  # Property editor
        ], lg=4, md=12),  # 4/12 columns on large, full on medium
    ]),
    
    # Row 3: Command window
    command_window,
    
    # Hidden components
    dcc.Store(id="store-history", data={}),  # Hidden storage for undo history
    
    # Modals (pop-up dialogs)
    about_modal,
    save_modal,
    # etc.
    
], fluid=True)
```

### 3.5 Data Flow Architecture

> 💡 **Tips:** This shows how data "flows" through the application — from user input to visual output. Understanding this flow is key to understanding how the app works.

**Complete Data Flow Diagram:**

```
                              USER INPUT
                                  │
               ┌──────────────────┼──────────────────┐
               │                  │                  │
               ▼                  ▼                  ▼
         CSV Upload         Button Click        Drawing Action
         (file data)        (n_clicks)          (relayoutData)
               │                  │                  │
               ▼                  ▼                  ▼
    ┌──────────────────────────────────────────────────────────┐
    │                  INPUT VALIDATION                         │
    │  ┌────────────────────────────────────────────────────┐  │
    │  │ • Check file format (CSV, JSON)                    │  │
    │  │ • Validate data types (numbers, strings)           │  │
    │  │ • Check required columns exist                     │  │
    │  │ • Convert to appropriate Python types              │  │
    │  └────────────────────────────────────────────────────┘  │
    │                                                          │
    │  If invalid: Show error message, stop processing         │
    └──────────────────────────────────────────────────────────┘
                                  │
                                  │ Valid data
                                  ▼
    ┌──────────────────────────────────────────────────────────┐
    │                  STATE UPDATE                             │
    │  ┌────────────────────────────────────────────────────┐  │
    │  │ 1. history_stack.push(current_state)  # Backup     │  │
    │  │ 2. figure_store.update_figure(new_fig) # Change    │  │
    │  │ 3. code_generator.generate_code()     # Update code│  │
    │  └────────────────────────────────────────────────────┘  │
    └──────────────────────────────────────────────────────────┘
                                  │
                                  │ New figure state
                                  ▼
    ┌──────────────────────────────────────────────────────────┐
    │                  UI REFRESH                               │
    │  ┌────────────────────────────────────────────────────┐  │
    │  │ Callback returns trigger AUTOMATIC updates:        │  │
    │  │                                                    │  │
    │  │ • main-graph.figure      → Chart re-renders        │  │
    │  │ • dd-element.options     → Dropdown updates        │  │
    │  │ • inspector.children     → Property panel updates  │  │
    │  │ • btn-undo.disabled      → Undo button enables     │  │
    │  │ • command-window.value   → Code updates            │  │
    │  └────────────────────────────────────────────────────┘  │
    └──────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                           VISUAL OUTPUT
                    (User sees updated chart!)
```

### 3.6 Design Decisions and Trade-offs

> 💡 **Tips:** Every engineering decision involves trade-offs. There's no "perfect" solution — only solutions that are better for specific situations. Here's why we made certain choices.

| Decision | What We Chose | Why | Trade-off |
|----------|---------------|-----|-----------|
| **Architecture** | Single-page application (SPA) | Simpler state management, no page reloads | Limited URL routing (can't bookmark specific charts) |
| **State Storage** | Store in callbacks (not database) | No database setup needed, works offline | Limited scalability (single user sessions) |
| **Undo Implementation** | Full figure copy per state | Simple to implement, 100% reliable | Higher memory usage (~50 states × figure size) |
| **Validation** | Client-side (in browser) | Faster user feedback, less server load | Need to duplicate some validation logic |
| **Authentication** | None (single user) | Simplifies deployment, no login needed | Can't share sessions between users |
| **Session Format** | JSON file | Human-readable, easy to debug, portable | Larger file size than binary formats |
| **Deployment** | PythonAnywhere | Free tier available, Python-native | Limited customization compared to Docker |

**Why These Trade-offs Make Sense for This Project:**

1. **Educational Focus:** Simpler architecture is easier to understand and teach
2. **Single-User Tool:** Like MATLAB's Figure Tool, designed for one person at a time
3. **Portability:** JSON sessions can be shared via email or Git
4. **Development Speed:** Avoiding complex infrastructure lets us focus on features

### 3.7 Extensibility Points

> 💡 **Tips:** Good software is designed to be extended without rewriting everything. These are the "hooks" where you can add new features.

**Where to Add New Features:**

| Extension Type | Location | How to Extend |
|----------------|----------|---------------|
| **New Chart Type** | `PLOTS` tab buttons + callback | Add button in ribbon, add case in `generate_smart_plot_code()` |
| **New Property** | Inspector panel | Add to property extraction logic, add input component |
| **New Export Format** | `CodeGenerator` class | Add new method like `generate_matplotlib_code()` |
| **New Theme** | Theme dropdown | Add to `TEMPLATE_OPTIONS` list |
| **New Drawing Tool** | `ANNOTATE` tab | Add button, set appropriate `dragmode` |

**Example: Adding a New Chart Type (Radar Chart)**

```python
# Step 1: Add button in PLOTS tab (ribbon definition)
dbc.Button("Radar", id="btn-plot-radar", color="outline-secondary", size="sm")

# Step 2: Add to callback Input list
Input("btn-plot-radar", "n_clicks"),

# Step 3: Add case in generate_smart_plot_code()
elif plot_type == 'radar':
    categories = cat_cols[0] if cat_cols else df.columns[0]
    values = num_cols[0] if num_cols else df.columns[1]
    cmd = f"fig = go.Figure(go.Scatterpolar(r={df_name}['{values}'], theta={df_name}['{categories}'], fill='toself'))"

# Step 4: Handle in callback
if ctx.triggered_id == "btn-plot-radar":
    plot_type = "radar"
```

---

## 4. Core Implementation Details

> 💡 **Tips:** This is the most technical chapter - it explains HOW the code actually works. Don't worry if you don't understand everything at first! Read through it once, then come back when you're working on specific features.

This chapter provides an in-depth examination of the actual implementation, with extensive code analysis and explanations suitable for understanding every aspect of the system.

### 4.1 Application Initialization - The "Birth" of Your App

> 💡 **Tips:** This section explains how the application "wakes up" when you run it. Think of it like starting a car - there's a specific sequence of things that must happen before you can drive!

#### 4.1.1 Understanding Auto-Dependency Installation

One clever feature of `app.py` is that it **automatically installs missing libraries** when you first run it. This is unusual but very user-friendly:

```python
# app.py (Lines 1-35) - Auto-Installation System
# =============================================================================
# Auto-Install Dependencies if Missing (Convenience Feature)
# =============================================================================

REQUIRED_PACKAGES = [
    ("dash", "dash"),
    ("dash_bootstrap_components", "dash-bootstrap-components"),
    ("plotly", "plotly"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
]

import subprocess
import sys

for import_name, pip_name in REQUIRED_PACKAGES:
    try:
        __import__(import_name)           # Try to import the package
    except ImportError:
        print(f"📦 Installing {pip_name}...")
        subprocess.check_call([           # If not found, install it!
            sys.executable, "-m", "pip", 
            "install", pip_name
        ])
```

**How This Works (Step by Step):**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AUTO-INSTALL FLOW                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  For each required package:                                         │
│                                                                     │
│  ┌──────────────────┐     Success    ┌─────────────────────┐       │
│  │ Try to import    │ ─────────────► │ Package OK!         │       │
│  │ the package      │                │ Move to next        │       │
│  └──────────────────┘                └─────────────────────┘       │
│          │                                                          │
│          │ ImportError (not installed)                              │
│          ▼                                                          │
│  ┌──────────────────┐                ┌─────────────────────┐       │
│  │ Run pip install  │ ─────────────► │ Now import it       │       │
│  │ automatically    │                │ and continue        │       │
│  └──────────────────┘                └─────────────────────┘       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

> 💡 **Tips:** This is like a restaurant that checks if ingredients are available before cooking. If something is missing, it sends someone to buy it first!

---

#### 4.1.2 Import Statements - Loading Your Toolbox

After dependencies are ensured, the app loads all necessary libraries:

```python
# app.py (Lines 40-75) - Core Imports
# =============================================================================
# IMPORTS
# =============================================================================
import dash
from dash import dcc, html, Input, Output, State, ctx
from dash.exceptions import PreventUpdate
from dash import dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import json
import base64
import io
import uuid
from collections import defaultdict
from typing import List, Dict, Any, Optional, cast
```

**Import Analysis Table - What Each Library Does:**

| Import | Real Name | Purpose | Where It's Used |
|--------|-----------|---------|-----------------|
| `dash` | Dash Core | The main framework that makes the app work | Creating the app, routing |
| `dcc` | Dash Core Components | Pre-built interactive widgets | Graph, Dropdown, Store, Upload |
| `html` | Dash HTML Components | Basic HTML elements as Python | Div, H1, Button, Span |
| `dbc` | Dash Bootstrap Components | Beautiful styled components | Card, Row, Col, Modal, Tabs |
| `go` | Plotly Graph Objects | Low-level chart building blocks | Scatter, Bar, Figure, Surface |
| `px` | Plotly Express | Quick chart creation | One-liner charts |
| `np` | NumPy | Fast numerical operations | Array math, linspace, random |
| `pd` | Pandas | Data manipulation | DataFrame, read_csv, describe |
| `json` | JSON Library | Save/load structured data | Session export/import |
| `base64` | Base64 Encoding | Handle binary data as text | Image upload, CSV upload |
| `ctx` | Callback Context | Know which button was clicked | Multi-input callbacks |
| `PreventUpdate` | Update Prevention | Stop a callback from doing anything | Guard clauses |

> 💡 **Tips:** Think of imports like unpacking a toolbox before starting a project. Each tool (library) has a specific purpose. You wouldn't use a hammer (NumPy) to paint a wall (make buttons), right?

---

#### 4.1.3 Application Creation - The "Main Engine"

```python
# app.py (Lines 80-95) - App Initialization
# =============================================================================
# APP INITIALIZATION
# =============================================================================
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,           # Beautiful Bootstrap styling
        dbc.icons.FONT_AWESOME          # Icons like 📊 📈 🗑️
    ],
    suppress_callback_exceptions=True   # Allow dynamic component IDs
)

server = app.server  # WSGI server for deployment
```

**What Each Parameter Does:**

| Parameter | Value | Why It's Needed |
|-----------|-------|-----------------|
| `__name__` | Module name | Helps Dash find static files |
| `external_stylesheets` | Bootstrap + Icons | Makes the app look professional |
| `suppress_callback_exceptions` | `True` | Allows callbacks for components that don't exist yet |
| `server` | WSGI object | Required for PythonAnywhere/Heroku deployment |

**Visual: What `suppress_callback_exceptions=True` Does:**

```
WITHOUT suppress_callback_exceptions=True:
┌─────────────────────────────────────────────────────────────────────┐
│  Callback for "inspector-controls" declared                         │
│          │                                                          │
│          ▼                                                          │
│  🔍 Dash checks: Does "inspector-controls" exist in layout?         │
│          │                                                          │
│          ▼ (Component is dynamically created later)                 │
│  ❌ ERROR: "Component with id 'inspector-controls' not found!"      │
└─────────────────────────────────────────────────────────────────────┘

WITH suppress_callback_exceptions=True:
┌─────────────────────────────────────────────────────────────────────┐
│  Callback for "inspector-controls" declared                         │
│          │                                                          │
│          ▼                                                          │
│  🔍 Dash checks: Does "inspector-controls" exist?                   │
│          │                                                          │
│          ▼ (Not found, but that's OK!)                              │
│  ✅ Continue - we'll find it when the component is created          │
└─────────────────────────────────────────────────────────────────────┘
```

> 💡 **Tips:** It's like writing instructions for "the new employee" before they're hired. With `suppress_callback_exceptions`, Dash trusts that the component will exist when needed!
| `callback_context` | Trigger detection | Multi-input callbacks |

**Application Creation:**

```python
# =============================================================================
# APP INITIALIZATION
# =============================================================================
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,           # Bootstrap CSS
        dbc.icons.FONT_AWESOME          # Font Awesome icons
    ],
    suppress_callback_exceptions=True   # Required for dynamic callbacks
)

server = app.server  # For WSGI deployment (PythonAnywhere, Heroku)
```

**Configuration Constants:**

```python
# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Supported trace types with their display names and Plotly constructors
TRACE_TYPE_OPTIONS = {
    # Basic Charts
    'scatter': {'label': 'Scatter', 'constructor': go.Scatter},
    'line': {'label': 'Line', 'constructor': go.Scatter, 'mode': 'lines'},
    'bar': {'label': 'Bar', 'constructor': go.Bar},
    'area': {'label': 'Area', 'constructor': go.Scatter, 'fill': 'tozeroy'},
    
    # Statistical Charts
    'histogram': {'label': 'Histogram', 'constructor': go.Histogram},
    'box': {'label': 'Box Plot', 'constructor': go.Box},
    'violin': {'label': 'Violin', 'constructor': go.Violin},
    'heatmap': {'label': 'Heatmap', 'constructor': go.Heatmap},
    
    # 3D Charts
    'scatter3d': {'label': '3D Scatter', 'constructor': go.Scatter3d},
    'surface': {'label': '3D Surface', 'constructor': go.Surface},
    'mesh3d': {'label': '3D Mesh', 'constructor': go.Mesh3d},
    
    # Specialized Charts
    'pie': {'label': 'Pie', 'constructor': go.Pie},
    'funnel': {'label': 'Funnel', 'constructor': go.Funnel},
    'waterfall': {'label': 'Waterfall', 'constructor': go.Waterfall},
    'candlestick': {'label': 'Candlestick', 'constructor': go.Candlestick},
    'ohlc': {'label': 'OHLC', 'constructor': go.Ohlc},
    
    # Geographic Charts
    'scattergeo': {'label': 'Geo Scatter', 'constructor': go.Scattergeo},
    'choropleth': {'label': 'Choropleth', 'constructor': go.Choropleth},
    'scattermapbox': {'label': 'Mapbox Scatter', 'constructor': go.Scattermapbox},
    
    # And many more...
}

# Available color templates
TEMPLATE_OPTIONS = [
    'plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn',
    'simple_white', 'presentation', 'xgridoff', 'ygridoff', 'gridon'
]

# Editable properties organized by category
EDITABLE_PROPERTIES = {
    'trace': {
        'name': {'type': 'text', 'label': 'Name'},
        'visible': {'type': 'dropdown', 'label': 'Visible', 
                    'options': [True, False, 'legendonly']},
        'opacity': {'type': 'number', 'label': 'Opacity', 'min': 0, 'max': 1, 'step': 0.1},
        # Line properties
        'line.color': {'type': 'color', 'label': 'Line Color'},
        'line.width': {'type': 'number', 'label': 'Line Width', 'min': 0, 'max': 20},
        'line.dash': {'type': 'dropdown', 'label': 'Line Style',
                      'options': ['solid', 'dot', 'dash', 'longdash', 'dashdot']},
        # Marker properties
        'marker.color': {'type': 'color', 'label': 'Marker Color'},
        'marker.size': {'type': 'number', 'label': 'Marker Size', 'min': 1, 'max': 50},
        'marker.symbol': {'type': 'dropdown', 'label': 'Marker Symbol',
                          'options': ['circle', 'square', 'diamond', 'cross', 'x', 
                                     'triangle-up', 'triangle-down', 'star', 'hexagon']},
    },
    'layout': {
        'title.text': {'type': 'text', 'label': 'Title'},
        'xaxis.title.text': {'type': 'text', 'label': 'X Axis Title'},
        'yaxis.title.text': {'type': 'text', 'label': 'Y Axis Title'},
        'showlegend': {'type': 'checkbox', 'label': 'Show Legend'},
        'template': {'type': 'dropdown', 'label': 'Template', 'options': TEMPLATE_OPTIONS},
    },
    'annotation': {
        'text': {'type': 'text', 'label': 'Text'},
        'x': {'type': 'number', 'label': 'X Position'},
        'y': {'type': 'number', 'label': 'Y Position'},
        'font.size': {'type': 'number', 'label': 'Font Size', 'min': 8, 'max': 72},
        'font.color': {'type': 'color', 'label': 'Font Color'},
        'showarrow': {'type': 'checkbox', 'label': 'Show Arrow'},
        'arrowhead': {'type': 'number', 'label': 'Arrow Head', 'min': 0, 'max': 8},
    },
    'shape': {
        'type': {'type': 'dropdown', 'label': 'Shape Type',
                 'options': ['rect', 'circle', 'line', 'path']},
        'line.color': {'type': 'color', 'label': 'Line Color'},
        'line.width': {'type': 'number', 'label': 'Line Width'},
        'fillcolor': {'type': 'color', 'label': 'Fill Color'},
        'opacity': {'type': 'number', 'label': 'Opacity', 'min': 0, 'max': 1},
    }
}
```

---

### 4.2 Layout Implementation - Building the User Interface

> 💡 **Tips:** The "layout" is like the blueprint of a house. It defines where each room (component) goes, but doesn't define what happens when you flip a light switch (that's callbacks!).

The application layout is built using Dash components organized in a hierarchical structure. Let's examine the ACTUAL code from `app.py`:

#### 4.2.1 Main Layout Structure - The "Blueprint"

```python
# app.py (Lines 1200-1225) - Main Layout
app.layout = dbc.Container([
    # Hidden state storage components (invisible to user)
    dcc.Store(id='figure-store-client', data=None),    # Current figure data
    dcc.Store(id='data-update-signal', data=0),        # Triggers data refresh
    dcc.Store(id='trigger-run-signal', data=0),        # Triggers code execution
    dcc.Download(id='download-component'),             # File download handler
    
    # Main visible structure
    dbc.Row([
        # Header with ribbon tabs
        dbc.Col(create_ribbon_tabs(), width=12)
    ]),
    
    dbc.Row([
        # Left: Workspace (Command Window + Data View)
        dbc.Col(workspace_panel, width=3),
        
        # Center: The Main Canvas (Graph)
        dbc.Col([
            dcc.Graph(id='main-graph', figure=create_initial_figure(), ...)
        ], width=6),
        
        # Right: Property Inspector
        dbc.Col(property_inspector, width=3),
    ]),
    
    # Hidden modals (pop-ups)
    annotation_modal,
    about_modal,
    
], fluid=True, style={"height": "100vh"})
```

**Visual: The Three-Column Layout**

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        RIBBON TABS (HOME | DATA | PLOTS | ANNOTATE | VIEW)│
├────────────────┬─────────────────────────────────────┬──────────────────────┤
│                │                                     │                      │
│   WORKSPACE    │          MAIN CANVAS               │  PROPERTY           │
│    PANEL       │                                     │   INSPECTOR         │
│                │      ┌─────────────────────┐       │                      │
│  ┌──────────┐  │      │                     │       │  Element: [Trace 0]  │
│  │ Command  │  │      │    YOUR CHART       │       │                      │
│  │ Window   │  │      │    APPEARS HERE     │       │  Color: [  ▼  ]     │
│  │          │  │      │                     │       │  Size:  [   10  ]   │
│  │ >>> code │  │      │   📊 📈 📉          │       │  Style: [  ▼  ]     │
│  └──────────┘  │      │                     │       │                      │
│                │      └─────────────────────┘       │  [Apply Changes]     │
│  ┌──────────┐  │                                     │  [Delete Element]    │
│  │ Data     │  │                                     │                      │
│  │ View     │  │                                     │                      │
│  └──────────┘  │                                     │                      │
│                │                                     │                      │
│  width=3 (25%) │     width=6 (50%)                  │   width=3 (25%)      │
└────────────────┴─────────────────────────────────────┴──────────────────────┘
```

> 💡 **Tips:** Bootstrap uses a 12-column grid system. So `width=3` means 3/12 = 25% of the screen width, and `width=6` means 50%.

---

#### 4.2.2 The Ribbon Tabs - ACTUAL Code from app.py

The ribbon is like Microsoft Office's ribbon - it organizes tools into categories:

```python
# app.py (Lines 900-1050) - Ribbon Implementation
ribbon_home = dbc.Card([
    dbc.CardBody([
        dbc.ButtonGroup([
            # Session Management
            dbc.Button([html.I(className="fas fa-save me-1"), "Save"], 
                      id="btn-save-session", color="outline-primary", size="sm"),
            dcc.Upload(id="upload-session", children=
                dbc.Button([html.I(className="fas fa-folder-open me-1"), "Load"],
                          color="outline-primary", size="sm"),
            ),
        ], className="me-3"),
        
        # Undo/Redo
        dbc.ButtonGroup([
            dbc.Button([html.I(className="fas fa-undo")], 
                      id="btn-undo", color="outline-secondary", size="sm"),
            dbc.Button([html.I(className="fas fa-redo")], 
                      id="btn-redo", color="outline-secondary", size="sm"),
        ], className="me-3"),
        
        # About
        dbc.Button([html.I(className="fas fa-info-circle me-1"), "About"],
                  id="btn-open-about", color="outline-info", size="sm"),
    ], className="py-1")
], className="border-0 bg-transparent")
```

**Ribbon Tab Organization (from app.py):**

| Tab | Components | Purpose |
|-----|------------|---------|
| **HOME** | Save, Load, Undo, Redo, About | Session management |
| **DATA** | Import CSV, Load Demo, Delete, Clean NA, View Table/Stats/Types | Data operations |
| **PLOTS** | 26+ chart type buttons (Scatter, Line, Bar, Pie, 3D...) | Create visualizations |
| **ANNOTATE** | Draw Line/Rect/Circle, Add Text, Upload Image | Add annotations |
| **VIEW** | Zoom, Pan, Reset, Inspector Toggle | Navigation tools |

---

#### 4.2.3 The PLOTS Tab - 26+ Chart Types in One Place

```python
# app.py (Lines 1000-1100) - PLOTS Tab with ALL Chart Types
ribbon_plots = dbc.Card([
    dbc.CardBody([
        # Basic 2D Charts
        html.Span("Basic:", className="text-muted small me-2"),
        dbc.ButtonGroup([
            dbc.Button("Scatter", id="btn-plot-scatter", color="outline-secondary", size="sm"),
            dbc.Button("Line", id="btn-plot-line", color="outline-secondary", size="sm"),
            dbc.Button("Bar", id="btn-plot-bar", color="outline-secondary", size="sm"),
            dbc.Button("Area", id="btn-plot-area", color="outline-secondary", size="sm"),
            dbc.Button("Bubble", id="btn-plot-bubble", color="outline-secondary", size="sm"),
        ], className="me-3"),
        
        # Distribution Charts
        html.Span("Dist:", className="text-muted small me-2"),
        dbc.ButtonGroup([
            dbc.Button("Pie", id="btn-plot-pie", color="outline-secondary", size="sm"),
            dbc.Button("Hist", id="btn-plot-hist", color="outline-secondary", size="sm"),
            dbc.Button("Box", id="btn-plot-box", color="outline-secondary", size="sm"),
            dbc.Button("Violin", id="btn-plot-violin", color="outline-secondary", size="sm"),
        ], className="me-3"),
        
        # 3D & Contour Charts
        html.Span("3D:", className="text-muted small me-2"),
        dbc.ButtonGroup([
            dbc.Button("3D Scatter", id="btn-plot-scatter3d", color="outline-secondary", size="sm"),
            dbc.Button("Line 3D", id="btn-plot-line3d", color="outline-secondary", size="sm"),
            dbc.Button("Surface", id="btn-plot-surface", color="outline-secondary", size="sm"),
            dbc.Button("Contour", id="btn-plot-contour", color="outline-secondary", size="sm"),
        ], className="me-3"),
        
        # Specialized Charts
        html.Span("Special:", className="text-muted small me-2"),
        dbc.ButtonGroup([
            dbc.Button("Sunburst", id="btn-plot-sunburst", color="outline-secondary", size="sm"),
            dbc.Button("Treemap", id="btn-plot-treemap", color="outline-secondary", size="sm"),
            dbc.Button("Heatmap", id="btn-plot-heatmap", color="outline-secondary", size="sm"),
            dbc.Button("Polar", id="btn-plot-polar", color="outline-secondary", size="sm"),
            dbc.Button("Candle", id="btn-plot-candle", color="outline-secondary", size="sm"),
            dbc.Button("Waterfall", id="btn-plot-waterfall", color="outline-secondary", size="sm"),
        ], className="me-3"),
        
        # Geographic Charts
        html.Span("Geo:", className="text-muted small me-2"),
        dbc.ButtonGroup([
            dbc.Button("Geo", id="btn-plot-scatgeo", color="outline-secondary", size="sm"),
            dbc.Button("Choro", id="btn-plot-choropleth", color="outline-secondary", size="sm"),
            dbc.Button("Globe", id="btn-plot-globe", color="outline-secondary", size="sm"),
        ]),
    ], className="py-1")
], className="border-0 bg-transparent")
```

**Chart Type Categorization:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         26+ CHART TYPES                                 │
├─────────────┬───────────────────────────────────────────────────────────┤
│ CATEGORY    │ CHART TYPES                                               │
├─────────────┼───────────────────────────────────────────────────────────┤
│ Basic 2D    │ Scatter • Line • Bar • Area • Bubble                      │
├─────────────┼───────────────────────────────────────────────────────────┤
│ Distribution│ Pie • Histogram • Box Plot • Violin                       │
├─────────────┼───────────────────────────────────────────────────────────┤
│ 3D & Contour│ 3D Scatter • 3D Line • Surface • Contour                  │
├─────────────┼───────────────────────────────────────────────────────────┤
│ Specialized │ Sunburst • Treemap • Heatmap • Polar •                    │
│             │ Candlestick • Waterfall • Funnel • OHLC                   │
├─────────────┼───────────────────────────────────────────────────────────┤
│ Geographic  │ Scatter Geo • Choropleth • Globe                          │
├─────────────┼───────────────────────────────────────────────────────────┤
│ Advanced    │ Scatter Matrix • Parallel Coordinates • Ternary           │
└─────────────┴───────────────────────────────────────────────────────────┘
```

> 💡 **Tips:** Each button is an `Input` to a callback. When you click "Scatter", the callback sees `ctx.triggered_id == "btn-plot-scatter"` and generates the right code!

---

#### 4.2.4 The Workspace Panel - Command Window & Data View

```python
# app.py (Lines 1100-1150) - Workspace Panel
workspace_panel = dbc.Card([
    dbc.CardHeader([
        dbc.Tabs([
            dbc.Tab(label="Command Window", tab_id="tab-cmd"),
            dbc.Tab(label="Data View", tab_id="tab-dataview"),
        ], id="workspace-tabs", active_tab="tab-cmd")
    ], className="py-1"),
    dbc.CardBody([
        # Command Window Content (MATLAB-style)
        html.Div([
            dcc.Textarea(
                id='code-editor',
                placeholder="# Enter Python/Plotly code here...\n# Variables: figure_store, px, go, pd, np",
                style={'width': '100%', 'height': '200px', 'fontFamily': 'monospace'}
            ),
            dbc.Button("▶ Run Code", id="btn-run-custom-code", color="success", size="sm"),
            html.Hr(),
            html.Pre(id="console-output", children=">>> Ready.",
                    style={'height': '200px', 'overflow': 'auto', 'backgroundColor': '#1e1e1e', 'color': '#00ff00'})
        ], id="workspace-content-cmd"),
        
        # Data View Content (Spreadsheet-style)
        html.Div([
            html.Div(id="data-table-container", children="Select a dataset to view.")
        ], id="workspace-content-dataview", style={"display": "none"}),
    ], className="p-2")
], className="h-100")
```

**The Command Window - How It Works:**

```
┌─────────────────────────────────────────────────────────────────────┐
│ COMMAND WINDOW FLOW                                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  User types in code-editor:                                         │
│  ┌─────────────────────────────────────────┐                       │
│  │ fig = px.scatter(demo_data, x='x_val',  │                       │
│  │                  y='y_val', color='cat')│                       │
│  └─────────────────────────────────────────┘                       │
│          │                                                          │
│          ▼ [▶ Run Code] clicked                                    │
│  ┌─────────────────────────────────────────┐                       │
│  │ execute_code() callback triggers        │                       │
│  │ 1. exec(code, {}, local_scope)         │                       │
│  │ 2. Check if 'fig' in local_scope       │                       │
│  │ 3. Update main-graph with new figure   │                       │
│  └─────────────────────────────────────────┘                       │
│          │                                                          │
│          ▼                                                          │
│  ┌─────────────────────────────────────────┐                       │
│  │ Console shows: ">>> Code executed       │                       │
│  │                  successfully."         │                       │
│  └─────────────────────────────────────────┘                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

> 💡 **Tips:** The Command Window is like a mini-Jupyter notebook inside the app. You can type any Python code that creates a `fig` variable, and it will appear on the canvas!

---

#### 4.2.5 The Property Inspector - Dynamic UI Generation

One of the most sophisticated parts of the app is the Property Inspector. It **dynamically generates different controls** based on what element you select!

```python
# app.py (Lines 1150-1180) - Property Inspector Structure
property_inspector = dbc.Card([
    dbc.CardHeader([
        html.H6([html.I(className="fas fa-sliders-h me-1"), "Inspector"], className="mb-0")
    ], className="py-2"),
    dbc.CardBody([
        # Element Selection Dropdown
        dbc.Label("Element:", className="small fw-bold"),
        dcc.Dropdown(id="dd-element-select", placeholder="Select element..."),
        
        html.Hr(),
        
        # Highlight Button
        dbc.Button("🔦 Highlight", id="btn-highlight", color="outline-warning", size="sm"),
        
        html.Hr(),
        
        # DYNAMIC CONTROLS CONTAINER - This gets replaced based on selection!
        html.Div(id="inspector-controls", children=[
            html.P("Select an element above to edit.", className="text-muted small")
        ])
    ], className="p-2", style={"maxHeight": "calc(100vh - 200px)", "overflowY": "auto"})
], className="h-100")
```

**How Dynamic Inspector Works (The Magic!):**

```
┌─────────────────────────────────────────────────────────────────────┐
│ DYNAMIC INSPECTOR GENERATION                                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  User selects "trace_0" from dropdown                               │
│          │                                                          │
│          ▼                                                          │
│  update_inspector_controls() callback runs                          │
│          │                                                          │
│          ▼                                                          │
│  ┌─────────────────────────────────────────┐                       │
│  │ if selected_element == "figure":        │                       │
│  │   → Show: Title, Width, Height, Theme   │                       │
│  │   → Show: X/Y Axis Titles, Legend, etc. │                       │
│  │                                          │                       │
│  │ elif selected_element.startswith("trace_"): │                   │
│  │   → Show: Name, Color, Size, Opacity    │                       │
│  │   → Show: Symbol, Line Width, Dash      │                       │
│  │                                          │                       │
│  │ elif selected_element.startswith("shape_"): │                   │
│  │   → Show: Line Color, Fill Color, Width │                       │
│  │                                          │                       │
│  │ elif selected_element.startswith("annot_"): │                   │
│  │   → Show: Text, Font Size, Position     │                       │
│  └─────────────────────────────────────────┘                       │
│          │                                                          │
│          ▼                                                          │
│  Returns: html.Div([...dynamically created controls...])            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Available Properties by Element Type (from actual code):**

| Element Type | Available Properties |
|-------------|---------------------|
| **Figure** | Title, Width, Height, Plot Color, Paper Color, Font, Theme, X/Y Titles, Legend, Hover Mode, Grid, Bar Mode, Log Scale, Spikes, Zero Line |
| **Trace** | Name, Color, Size, Opacity, Symbol, Line Width, Dash Style, Mode, Fill, Text Position, Border Color, Line Shape |
| **Annotation** | Text, Color, Font Size, Font Family, X/Y Position, Show Arrow, Background Color, Text Angle |
| **Shape** | Line Color, Line Width, Opacity, Dash Style, Fill Color |
| **Image** | Opacity, Size X/Y, Position X/Y |

> 💡 **Tips:** This is why the Inspector looks different depending on what you select. The same `inspector-controls` div gets REPLACED with different content each time!

### 4.3 Callback Implementation - The "Brain" of the Application

> 💡 **Tips:** If the layout is the "body" of the app, callbacks are the "brain". They define WHAT HAPPENS when users interact with components. This is where the real magic happens!

Callbacks are the heart of the application's interactivity. Let's analyze the ACTUAL callbacks from `app.py`:

#### 4.3.1 Understanding Callback Structure

Every callback in Dash follows this pattern:

```python
@app.callback(
    Output("component-id", "property-to-update"),   # What changes
    Input("trigger-component", "property-that-triggers"),   # What causes the change
    State("another-component", "property-to-read"),   # Extra info needed
    prevent_initial_call=True   # Don't run when app first loads
)
def callback_function(trigger_value, state_value):
    # Your logic here
    return new_value_for_output
```

**The Three Types of Callback Arguments:**

```
┌─────────────────────────────────────────────────────────────────────┐
│ CALLBACK ARGUMENT TYPES                                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  INPUT  ──────────────────────────────────────────────────────────  │
│  • Triggers the callback when its value changes                     │
│  • Example: Button click, dropdown selection, text input            │
│  • If there are multiple Inputs, ANY change triggers the callback   │
│                                                                     │
│  STATE  ──────────────────────────────────────────────────────────  │
│  • Provides additional data but does NOT trigger the callback       │
│  • Example: Current figure, selected dropdown value                 │
│  • Think of it as "read-only context"                               │
│                                                                     │
│  OUTPUT ─────────────────────────────────────────────────────────── │
│  • What gets updated when the callback runs                         │
│  • Example: Figure, text content, component visibility              │
│  • Can have multiple outputs (return a tuple)                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

> 💡 **Tips:** Think of it like a recipe. **Input** is "when to start cooking" (turning on the stove), **State** is "ingredients already on the counter", and **Output** is "the finished dish"!

---

#### 4.3.2 Tab Switching Callback - The Simplest Example

Let's start with the simplest callback - switching ribbon tabs:

```python
# app.py (Lines 1235-1250) - Ribbon Tab Switching
@app.callback(
    Output("ribbon-content-home", "style"),
    Output("ribbon-content-data", "style"),
    Output("ribbon-content-plots", "style"),
    Output("ribbon-content-annotate", "style"),
    Output("ribbon-content-view", "style"),
    Input("ribbon-tabs", "active_tab"),
)
def toggle_ribbon(active_tab):
    """Show only the content for the active tab."""
    show = {"display": "block"}   # CSS to show element
    hide = {"display": "none"}    # CSS to hide element
    
    return (
        show if active_tab == "tab-home" else hide,      # HOME tab content
        show if active_tab == "tab-data" else hide,      # DATA tab content
        show if active_tab == "tab-plots" else hide,     # PLOTS tab content
        show if active_tab == "tab-annotate" else hide,  # ANNOTATE tab content
        show if active_tab == "tab-view" else hide,      # VIEW tab content
    )
```

**How This Works (Visual):**

```
┌─────────────────────────────────────────────────────────────────────┐
│ TAB SWITCHING FLOW                                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  User clicks "PLOTS" tab                                            │
│          │                                                          │
│          ▼                                                          │
│  ribbon-tabs.active_tab becomes "tab-plots"                         │
│          │                                                          │
│          ▼                                                          │
│  toggle_ribbon("tab-plots") is called                               │
│          │                                                          │
│          ▼                                                          │
│  Returns: (hide, hide, SHOW, hide, hide)                            │
│            HOME   DATA  PLOTS ANNOT VIEW                            │
│          │                                                          │
│          ▼                                                          │
│  Only PLOTS content is visible, others are hidden                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

#### 4.3.3 Data Management Callback - A Complex Multi-Input Example

This is one of the most complex callbacks - it handles CSV upload, demo data generation, deletion, and cleaning:

```python
# app.py (Lines 1275-1380) - Data Management Callback
@app.callback(
    Output("dd-dataframe-select", "options"),      # Update dropdown options
    Output("dd-dataframe-select", "value"),        # Update selected value
    Output("data-info-label", "children"),         # Update info text
    Output("data-update-signal", "data"),          # Signal other callbacks
    Output("main-graph", "figure", allow_duplicate=True),  # Update plot
    Input("upload-csv", "contents"),               # CSV upload trigger
    Input("btn-gen-demo", "n_clicks"),             # Demo data button
    Input("btn-delete-data", "n_clicks"),          # Delete button
    Input("btn-clean-na", "n_clicks"),             # Clean NA button
    State("upload-csv", "filename"),               # Get filename
    State("dd-dataframe-select", "value"),         # Get current selection
    State("data-update-signal", "data"),           # Get current signal
    prevent_initial_call=True
)
def manage_data(upload_content, _n_demo, _n_delete, _n_clean, 
                filename, current_selection, current_signal):
    """
    Central data management callback handling multiple operations.
    Uses ctx.triggered_id to determine which action was triggered.
    """
    ctx_id = ctx.triggered_id  # Which component triggered this callback?
    current_signal = current_signal or 0
    fig_update = dash.no_update  # Default: don't change the figure
    
    # ============= HANDLE CSV UPLOAD =============
    if ctx_id == "upload-csv" and upload_content:
        # Decode the uploaded file
        _content_type, content_string = upload_content.split(',')
        decoded = base64.b64decode(content_string)
        
        try:
            # Parse CSV into DataFrame
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            name = filename.split('.')[0]  # Remove .csv extension
            
            # Store in FigureStore
            figure_store.add_dataframe(name, df)
            current_selection = name  # Auto-select new data
            current_signal += 1       # Signal update
            
        except Exception as e:
            print(f"Error parsing CSV: {e}")
    
    # ============= HANDLE DEMO DATA GENERATION =============
    if ctx_id == "btn-gen-demo":
        n_points = 200
        t = np.linspace(0, 10, n_points)
        
        # Create rich dataset for ALL plot types
        df = pd.DataFrame({
            "time": t,
            "signal": np.sin(t) * 10 + np.random.normal(0, 1, n_points),
            "noise": np.random.randn(n_points),
            "category": np.random.choice(['A', 'B', 'C', 'D'], n_points),
            "x_val": np.random.randn(n_points) * 10,
            "y_val": np.random.randn(n_points) * 10,
            "z_val": np.random.randn(n_points) * 10,  # For 3D plots
            "lat": np.random.uniform(-50, 70, n_points),  # For maps
            "lon": np.random.uniform(-120, 140, n_points),
            # ... more columns for specialized charts
        })
        
        name = f"demo_{uuid.uuid4().hex[:4]}"
        figure_store.add_dataframe(name, df)
        current_selection = name
        current_signal += 1
    
    # ============= HANDLE DELETION =============
    if ctx_id == "btn-delete-data" and current_selection:
        # Remove from repository
        if current_selection in figure_store.data_repository:
            del figure_store.data_repository[current_selection]
            
            # Also remove associated traces
            keys_to_remove = [k for k, d in figure_store.datasets.items() 
                            if d.name == current_selection]
            for k in keys_to_remove:
                del figure_store.datasets[k]
            
            # Rebuild figure without deleted traces
            figure_store.rebuild_figure_from_datasets()
            fig_update = figure_store.get_figure_dict()
            
            current_selection = None
            current_signal += 1
    
    # ============= HANDLE CLEANING =============
    if ctx_id == "btn-clean-na" and current_selection:
        df = figure_store.get_dataframe(current_selection)
        if df is not None:
            # Smart cleaning: convert numeric strings, drop NaN
            for col in df.columns:
                if df[col].dtype == 'object':
                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                    if numeric_col.count() / df[col].count() > 0.5:
                        df[col] = numeric_col
            
            df = df.dropna()
            figure_store.add_dataframe(current_selection, df)
            # ... update associated traces ...
            current_signal += 1
    
    # Build dropdown options
    options = [{"label": k, "value": k} 
               for k in figure_store.data_repository.keys()]
    
    # Build info label
    info_text = "No data loaded"
    if current_selection:
        df = figure_store.get_dataframe(current_selection)
        if df is not None:
            info_text = f"{len(df)} rows × {len(df.columns)} cols"
    
    return options, current_selection, info_text, current_signal, fig_update
```

**Visual: How ctx.triggered_id Works**

```
┌─────────────────────────────────────────────────────────────────────┐
│ MULTI-INPUT CALLBACK WITH ctx.triggered_id                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  FOUR DIFFERENT BUTTONS can trigger the same callback:              │
│                                                                     │
│  [Upload CSV] ──┐                                                   │
│  [Load Demo]  ──┼──► manage_data() callback                         │
│  [Delete]     ──┤                                                   │
│  [Clean NA]   ──┘                                                   │
│                      │                                              │
│                      ▼                                              │
│              ctx.triggered_id tells us WHICH ONE was clicked:       │
│                                                                     │
│              if ctx.triggered_id == "upload-csv":                   │
│                  → Handle CSV upload                                │
│              elif ctx.triggered_id == "btn-gen-demo":               │
│                  → Generate demo data                               │
│              elif ctx.triggered_id == "btn-delete-data":            │
│                  → Delete selected data                             │
│              elif ctx.triggered_id == "btn-clean-na":               │
│                  → Clean the data                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

> 💡 **Tips:** This pattern is VERY common in Dash apps. Instead of writing 4 separate callbacks, we combine related operations into one and use `ctx.triggered_id` to know which button was clicked!

---

#### 4.3.4 Code Generation Callback - Auto-Generate Plotly Code

This callback generates executable Python code when you click a chart type button:

```python
# app.py (Lines 1655-1710) - Code Generation
@app.callback(
    Output("code-editor", "value"),          # Put code in editor
    Output("trigger-run-signal", "data"),    # Trigger auto-run
    Input("btn-plot-scatter", "n_clicks"),
    Input("btn-plot-line", "n_clicks"),
    Input("btn-plot-bar", "n_clicks"),
    # ... 23 more chart type buttons ...
    Input("btn-plot-globe", "n_clicks"),
    State("dd-dataframe-select", "value"),
    State("trigger-run-signal", "data"),
    prevent_initial_call=True
)
def generate_and_trigger_plot(_n_sc, _n_ln, _n_bar, ..., df_name, current_signal):
    """Generate smart Plotly code based on chart type and data."""
    if not df_name:
        return "# Please select a dataset first.", dash.no_update
        
    ctx_id = ctx.triggered_id
    if not ctx_id:
        raise PreventUpdate
        
    # Extract plot type from button ID: "btn-plot-scatter" → "scatter"
    plot_type = ctx_id.replace("btn-plot-", "")
    
    df = figure_store.get_dataframe(df_name)
    if df is None:
        return "# Error: Dataset not found.", dash.no_update

    # Use CodeGenerator to create smart code
    cmd = code_generator.generate_smart_plot_code(df_name, plot_type, df)
    
    # Increment signal to trigger auto-execution
    new_signal = (current_signal or 0) + 1
    return cmd, new_signal
```

**What generate_smart_plot_code() Does:**

```python
# Simplified from CodeGenerator class (Lines 600-750)
def generate_smart_plot_code(self, df_name, plot_type, df):
    """Intelligently generate code based on data types."""
    
    # Analyze column types
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Smart column selection based on plot type
    if plot_type == 'scatter':
        x = num_cols[0] if num_cols else df.columns[0]
        y = num_cols[1] if len(num_cols) > 1 else num_cols[0]
        return f"fig = px.scatter({df_name}, x='{x}', y='{y}')"
        
    elif plot_type == 'pie':
        names = cat_cols[0] if cat_cols else df.columns[0]
        values = num_cols[0] if num_cols else df.columns[1]
        return f"fig = px.pie({df_name}, names='{names}', values='{values}')"
        
    elif plot_type == 'scatter3d':
        x, y, z = num_cols[0], num_cols[1], num_cols[2]
        return f"fig = px.scatter_3d({df_name}, x='{x}', y='{y}', z='{z}')"
        
    # ... more plot types ...
```

**How Code Generation + Auto-Execution Works:**

```
┌─────────────────────────────────────────────────────────────────────┐
│ CODE GENERATION + AUTO-EXECUTION FLOW                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. User clicks [Scatter] button                                    │
│          │                                                          │
│          ▼                                                          │
│  2. generate_and_trigger_plot() runs                                │
│     → Generates: "fig = px.scatter(demo, x='x_val', y='y_val')"    │
│     → Returns: (code, signal+1)                                     │
│          │                                                          │
│          ▼                                                          │
│  3. Code appears in Command Window (code-editor)                    │
│  4. trigger-run-signal changes value                                │
│          │                                                          │
│          ▼                                                          │
│  5. execute_code() callback triggers (it listens to signal)         │
│     → exec(code, {}, local_scope)                                   │
│     → fig variable is created                                       │
│     → Updates main-graph with new figure                            │
│          │                                                          │
│          ▼                                                          │
│  6. Chart appears on canvas! ✨                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

#### 4.3.5 Code Execution Callback - Running User Code

This callback safely executes Python code from the Command Window:

```python
# app.py (Lines 1715-1760) - Code Execution
@app.callback(
    Output("main-graph", "figure"),
    Output("console-output", "children"),
    Input("trigger-run-signal", "data"),        # Auto-run trigger
    Input("btn-run-custom-code", "n_clicks"),   # Manual run button
    State("code-editor", "value"),
    State("console-output", "children"),
    prevent_initial_call=True
)
def execute_code(signal, n_clicks, code, current_console):
    """Execute code from the command window."""
    if not code:
        raise PreventUpdate

    try:
        # Create a safe execution environment
        local_scope = {
            "pd": pd,              # Pandas
            "px": px,              # Plotly Express
            "go": go,              # Graph Objects
            "np": np,              # NumPy
            "figure_store": figure_store  # Access to data
        }
        
        # Inject all loaded dataframes into scope
        for name, df in figure_store.data_repository.items():
            local_scope[name] = df   # demo_abc → actual DataFrame
            
        # EXECUTE THE CODE
        exec(code, {}, local_scope)
        
        # Check if 'fig' variable was created
        if "fig" in local_scope:
            fig = local_scope["fig"]
            if isinstance(fig, go.Figure):
                figure_store.update_figure(fig)
                return fig, f"{current_console}\n>>> Code executed successfully."
            else:
                return dash.no_update, f"{current_console}\n>>> Error: 'fig' is not a Figure."
        else:
            return dash.no_update, f"{current_console}\n>>> Code ran, but no 'fig' found."
            
    except Exception as e:
        return dash.no_update, f"{current_console}\n>>> Execution Error: {e}"
```

**Security Note:**

> ⚠️ **Important:** Using `exec()` to run user code is powerful but can be dangerous in production. Our app is designed for single-user local use, so this is acceptable. For a public web app, you'd need sandboxing!

---

#### 4.3.6 Property Editor Callback - The "Big One" (35+ Properties)

This is the largest callback in the app - it handles editing ANY property of ANY element:

```python
# app.py (Lines 1770-1900) - Property Editor (Simplified)
@app.callback(
    Output("main-graph", "figure", allow_duplicate=True),
    Input("btn-apply-props", "n_clicks"),
    State("dd-element-select", "value"),           # Which element?
    State("input-prop-name", "value"),             # Name/Title
    State("input-prop-color", "value"),            # Color
    State("input-prop-size", "value"),             # Size
    State("input-prop-opacity", "value"),          # Opacity
    State("input-prop-symbol", "value"),           # Symbol
    State("input-prop-width", "value"),            # Line Width
    State("input-prop-dash", "value"),             # Line Dash
    # ... 25+ more State inputs ...
    State("input-prop-global_font_size", "value"), # Global Font Size
    State("main-graph", "figure"),
    prevent_initial_call=True
)
def apply_property_changes(n_clicks, selected_element, name, color, size, 
                           opacity, symbol, width, dash_style, ...):
    """Apply property changes to the selected element."""
    if not n_clicks or not selected_element:
        raise PreventUpdate
        
    fig = go.Figure(fig_dict)
    
    # Collect all non-None property values
    props = {}
    if name: props['name'] = name
    if color: props['color'] = color
    if size: props['size'] = size
    # ... collect all properties ...
    
    if not props:
        return dash.no_update

    # Apply based on element type
    if selected_element == "figure":
        # Update layout properties (title, axes, theme, etc.)
        layout_updates = {}
        if 'name' in props: 
            layout_updates['title'] = dict(text=props['name'])
        if 'template' in props: 
            layout_updates['template'] = props['template']
        # ... more layout updates ...
        fig.update_layout(**layout_updates)
        fig.update_xaxes(**xaxis_opts)
        fig.update_yaxes(**yaxis_opts)

    elif selected_element.startswith("trace_"):
        # Update trace properties (color, size, marker, line, etc.)
        idx = int(selected_element.split("_")[1])
        trace = fig.data[idx]
        
        marker_updates = {}
        if 'color' in props: marker_updates['color'] = props['color']
        if 'size' in props: marker_updates['size'] = props['size']
        if marker_updates: 
            trace.update(marker=marker_updates)
            
        line_updates = {}
        if 'width' in props: line_updates['width'] = props['width']
        if line_updates:
            trace.update(line=line_updates)

    elif selected_element.startswith("annot_"):
        # Update annotation properties
        idx = int(selected_element.split("_")[1])
        annot = fig.layout.annotations[idx]
        annot.update(text=props.get('text'), font=dict(color=props.get('color')))
        
    elif selected_element.startswith("shape_"):
        # Update shape properties
        idx = int(selected_element.split("_")[1])
        shape = fig.layout.shapes[idx]
        shape.update(line=dict(color=props.get('color'), width=props.get('width')))

    figure_store.update_figure(fig)
    return fig
```

**Property Application Logic:**

```
┌─────────────────────────────────────────────────────────────────────┐
│ PROPERTY EDITOR FLOW                                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  User selects "trace_0" → Changes Color to "red" → Clicks [Apply]   │
│                                                                     │
│  ┌─────────────────────────────────────────┐                       │
│  │ 1. selected_element = "trace_0"         │                       │
│  │ 2. props = {'color': 'red'}             │                       │
│  │ 3. Extract index: idx = 0               │                       │
│  │ 4. Get trace: trace = fig.data[0]       │                       │
│  │ 5. Apply: trace.update(marker={'color': 'red'})                 │
│  │ 6. Also: trace.update(line={'color': 'red'})                    │
│  │ 7. Return updated figure                 │                       │
│  └─────────────────────────────────────────┘                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

#### 4.3.7 History (Undo/Redo) Callback

```python
# app.py (Lines 2525-2565) - History Management
@app.callback(
    Output("main-graph", "figure", allow_duplicate=True),
    Output("btn-undo", "disabled"),            # Disable when empty
    Output("btn-redo", "disabled"),
    Input("btn-undo", "n_clicks"),
    Input("btn-redo", "n_clicks"),
    Input("main-graph", "figure"),             # Listen for changes
    State("main-graph", "figure"),
    prevent_initial_call=True
)
def manage_history(n_undo, n_redo, fig_trigger, current_fig_dict):
    ctx_id = ctx.triggered_id
    
    # If triggered by graph update, just update button states
    if ctx_id == "main-graph":
        return (dash.no_update, 
                not history_stack.can_undo(),  # Disable if can't undo
                not history_stack.can_redo())  # Disable if can't redo
        
    # Handle Undo
    if ctx_id == "btn-undo":
        new_fig = history_stack.undo()
        if new_fig:
            fig = go.Figure(new_fig)
            figure_store.update_figure(fig)
            return fig, not history_stack.can_undo(), not history_stack.can_redo()
    
    # Handle Redo
    elif ctx_id == "btn-redo":
        new_fig = history_stack.redo()
        if new_fig:
            fig = go.Figure(new_fig)
            figure_store.update_figure(fig)
            return fig, not history_stack.can_undo(), not history_stack.can_redo()
        
    return dash.no_update, not history_stack.can_undo(), not history_stack.can_redo()
```

**Undo/Redo Stack Visualization:**

```
┌─────────────────────────────────────────────────────────────────────┐
│ HISTORY STACK VISUALIZATION                                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Initial State:                                                     │
│  UNDO: []                    REDO: []                               │
│                                                                     │
│  After Adding Scatter:                                              │
│  UNDO: [state_0]             REDO: []                               │
│                                                                     │
│  After Changing Color:                                              │
│  UNDO: [state_0, state_1]    REDO: []                               │
│                                                                     │
│  After Clicking UNDO:                                               │
│  UNDO: [state_0]             REDO: [state_1]                        │
│  (Restored to state_1, color change undone)                         │
│                                                                     │
│  After Clicking REDO:                                               │
│  UNDO: [state_0, state_1]    REDO: []                               │
│  (Back to state with color change)                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

> 💡 **Tips:** The undo/redo system saves a complete copy of the figure before each change. This is memory-intensive but very reliable!

### 4.4 Helper Functions - The "Utility Belt"

> 💡 **Tips:** Helper functions are like tools in Batman's utility belt - they're not the main show, but the hero couldn't do their job without them!

Several utility functions support the callback logic. Let's examine the key ones from `app.py`:

#### 4.4.1 The clean_figure_dict() Function - Data Sanitizer

One of the most important helper functions ensures figure data is clean before processing:

```python
# app.py (Lines 800-850) - Figure Cleaning
def clean_figure_dict(fig_dict):
    """
    Clean a figure dictionary to remove problematic values.
    
    Why this is needed:
    - Dash sometimes passes weird values (like undefined JavaScript objects)
    - NumPy arrays need to be converted to lists for JSON serialization
    - NaN values cause problems and should be filtered
    """
    if fig_dict is None:
        return None
        
    import copy
    cleaned = copy.deepcopy(fig_dict)  # Don't modify original!
    
    def clean_value(v):
        """Recursively clean a value."""
        if isinstance(v, dict):
            return {k: clean_value(val) for k, val in v.items() 
                   if val is not None}
        elif isinstance(v, list):
            return [clean_value(item) for item in v 
                   if item is not None]
        elif isinstance(v, (np.ndarray,)):
            return v.tolist()  # Convert NumPy to list
        elif isinstance(v, (np.floating, np.integer)):
            return float(v) if np.isfinite(v) else None
        else:
            return v
    
    return clean_value(cleaned)
```

**Why clean_figure_dict() is Critical:**

```
┌─────────────────────────────────────────────────────────────────────┐
│ WITHOUT clean_figure_dict():                                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  fig_dict = {                                                       │
│      'data': [{                                                     │
│          'x': np.array([1, 2, 3]),    ← NumPy array (can't JSON)   │
│          'y': [1, np.nan, 3],         ← NaN value (causes errors)  │
│      }],                                                            │
│      'layout': {                                                    │
│          'undefined_property': undefined  ← From JavaScript        │
│      }                                                              │
│  }                                                                  │
│                                                                     │
│  json.dumps(fig_dict)  → ERROR! ❌                                  │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ WITH clean_figure_dict():                                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  cleaned = {                                                        │
│      'data': [{                                                     │
│          'x': [1, 2, 3],              ← Now a Python list ✓        │
│          'y': [1, 3],                 ← NaN removed ✓              │
│      }],                                                            │
│      'layout': {}                      ← Undefined removed ✓       │
│  }                                                                  │
│                                                                     │
│  json.dumps(cleaned)  → Works! ✓                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

#### 4.4.2 The create_initial_figure() Function - Default Canvas

This function creates the initial empty figure when the app starts:

```python
# app.py (Lines 855-880) - Initial Figure
def create_initial_figure():
    """
    Create the initial empty figure shown when app loads.
    Sets up a professional-looking blank canvas.
    """
    fig = go.Figure()
    
    fig.update_layout(
        title=dict(
            text="Interactive Figure Editor",
            font=dict(size=20, color="#333")
        ),
        xaxis=dict(
            title="X Axis",
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title="Y Axis", 
            showgrid=True,
            gridcolor='lightgray'
        ),
        template="plotly_white",
        showlegend=True,
        margin=dict(l=60, r=40, t=60, b=40),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig
```

---

#### 4.4.3 Data Parsing Functions - Understanding User Input

The CodeGenerator class includes smart data analysis:

```python
# From CodeGenerator class (Lines 650-750)
def generate_smart_plot_code(self, df_name, plot_type, df):
    """
    Intelligently generate plotting code by analyzing data types.
    
    This is 'smart' because it:
    1. Detects numeric vs categorical columns
    2. Chooses appropriate columns for each plot type
    3. Handles edge cases (missing columns, wrong types)
    """
    # Detect column types
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    
    # Smart column mapping based on plot type
    column_mapping = {
        'scatter':    {'x': num_cols[0], 'y': num_cols[1] if len(num_cols) > 1 else num_cols[0]},
        'line':       {'x': date_cols[0] if date_cols else num_cols[0], 'y': num_cols[0]},
        'bar':        {'x': cat_cols[0] if cat_cols else num_cols[0], 'y': num_cols[0]},
        'pie':        {'names': cat_cols[0] if cat_cols else 'category', 'values': num_cols[0]},
        'histogram':  {'x': num_cols[0]},
        'box':        {'x': cat_cols[0] if cat_cols else None, 'y': num_cols[0]},
        'heatmap':    {'x': num_cols[0], 'y': num_cols[1], 'z': num_cols[2] if len(num_cols) > 2 else num_cols[0]},
        'scatter3d':  {'x': num_cols[0], 'y': num_cols[1], 'z': num_cols[2]},
        'scattergeo': {'lat': 'lat', 'lon': 'lon'},
        # ... more mappings
    }
    
    # Generate appropriate code
    mapping = column_mapping.get(plot_type, {})
    
    if plot_type == 'scatter':
        return f"fig = px.scatter({df_name}, x='{mapping['x']}', y='{mapping['y']}')"
    elif plot_type == 'pie':
        return f"fig = px.pie({df_name}, names='{mapping['names']}', values='{mapping['values']}')"
    # ... etc
```

**Smart Column Detection Example:**

```
┌─────────────────────────────────────────────────────────────────────┐
│ SMART COLUMN DETECTION                                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  DataFrame:                                                         │
│  ┌────────────┬────────┬──────────┬───────────┐                    │
│  │ date       │ sales  │ category │ profit    │                    │
│  ├────────────┼────────┼──────────┼───────────┤                    │
│  │ 2024-01-01 │ 100    │ "A"      │ 10.5      │                    │
│  │ 2024-01-02 │ 150    │ "B"      │ 15.2      │                    │
│  └────────────┴────────┴──────────┴───────────┘                    │
│                                                                     │
│  Analysis:                                                          │
│  • num_cols  = ['sales', 'profit']                                  │
│  • cat_cols  = ['category']                                         │
│  • date_cols = ['date']                                             │
│                                                                     │
│  For "scatter" plot:                                                │
│  → x = 'sales' (first numeric)                                      │
│  → y = 'profit' (second numeric)                                    │
│  → Code: fig = px.scatter(df, x='sales', y='profit')               │
│                                                                     │
│  For "bar" plot:                                                    │
│  → x = 'category' (first categorical)                               │
│  → y = 'sales' (first numeric)                                      │
│  → Code: fig = px.bar(df, x='category', y='sales')                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

> 💡 **Tips:** This is why clicking "Scatter" with demo data just WORKS - the code generator is smart enough to find the right columns!

---

#### 4.4.4 Inspector Control Generation - Dynamic UI Building

The `update_inspector_controls` callback generates different UI controls based on context:

```python
# From update_inspector_controls callback (Lines 2085-2300)
def make_row(label, id_suffix, input_type="text", value=None, options=None, visible=True):
    """
    Helper to create a property control row.
    
    Parameters:
    - label: Display name (e.g., "Color")
    - id_suffix: ID suffix (e.g., "color" → "input-prop-color")
    - input_type: "text", "number", or "select"
    - value: Current value to display
    - options: For select type, list of options
    - visible: Whether to show this row
    """
    style = {} if visible else {"display": "none"}
    
    if input_type == "select" and options:
        # Create dropdown
        input_component = dbc.Select(
            id=f"input-prop-{id_suffix}",
            options=[{"label": o.title(), "value": o} for o in options]
                   if isinstance(options[0], str) else options,
            value=value,
            size="sm"
        )
    else:
        # Create text/number input
        input_component = dbc.Input(
            id=f"input-prop-{id_suffix}",
            type=input_type,
            value=value,
            size="sm"
        )

    return dbc.Row([
        dbc.Col(dbc.Label(label, className="small mb-0"), width=4),
        dbc.Col(input_component, width=8)
    ], className="mb-2", style=style)
```

**Property Configuration Dictionary (Actual Code):**

```python
# The config dict defines ALL possible properties
config = {
    "name":    {"visible": False, "label": "Name", "value": None, "type": "text"},
    "color":   {"visible": False, "label": "Color", "value": None, "type": "select", 
                "options": ['black', 'white', 'red', 'green', 'blue', ...]},
    "size":    {"visible": False, "label": "Size", "value": None, "type": "number"},
    "opacity": {"visible": False, "label": "Opacity", "value": None, "type": "number"},
    "symbol":  {"visible": False, "label": "Symbol", "value": None, "type": "select",
                "options": ['circle', 'square', 'diamond', 'cross', 'x', ...]},
    "width":   {"visible": False, "label": "Width", "value": None, "type": "number"},
    "dash":    {"visible": False, "label": "Dash", "value": None, "type": "select",
                "options": ['solid', 'dot', 'dash', 'longdash', 'dashdot']},
    # ... 25+ more properties
}

# Then based on selected element, enable relevant properties:
if selected_element == "figure":
    config["name"].update({"visible": True, "label": "Title"})
    config["template"].update({"visible": True})
    config["xaxis"].update({"visible": True})
    # ... etc

elif selected_element.startswith("trace_"):
    config["name"].update({"visible": True})
    config["color"].update({"visible": True})
    config["size"].update({"visible": True})
    config["opacity"].update({"visible": True})
    # ... etc
```

---

#### 4.4.5 Application Launch - The Final Piece

```python
# app.py (Lines 2650-2677) - Launch Code
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
```

**What app.run() Parameters Mean:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `debug=True` | Boolean | Enables auto-reload on code changes, shows detailed errors |
| `jupyter_mode='inline'` | String | Runs inline when in Jupyter, opens browser otherwise |
| `port=8051` | Integer | Which port to run on (default is 8050) |

---

### 4.5 Complete Callback Reference Table

Here's a comprehensive table of ALL callbacks in `app.py`:

| Callback Name | Lines | Inputs | Outputs | Purpose |
|--------------|-------|--------|---------|---------|
| `toggle_ribbon` | 1235-1250 | ribbon-tabs.active_tab | 5 ribbon content styles | Switch ribbon tabs |
| `toggle_workspace` | 1255-1265 | workspace-tabs.active_tab | 2 workspace content styles | Switch workspace tabs |
| `toggle_inspector` | 1270-1275 | chk-inspector-toggle.value | col-inspector.style | Show/hide inspector |
| `manage_data` | 1280-1380 | 4 data buttons + upload | dropdown, info, signal, figure | Central data management |
| `update_data_view` | 1385-1430 | dropdown, 3 view buttons, signal | table, active_tab | Show data table/stats |
| `sync_data_from_table` | 1435-1480 | interactive-data-table.data | console, figure | Sync edits back |
| `remove_selected_points` | 1485-1580 | btn-remove-selected | figure, selectedData, console | Delete selected points |
| `generate_and_trigger_plot` | 1655-1710 | 26 plot buttons | code-editor, signal | Generate plot code |
| `execute_code` | 1715-1760 | signal, run button | figure, console | Run user code |
| `apply_property_changes` | 1770-1900 | apply button + 35 states | figure | Edit element properties |
| `delete_element` | 2000-2030 | delete button | figure, element-select | Delete element |
| `highlight_element` | 2035-2080 | highlight button, dropdown | figure | Highlight selected |
| `update_element_options` | 2085-2115 | main-graph.figure | dd-element-select.options | Update element list |
| `update_inspector_controls` | 2120-2300 | dd-element-select | inspector-controls | Generate property UI |
| `set_shape_draw_mode` | 2305-2350 | 5 draw buttons | figure | Enable drawing tools |
| `sync_drawn_shapes` | 2400-2430 | relayoutData | figure-store, figure | Save drawn shapes |
| `toggle_about_modal` | 2435-2445 | open/close buttons | modal.is_open | Show/hide about |
| `toggle_annot_modal` | 2450-2460 | add-text/confirm buttons | modal.is_open | Show/hide annotation modal |
| `add_text_annotation` | 2465-2500 | confirm button + states | figure | Add text annotation |
| `add_background_image` | 2510-2540 | upload-image.contents | figure | Add image overlay |
| `manage_history` | 2545-2590 | undo/redo buttons, figure | figure, button states | Undo/redo system |
| `save_session` | 2595-2605 | save button | download-component | Export to JSON |
| `load_session` | 2610-2630 | upload-session.contents | figure | Import from JSON |
| `view_tools` | 2635-2660 | zoom/pan/reset buttons | figure | Navigation tools |
| `show_selection_stats` | 2665-2690 | selectedData | console | Show stats on selection |

**Total: 24 callbacks managing 50+ component interactions!**

> 💡 **Tips:** Don't try to memorize all these! Use this table as a reference when you want to understand or modify a specific feature.

---

## 5. Complete Feature Documentation - Your User Manual

> 💡 **Tips:** This chapter is your **complete user manual**! It explains every feature in PyFigureEditor with step-by-step instructions. Keep this section bookmarked - you'll refer to it often!

This chapter provides comprehensive documentation of all features available in PyFigureEditor, organized by functional category. Each feature includes usage instructions, supported options, and practical examples.

### 5.1 Plot Types and Chart Creation - Your 26+ Chart Arsenal

> 💡 **Tips:** One of the biggest advantages of this app is that you don't need to memorize Plotly syntax for each chart type. Just click a button and the code is generated for you!

PyFigureEditor supports **26+ plot types**, covering virtually every common visualization need. Let's explore them category by category:

#### 5.1.1 Basic 2D Charts - Where Most Analysis Starts

**Overview Table:**

| Plot Type | Button | Best For | Minimum Data Needed |
|-----------|--------|----------|---------------------|
| **Scatter** | `[Scatter]` | Correlation, clusters | 2 numeric columns |
| **Line** | `[Line]` | Trends, time series | 2 numeric columns |
| **Bar** | `[Bar]` | Comparisons | 1 categorical + 1 numeric |
| **Area** | `[Area]` | Cumulative totals | 2 numeric columns |
| **Bubble** | `[Bubble]` | 3-variable scatter | 3 numeric columns |

**Step-by-Step: Creating Your First Scatter Plot**

```
┌─────────────────────────────────────────────────────────────────────┐
│ CREATING A SCATTER PLOT                                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 1: Load Data                                                  │
│  ┌─────────────────────────────────────────┐                       │
│  │ Click [Load Demo] OR [Import CSV]       │                       │
│  │ Wait for "200 rows × 15 cols" message   │                       │
│  └─────────────────────────────────────────┘                       │
│                                                                     │
│  Step 2: Select Chart Type                                          │
│  ┌─────────────────────────────────────────┐                       │
│  │ Go to PLOTS tab                         │                       │
│  │ Click [Scatter] button                  │                       │
│  └─────────────────────────────────────────┘                       │
│                                                                     │
│  Step 3: Watch the Magic!                                           │
│  ┌─────────────────────────────────────────┐                       │
│  │ Code appears in Command Window:         │                       │
│  │ fig = px.scatter(demo_xxxx, x='x_val',  │                       │
│  │                  y='y_val')             │                       │
│  │                                          │                       │
│  │ Chart appears on canvas automatically!  │                       │
│  └─────────────────────────────────────────┘                       │
│                                                                     │
│  Step 4: Customize (Optional)                                       │
│  ┌─────────────────────────────────────────┐                       │
│  │ Select "Trace 0" in Property Inspector  │                       │
│  │ Change Color → red                      │                       │
│  │ Change Size → 12                        │                       │
│  │ Click [Apply Changes]                   │                       │
│  └─────────────────────────────────────────┘                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Pro Tip for Line Charts:**

```python
# The app auto-generates this when you click [Line]:
fig = px.line(demo_data, x='time', y='signal')

# But you can modify it in Command Window to add features:
fig = px.line(demo_data, x='time', y='signal', 
              color='category',           # Different colors per category
              markers=True,               # Add dots at data points
              title='My Time Series')     # Add title
```

---

#### 5.1.2 Statistical Charts - For Data Scientists

> 💡 **Tips:** These charts help you understand the **distribution** and **statistical properties** of your data. They're essential for exploratory data analysis!

**Overview Table:**

| Plot Type | Button | What It Shows | When to Use |
|-----------|--------|---------------|-------------|
| **Histogram** | `[Hist]` | Distribution shape | "How are my values spread out?" |
| **Box Plot** | `[Box]` | Quartiles, outliers | "What are the median and outliers?" |
| **Violin** | `[Violin]` | Distribution + density | "What's the full shape of my data?" |
| **Heatmap** | `[Heatmap]` | 2D correlations | "How do variables relate?" |

**Understanding Box Plots - A Visual Guide:**

```
┌─────────────────────────────────────────────────────────────────────┐
│ ANATOMY OF A BOX PLOT                                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│         ●  ← Outlier (> 1.5 × IQR above Q3)                        │
│                                                                     │
│       ─┬─  ← Maximum (excluding outliers)                          │
│        │                                                            │
│        │   Upper Whisker                                            │
│        │                                                            │
│       ┌┴┐                                                           │
│       │ │  ← Q3 (75th percentile)                                  │
│       │ │                                                           │
│       │━│  ← Median (50th percentile)                              │
│       │ │                                                           │
│       │ │  ← Q1 (25th percentile)                                  │
│       └┬┘                                                           │
│        │                                                            │
│        │   Lower Whisker                                            │
│        │                                                            │
│       ─┴─  ← Minimum (excluding outliers)                          │
│                                                                     │
│         ●  ← Outlier (< 1.5 × IQR below Q1)                        │
│                                                                     │
│  IQR = Interquartile Range = Q3 - Q1                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Creating a Box Plot:**

```
1. Load your data (demo or CSV)
2. Go to PLOTS tab → Click [Box]
3. Generated code: fig = px.box(demo_data, x='category', y='signal')
4. The chart automatically groups data by category!

To customize:
- Select "Figure Settings" → Change Theme to "plotly_white"
- The box plot now has a cleaner background
```

---

#### 5.1.3 3D Charts - When Two Dimensions Aren't Enough

> 💡 **Tips:** 3D charts are powerful but can be misleading if overused. Use them when your data truly has three important dimensions!

**Overview Table:**

| Plot Type | Button | Data Needed | Interactive Features |
|-----------|--------|-------------|---------------------|
| **3D Scatter** | `[3D Scatter]` | x, y, z columns | Rotate, zoom, pan |
| **3D Line** | `[Line 3D]` | x, y, z columns | Trace paths in 3D |
| **Surface** | `[Surface]` | z matrix (2D array) | Explore terrain-like data |
| **Contour** | `[Contour]` | z matrix | Like Surface but flat |

**Creating a 3D Scatter Plot:**

```
1. Load demo data (it has x_val, y_val, z_val columns)
2. Click [3D Scatter]
3. Generated code: 
   fig = px.scatter_3d(demo_data, x='x_val', y='y_val', z='z_val')
4. Interact with the plot:
   - Click + drag to rotate
   - Scroll to zoom
   - Shift + drag to pan
```

**Creating a Surface Plot (More Advanced):**

```python
# The app generates this, but you can customize:
fig = go.Figure(data=[go.Surface(
    z=demo_data[['x_val', 'y_val', 'z_val']].values,
    colorscale='Viridis'
)])

# Or create your own mathematical surface:
import numpy as np
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

fig = go.Figure(data=[go.Surface(z=Z, x=x, y=y)])
```

---

#### 5.1.4 Specialized Charts - For Specific Use Cases

> 💡 **Tips:** These charts solve specific visualization problems. You might not use them every day, but when you need them, they're incredibly useful!

**Chart-by-Chart Guide:**

| Chart | Button | Use Case | Example |
|-------|--------|----------|---------|
| **Pie** | `[Pie]` | Show proportions | Market share |
| **Sunburst** | `[Sunburst]` | Hierarchical proportions | Org chart + sizes |
| **Treemap** | `[Treemap]` | Nested categories | File sizes |
| **Funnel** | `[Funnel]` | Sequential stages | Sales pipeline |
| **Waterfall** | `[Waterfall]` | Cumulative changes | Financial P&L |
| **Candlestick** | `[Candle]` | Stock OHLC data | Stock prices |
| **Polar** | `[Polar]` | Circular/angular data | Wind direction |
| **Ternary** | `[Ternary]` | 3-way compositions | Chemical mixtures |

**Creating a Pie Chart:**

```
1. Load demo data (has 'category' column)
2. Click [Pie]
3. Generated code: 
   fig = px.pie(demo_data, names='category', 
                values='signal')
4. The pie chart shows distribution across categories!

Pro tip: For donut chart, modify in Command Window:
fig.update_traces(hole=0.4)  # Adds center hole
```

**Understanding Financial Charts (Candlestick):**

```
┌─────────────────────────────────────────────────────────────────────┐
│ CANDLESTICK ANATOMY                                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  GREEN (Up Day)          RED (Down Day)                             │
│                                                                     │
│       │ High                   │ High                               │
│       │                        │                                    │
│     ┌─┴─┐ Close             ┌─┴─┐ Open                             │
│     │   │                    │   │                                  │
│     │   │ Body               │   │ Body                             │
│     │   │                    │   │                                  │
│     └─┬─┘ Open              └─┬─┘ Close                            │
│       │                        │                                    │
│       │ Low                    │ Low                                │
│                                                                     │
│  Demo data includes: open, high, low, close columns               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

#### 5.1.5 Geographic Charts - Map Your Data

> 💡 **Tips:** Geographic visualization is a huge field. Our app supports basic mapping, which is often enough for most analysis needs!

**Overview Table:**

| Chart | Button | Data Needed | Best For |
|-------|--------|-------------|----------|
| **Scatter Geo** | `[Geo]` | lat, lon columns | Point locations |
| **Choropleth** | `[Choro]` | country/state codes + values | Regional statistics |
| **Globe** | `[Globe]` | lat, lon columns | Global view |

**Creating a Geographic Scatter:**

```
1. Load demo data (has lat, lon columns)
2. Click [Geo]
3. Generated code:
   fig = px.scatter_geo(demo_data, lat='lat', lon='lon')
4. Points appear on world map!

Customize:
- Add color: color='country'
- Add size: size='signal'
- Change projection: fig.update_geos(projection_type="natural earth")
```

**Creating a Choropleth (Colored Regions):**

```python
# The app generates this for choropleth:
fig = px.choropleth(demo_data, 
                    locations='iso_alpha',  # Country codes like 'USA', 'GBR'
                    color='signal',         # Values to show
                    hover_name='country')   # On-hover label
```

---

### 5.2 Property Editing System - Fine-Tune Everything

> 💡 **Tips:** The Property Inspector is like Photoshop's properties panel for charts. Every visual aspect can be adjusted without writing code!

The Property Inspector provides fine-grained control over every visual aspect of the figure.

#### 5.2.1 How to Use the Property Inspector

```
┌─────────────────────────────────────────────────────────────────────┐
│ PROPERTY INSPECTOR WORKFLOW                                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 1: Select What to Edit                                        │
│  ┌─────────────────────────────────────────┐                       │
│  │ Element dropdown shows:                  │                       │
│  │ • Figure Settings (global)               │                       │
│  │ • Trace 0: scatter                       │                       │
│  │ • Trace 1: line                          │                       │
│  │ • Shape 0: rect                          │                       │
│  │ • Annot 0: "My Label"                    │                       │
│  └─────────────────────────────────────────┘                       │
│                                                                     │
│  Step 2: Change Properties                                          │
│  ┌─────────────────────────────────────────┐                       │
│  │ Different controls appear based on what │                       │
│  │ you selected:                           │                       │
│  │                                          │                       │
│  │ For Trace:                               │                       │
│  │   Color: [  red  ▼]                      │                       │
│  │   Size:  [   10   ]                      │                       │
│  │   Symbol:[circle ▼]                      │                       │
│  │                                          │                       │
│  │ For Figure:                              │                       │
│  │   Title: [My Chart     ]                │                       │
│  │   Theme: [plotly_white▼]                │                       │
│  │   X Title:[Time        ]                │                       │
│  └─────────────────────────────────────────┘                       │
│                                                                     │
│  Step 3: Apply Changes                                              │
│  ┌─────────────────────────────────────────┐                       │
│  │ Click [Apply Changes] button             │                       │
│  │ Changes appear immediately on canvas!    │                       │
│  └─────────────────────────────────────────┘                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### 5.2.2 Trace Properties - 35+ Customization Options

**Line Properties:**

| Property | Control Type | Options | What It Does |
|----------|-------------|---------|--------------|
| Color | Dropdown | 17 named colors | Line/marker color |
| Size | Number | 1-50 | Marker diameter in pixels |
| Opacity | Number | 0-1 | Transparency (0=invisible) |
| Symbol | Dropdown | 10 shapes | Marker shape |
| Line Width | Number | 0-20 | Line thickness |
| Dash Style | Dropdown | solid, dot, dash, etc. | Line pattern |
| Mode | Dropdown | lines, markers, lines+markers | What to show |
| Fill | Dropdown | none, tozeroy, tonexty | Area filling |

**Available Colors (17 Named):**

```
Basic:    black, white, red, green, blue, cyan, magenta, yellow
Extended: orange, purple, grey, brown, pink, gold, teal, navy
Special:  transparent
```

**Available Marker Symbols (10 Basic):**

```
┌─────────────────────────────────────────────────────────────────────┐
│ MARKER SYMBOLS                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  circle    ●       square    ■       diamond   ◆                   │
│                                                                     │
│  cross     +       x         ×       triangle-up  ▲                │
│                                                                     │
│  triangle-down ▼   star      ★       hexagram  ✡                   │
│                                                                     │
│  pentagon  ⬠                                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### 5.2.3 Figure (Layout) Properties

| Property | Control | What It Does |
|----------|---------|--------------|
| Title | Text input | Figure title text |
| Width | Number | Figure width in pixels |
| Height | Number | Figure height in pixels |
| Plot Color | Color dropdown | Background of plot area |
| Paper Color | Color dropdown | Background outside plot |
| Font | Font dropdown | Global font family |
| Font Size | Number | Global base font size |
| Theme | Dropdown | Apply complete template |
| X Title | Text input | X-axis label |
| Y Title | Text input | Y-axis label |
| Legend | Show/Hide | Toggle legend visibility |
| Legend Dir | v/h | Vertical or horizontal |
| Legend Pos | Dropdown | Corner placement |
| Hover Mode | Dropdown | How tooltips appear |
| Grid X/Y | Show/Hide | Toggle grid lines |
| Bar Mode | Dropdown | grouped, stacked, etc. |
| X/Y Scale | linear/log | Axis scale type |
| Spikes | Show/Hide | Crosshair lines |
| Zero Line | Show/Hide | Line at y=0 |

#### 5.2.4 Annotation Properties

| Property | Control | What It Does |
|----------|---------|--------------|
| Text | Text input | Annotation content |
| Color | Color dropdown | Text color |
| Size | Number | Font size |
| Font | Font dropdown | Font family |
| X, Y | Number | Position |
| Arrow | Show/Hide | Display arrow |
| Bg Color | Color | Background fill |
| Angle | Number | Text rotation |

#### 5.2.5 Shape Properties

| Property | Control | What It Does |
|----------|---------|--------------|
| Line Color | Color | Outline color |
| Line Width | Number | Outline thickness |
| Opacity | Number | Transparency |
| Dash | Dropdown | Line pattern |
| Fill Color | Color | Interior fill |

### 5.3 Drawing and Annotation Tools - Make Your Charts Talk

> 💡 **Tips:** Annotations and shapes help you **tell a story** with your data. A well-placed arrow pointing to an outlier with a note like "Equipment malfunction on this day" can make your visualization much more meaningful!

PyFigureEditor provides MATLAB-style drawing tools for adding visual elements directly on your charts.

#### 5.3.1 The Drawing Toolbar - Your Creative Tools

```
┌─────────────────────────────────────────────────────────────────────┐
│ ANNOTATE TAB - DRAWING TOOLS                                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Shape Drawing:                                                     │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                          │
│  │Line │ │Rect │ │Circle│ │Free │ │Poly │                          │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘                          │
│     │       │        │       │       │                              │
│     ▼       ▼        ▼       ▼       ▼                              │
│  Draw    Draw    Draw     Draw    Draw                              │
│  straight rectangles circles  freehand closed                       │
│  lines                       paths   polygons                       │
│                                                                     │
│  Text & Images:                                                     │
│  ┌───────────┐ ┌───────────┐                                       │
│  │ Add Text  │ │ Add Image │                                       │
│  │ Annotation│ │ Upload    │                                       │
│  └───────────┘ └───────────┘                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**How Drawing Works:**

```
1. Click a drawing tool button (e.g., [Rect])
2. The canvas enters "drawing mode" (cursor changes)
3. Click and drag on the canvas to draw
4. Release mouse to complete the shape
5. Shape appears and can be edited in Property Inspector!
```

#### 5.3.2 Adding Text Annotations - Step by Step

```
┌─────────────────────────────────────────────────────────────────────┐
│ ADDING A TEXT ANNOTATION                                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 1: Click [Add Text] in ANNOTATE tab                          │
│          A modal dialog opens                                       │
│                                                                     │
│  Step 2: Fill in the dialog                                         │
│  ┌─────────────────────────────────────────┐                       │
│  │ Text:    [Important finding!       ]    │                       │
│  │ X Pos:   [0.5    ] (or click on graph)  │                       │
│  │ Y Pos:   [0.5    ] (or click on graph)  │                       │
│  │ Arrow:   [✓] Show Arrow                 │                       │
│  └─────────────────────────────────────────┘                       │
│                                                                     │
│  Step 3: Click [Confirm]                                            │
│          Annotation appears on canvas!                              │
│                                                                     │
│  Step 4: Customize in Property Inspector                            │
│  ┌─────────────────────────────────────────┐                       │
│  │ Select: "Annot 0: Important finding!"   │                       │
│  │                                          │                       │
│  │ Color:    [red   ▼]                      │                       │
│  │ Size:     [  14   ]                      │                       │
│  │ Font:     [Arial ▼]                      │                       │
│  │ Bg Color: [white ▼]                      │                       │
│  │                                          │                       │
│  │ [Apply Changes]                          │                       │
│  └─────────────────────────────────────────┘                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Pro Tips for Annotations:**

```python
# Annotations support basic HTML! Try these in the text field:

"Sales <b>increased</b> by 20%"      # Bold
"Temperature <i>dropped</i>"          # Italic
"Break<br>line"                       # Line break
"<sup>2</sup>nd peak"                 # Superscript
"H<sub>2</sub>O"                      # Subscript
```

#### 5.3.3 Adding Images - Overlay Reference Pictures

```
┌─────────────────────────────────────────────────────────────────────┐
│ ADDING A BACKGROUND IMAGE                                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Use Case: Overlay a reference image under your data                │
│  Example: Trace data points on top of a map image                   │
│                                                                     │
│  Step 1: Click [Add Image] (Upload button)                         │
│                                                                     │
│  Step 2: Select an image file                                       │
│          Supported: PNG, JPG, SVG                                  │
│                                                                     │
│  Step 3: Image appears as background layer                          │
│          Default: 50% opacity, stretched to fit                    │
│                                                                     │
│  Step 4: Adjust in Property Inspector                               │
│  ┌─────────────────────────────────────────┐                       │
│  │ Select: "Image 0"                        │                       │
│  │                                          │                       │
│  │ Opacity: [  0.3  ] (more transparent)   │                       │
│  │ Size X:  [  1.0  ] (paper fraction)     │                       │
│  │ Size Y:  [  1.0  ]                       │                       │
│  │ X:       [  0    ]                       │                       │
│  │ Y:       [  1    ]                       │                       │
│  │                                          │                       │
│  │ [Apply Changes]                          │                       │
│  └─────────────────────────────────────────┘                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 5.4 Data Management - Load, Edit, Clean

> 💡 **Tips:** Data management is the foundation of visualization. PyFigureEditor lets you load CSV files, generate demo data, view statistics, and even edit data directly in the app!

#### 5.4.1 Loading Data - Three Options

```
┌─────────────────────────────────────────────────────────────────────┐
│ DATA LOADING OPTIONS                                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Option 1: Import Your Own CSV                                      │
│  ┌─────────────────────────────────────────┐                       │
│  │ [Import CSV] → Select file → Done!      │                       │
│  │                                          │                       │
│  │ Supported: .csv files with headers      │                       │
│  │ Auto-detects: Column types              │                       │
│  └─────────────────────────────────────────┘                       │
│                                                                     │
│  Option 2: Generate Demo Data                                       │
│  ┌─────────────────────────────────────────┐                       │
│  │ [Load Demo] → Instant 200-row dataset!  │                       │
│  │                                          │                       │
│  │ Contains: numeric, categorical, dates,  │                       │
│  │          lat/lon, OHLC data for testing │                       │
│  │          ALL chart types!               │                       │
│  └─────────────────────────────────────────┘                       │
│                                                                     │
│  Option 3: Code Your Own Data                                       │
│  ┌─────────────────────────────────────────┐                       │
│  │ In Command Window:                       │                       │
│  │ my_data = pd.DataFrame({                │                       │
│  │     'x': [1, 2, 3, 4, 5],               │                       │
│  │     'y': [10, 20, 15, 25, 30]           │                       │
│  │ })                                       │                       │
│  │ figure_store.add_dataframe('my_data',   │                       │
│  │                            my_data)     │                       │
│  └─────────────────────────────────────────┘                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### 5.4.2 Viewing Data - Three Inspection Modes

```
DATA Tab buttons:

[View Table] → See raw data (first 100 rows, editable!)
[View Stats] → See statistical summary (count, mean, std, etc.)
[View Types] → See column data types (int64, float64, object, etc.)
```

**Example Statistics Output:**

```
┌──────────┬───────┬──────────┬───────────┬─────────┐
│ Column   │ count │ mean     │ std       │ min     │
├──────────┼───────┼──────────┼───────────┼─────────┤
│ signal   │ 200   │ 0.0234   │ 3.4521    │ -9.8765 │
│ x_val    │ 200   │ 0.1456   │ 10.0213   │ -28.234 │
│ y_val    │ 200   │ -0.0567  │ 9.8765    │ -25.123 │
└──────────┴───────┴──────────┴───────────┴─────────┘
```

#### 5.4.3 Editing Data - Direct Manipulation

```
┌─────────────────────────────────────────────────────────────────────┐
│ DATA EDITING FEATURES                                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Edit Cells in Table View                                        │
│     - Click [View Table]                                            │
│     - Click any cell to edit                                        │
│     - Type new value                                                │
│     - Changes sync automatically to data & chart!                  │
│                                                                     │
│  2. Delete Rows                                                     │
│     - Each row has a 🗑️ delete button                              │
│     - Click to remove that row                                     │
│     - Chart updates automatically                                   │
│                                                                     │
│  3. Remove Selected Points (Lasso Selection)                        │
│     - Use lasso tool on canvas to select points                    │
│     - Click [Remove Selected] in DATA tab                          │
│     - Points removed from BOTH chart AND data!                     │
│                                                                     │
│  4. Clean Data                                                      │
│     - Click [Clean NA] to:                                         │
│       • Convert string numbers to actual numbers                   │
│       • Remove rows with missing values (NaN)                      │
│                                                                     │
│  5. Delete Entire Dataset                                           │
│     - Select dataset in dropdown                                   │
│     - Click [Delete Data]                                          │
│     - Dataset and associated traces removed                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 5.5 Session Management - Save and Share Your Work

> 💡 **Tips:** Session saving lets you stop working, close the browser, and come back later to exactly where you left off! It also lets you share your visualization work with colleagues.

#### 5.5.1 What Gets Saved in a Session?

```
┌─────────────────────────────────────────────────────────────────────┐
│ SESSION FILE CONTENTS (.json)                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ✅ SAVED:                                                          │
│  • Complete figure (all traces, shapes, annotations)                │
│  • All layout settings (title, axes, theme, etc.)                  │
│  • Drawing history for undo/redo                                    │
│                                                                     │
│  ⚠️ NOT SAVED (for now):                                            │
│  • Loaded CSV data (re-import if needed)                           │
│  • Demo data (regenerate if needed)                                │
│  • Custom code in Command Window                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### 5.5.2 Saving Your Work

```
Step 1: Click [Save] in HOME tab
Step 2: Browser downloads: session.json
Step 3: Store the file somewhere safe!

Pro tip: Rename with meaningful name like:
  "quarterly_sales_analysis_2024Q3.json"
```

#### 5.5.3 Loading a Previous Session

```
Step 1: Click [Load] in HOME tab
Step 2: Select your .json session file
Step 3: Figure is restored exactly as saved!
        (You may need to reload data separately)
```

---

### 5.6 Code Export - Take Your Work Anywhere

> 💡 **Tips:** One of the BEST features! After creating your perfect visualization in the GUI, you can export the code to use in your own Python scripts, Jupyter notebooks, or share with colleagues who don't have PyFigureEditor!

#### 5.6.1 How Code Export Works

```
┌─────────────────────────────────────────────────────────────────────┐
│ CODE EXPORT FLOW                                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Your visual figure → Python code that recreates it                │
│                                                                     │
│  ┌────────────────────┐      ┌────────────────────────┐           │
│  │   📊               │      │ import plotly.express  │           │
│  │  Your beautiful    │  →   │ fig = px.scatter(...)  │           │
│  │    chart on        │      │ fig.update_layout(...) │           │
│  │     canvas         │      │ fig.show()             │           │
│  └────────────────────┘      └────────────────────────┘           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### 5.6.2 The Generated Code

When you create a scatter plot and click export, you get:

```python
import plotly.express as px
import pandas as pd

# Your data (you'll need to provide this)
# df = pd.read_csv('your_data.csv')

# Generated code from PyFigureEditor
fig = px.scatter(demo_abc123, x='x_val', y='y_val')

# Customizations applied in the GUI
fig.update_traces(
    marker=dict(color='red', size=12, symbol='circle'),
    opacity=0.8
)

fig.update_layout(
    title=dict(text='My Analysis', font=dict(size=20)),
    template='plotly_white',
    xaxis_title='X Values',
    yaxis_title='Y Values',
    showlegend=True
)

fig.show()
```

#### 5.6.3 Using Exported Code

```
1. Copy the code from Command Window
2. Paste into your Jupyter notebook or .py file
3. Replace data variable with your actual data
4. Run and enjoy your reproducible visualization!
```

---

### 5.7 Undo/Redo System - Never Lose Your Work

> 💡 **Tips:** Made a mistake? Don't worry! The undo/redo system remembers up to 50 steps, so you can always go back!

#### 5.7.1 What Can Be Undone?

| Action | Undoable? | Notes |
|--------|-----------|-------|
| Add trace | ✅ Yes | Removes the added trace |
| Delete trace | ✅ Yes | Restores the deleted trace |
| Change color | ✅ Yes | Reverts to previous color |
| Change theme | ✅ Yes | Reverts template |
| Add annotation | ✅ Yes | Removes annotation |
| Draw shape | ✅ Yes | Removes shape |
| Move element | ✅ Yes | Returns to original position |
| Delete data | ⚠️ Partial | Chart undone, data not restored |

#### 5.7.2 Using Undo/Redo

```
Method 1: Buttons
  [Undo] ← in HOME tab
  [Redo] → in HOME tab

Method 2: (If supported by your browser)
  Ctrl+Z = Undo
  Ctrl+Y = Redo
```

#### 5.7.3 Understanding the History Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│ HISTORY STACK EXAMPLE                                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Actions you take:                                                  │
│  1. Load data         → State 1 saved                              │
│  2. Add scatter       → State 2 saved                              │
│  3. Change to red     → State 3 saved                              │
│  4. Add title         → State 4 saved (current)                    │
│                                                                     │
│  UNDO Stack: [1, 2, 3, 4]    REDO Stack: []                        │
│                                                                     │
│  Click Undo:                                                        │
│  UNDO Stack: [1, 2, 3]       REDO Stack: [4]                       │
│  (Title removed, can redo to restore)                              │
│                                                                     │
│  Click Undo again:                                                  │
│  UNDO Stack: [1, 2]          REDO Stack: [3, 4]                    │
│  (Color back to default)                                           │
│                                                                     │
│  Click Redo:                                                        │
│  UNDO Stack: [1, 2, 3]       REDO Stack: [4]                       │
│  (Red color restored)                                              │
│                                                                     │
│  If you now make a NEW action (add line):                          │
│  UNDO Stack: [1, 2, 3, 5]    REDO Stack: [] ← cleared!             │
│  (Can't redo state 4 anymore - standard undo behavior)             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 5.8 Template System - Instant Professional Styling

> 💡 **Tips:** Templates are like Instagram filters for your charts. One click and your entire visualization gets a coordinated color scheme and style!

#### 5.8.1 Available Templates (10 Built-in)

| Template | Visual Style | Best For |
|----------|-------------|----------|
| `plotly` | Default Plotly colors | General use |
| `plotly_white` | White background | Publications, reports |
| `plotly_dark` | Dark background | Presentations, dashboards |
| `ggplot2` | R ggplot2 style | Statistical graphics |
| `seaborn` | Python seaborn style | Data science |
| `simple_white` | Minimal, clean | Academic papers |
| `presentation` | Large fonts, bold | Slideshows |
| `xgridoff` | No vertical grid | Time series focus |
| `ygridoff` | No horizontal grid | Bar chart focus |
| `gridon` | Full grid visible | Technical/engineering |

#### 5.8.2 Applying a Template

```
Method 1: Property Inspector
  - Select "Figure Settings"
  - Find "Theme" dropdown
  - Select template
  - Click [Apply Changes]

Method 2: Command Window
  fig.update_layout(template='plotly_dark')
```

#### 5.8.3 Template Visual Comparison

```
┌─────────────────────────────────────────────────────────────────────┐
│ TEMPLATE COMPARISON                                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  plotly_white:        plotly_dark:        ggplot2:                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
│  │ ░░░░░░░░░░░░ │    │ ▓▓▓▓▓▓▓▓▓▓▓▓ │    │ ▒▒▒▒▒▒▒▒▒▒▒▒ │         │
│  │ ░░ ● ●  ●░░ │    │ ▓▓ ○ ○  ○▓▓ │    │ ▒▒ ◆ ◆  ◆▒▒ │         │
│  │ ░░░●░░●░░░░ │    │ ▓▓▓○▓▓○▓▓▓▓ │    │ ▒▒▒◆▒▒◆▒▒▒▒ │         │
│  │ ░░░░░●░░░░░ │    │ ▓▓▓▓▓○▓▓▓▓▓ │    │ ▒▒▒▒▒◆▒▒▒▒▒ │         │
│  └──────────────┘    └──────────────┘    └──────────────┘         │
│  White bg, blue      Dark bg, bright    Gray bg, bold            │
│  markers             markers            markers                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 5.9 View Tools - Navigate Your Visualization

#### 5.9.1 Navigation Controls (VIEW Tab)

```
[Zoom] - Click+drag to zoom into a region
[Pan]  - Click+drag to move around
[Reset] - Return to original view (autoscale)
[Inspector Toggle] - Show/hide right panel
```

#### 5.9.2 Interactive Features (Built into Canvas)

```
• Scroll wheel = Zoom in/out
• Double-click = Reset zoom
• Hover = See data values in tooltip
• Click legend item = Toggle trace visibility
• Drag legend = Reposition
```

---

### 5.10 Selection and Statistics - Interactive Analysis

> 💡 **Tips:** This feature lets you select points directly on the chart and see statistics about your selection - like a quick "what's in this cluster?" analysis!

#### 5.10.1 Lasso Selection

```
1. Use the lasso tool (in plotly modebar)
2. Draw around points you want to analyze
3. Console shows: ">>> Selected 42 points. Y-Stats: Mean=15.3, Min=2.1, Max=28.7"
```

#### 5.10.2 Box Selection

```
1. Use the box select tool
2. Draw rectangle around points
3. Same statistics appear in console
```

#### 5.10.3 Remove Selected Points

```
After selecting points:
1. Go to DATA tab
2. Click [Remove Selected]
3. Points removed from BOTH:
   - The visual chart
   - The underlying dataset!
```
