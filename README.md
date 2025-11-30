# PyFigureEditor: A Python-Based Interactive Scientific Visualization Platform with MATLAB-Style Editing Capabilities

<div align="center">

## MATH 4710 - Final Project Report

### A Comprehensive Technical Documentation and Project Report

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

*Click the link above to experience the fully functional application without any installation.*



</div>

---

## Abstract

This report presents **PyFigureEditor**, a comprehensive web-based interactive scientific visualization platform developed in Python that replicates and extends the functionality of MATLAB's Figure Tool. The project addresses a fundamental limitation in traditional Python plotting workflows: the inability to interactively edit and modify visualizations after their initial creation without re-executing code.

The application provides a complete graphical user interface (GUI) featuring **26+ chart types**, **real-time property editing**, **drawing and annotation tools**, **session management**, and **automatic Python code generation**. Built upon the Dash framework with Plotly.js as the visualization engine, the system implements a sophisticated **reactive callback architecture** for seamless user interaction.

Two implementation versions are provided: (1) a **Jupyter Notebook version** (`Final_Project_Implementation.ipynb`) optimized for educational and development environments, and (2) a **standalone server application** (`app.py`) suitable for production deployment. The application has been successfully deployed to PythonAnywhere and is accessible at **https://zye.pythonanywhere.com/**.

Key technical contributions include: a centralized **FigureStore state management system**, a **dynamic UI generation mechanism** for context-sensitive property editing, a **smart code generator** with automatic column type inference, and a comprehensive **undo/redo history stack**. The system demonstrates that Python can provide an interactive data visualization experience comparable to commercial software like MATLAB.

**Keywords:** Interactive Visualization, Python, Dash, Plotly, GUI Development, Scientific Computing, MATLAB Alternative, Web Application, Data Science Tools

---

## Table of Contents

- [PyFigureEditor: A Python-Based Interactive Scientific Visualization Platform with MATLAB-Style Editing Capabilities](#pyfigureeditor-a-python-based-interactive-scientific-visualization-platform-with-matlab-style-editing-capabilities)
  - [MATH 4710 - Final Project Report](#math-4710---final-project-report)
    - [A Comprehensive Technical Documentation and Academic Analysis](#a-comprehensive-technical-documentation-and-academic-analysis)
    - [🌐 Live Production Deployment](#-live-production-deployment)
  - [**https://zye.pythonanywhere.com/**](#httpszyepythonanywherecom)
  - [Abstract](#abstract)
  - [Table of Contents](#table-of-contents)
  - [1. Introduction](#1-introduction)
    - [1.1 Project Background and Motivation](#11-project-background-and-motivation)
    - [1.2 Problem Statement](#12-problem-statement)
    - [1.3 Project Objectives](#13-project-objectives)
    - [1.4 Scope and Deliverables](#14-scope-and-deliverables)
  - [2. Literature Review and Technical Foundation](#2-literature-review-and-technical-foundation)
    - [2.1 Overview of Python Visualization Libraries](#21-overview-of-python-visualization-libraries)
      - [2.1.1 Matplotlib: The Foundation](#211-matplotlib-the-foundation)
      - [2.1.2 Seaborn: Statistical Visualization](#212-seaborn-statistical-visualization)
      - [2.1.3 Bokeh: Interactive Web Visualizations](#213-bokeh-interactive-web-visualizations)
      - [2.1.4 Plotly: The Selected Foundation](#214-plotly-the-selected-foundation)
      - [2.1.5 Comparative Analysis Summary](#215-comparative-analysis-summary)
    - [2.2 MATLAB Figure Tool Analysis](#22-matlab-figure-tool-analysis)
      - [2.2.1 MATLAB Figure Tool Architecture](#221-matlab-figure-tool-architecture)
      - [2.2.2 Feature Mapping: MATLAB to PyFigureEditor](#222-feature-mapping-matlab-to-pyfigureeditor)
      - [2.2.3 Interaction Paradigms](#223-interaction-paradigms)
    - [2.3 Web-Based GUI Frameworks for Python](#23-web-based-gui-frameworks-for-python)
      - [2.3.1 Framework Comparison](#231-framework-comparison)
      - [2.3.2 Why Dash Was Selected](#232-why-dash-was-selected)
      - [2.3.3 Dash Architecture Overview](#233-dash-architecture-overview)
    - [2.4 Reactive Programming Paradigm](#24-reactive-programming-paradigm)
      - [2.4.1 What is Reactive Programming?](#241-what-is-reactive-programming)
      - [2.4.2 Dash's Callback System](#242-dashs-callback-system)
      - [2.4.3 Callback Graph and Execution Order](#243-callback-graph-and-execution-order)
      - [2.4.4 Handling Multiple Outputs](#244-handling-multiple-outputs)
  - [3. System Architecture and Design](#3-system-architecture-and-design)
    - [3.1 High-Level System Architecture](#31-high-level-system-architecture)
    - [3.2 Core Class Design](#32-core-class-design)
      - [3.2.1 FigureStore Class](#321-figurestore-class)
      - [3.2.2 HistoryStack Class](#322-historystack-class)
      - [3.2.3 TraceDataset Class](#323-tracedataset-class)
      - [3.2.4 CodeGenerator Class](#324-codegenerator-class)
    - [3.3 Component Interaction Diagram](#33-component-interaction-diagram)
    - [3.4 Layout Architecture](#34-layout-architecture)
    - [3.5 Data Flow Architecture](#35-data-flow-architecture)
    - [3.6 Design Decisions and Trade-offs](#36-design-decisions-and-trade-offs)
    - [3.7 Extensibility Points](#37-extensibility-points)
  - [4. Core Implementation Details](#4-core-implementation-details)
    - [4.1 Application Initialization and Configuration](#41-application-initialization-and-configuration)
    - [4.2 Layout Implementation](#42-layout-implementation)
      - [4.2.1 Main Layout Structure](#421-main-layout-structure)
      - [4.2.2 Header Component Implementation](#422-header-component-implementation)
      - [4.2.3 Graph Panel Implementation](#423-graph-panel-implementation)
      - [4.2.4 Property Inspector Implementation](#424-property-inspector-implementation)
    - [4.3 Callback Implementation](#43-callback-implementation)
      - [4.3.1 Add Trace Callback](#431-add-trace-callback)
      - [4.3.2 Element Selection Update Callback](#432-element-selection-update-callback)
      - [4.3.3 Apply Properties Callback](#433-apply-properties-callback)
      - [4.3.4 Delete Element Callback](#434-delete-element-callback)
    - [4.4 Helper Functions](#44-helper-functions)
      - [4.4.1 Data Parsing Functions](#441-data-parsing-functions)
      - [4.4.2 History Management Functions](#442-history-management-functions)
      - [4.4.3 Nested Property Access Functions](#443-nested-property-access-functions)
  - [5. Complete Feature Documentation](#5-complete-feature-documentation)
    - [5.1 Plot Types and Chart Creation](#51-plot-types-and-chart-creation)
      - [5.1.1 Basic Charts](#511-basic-charts)
      - [5.1.2 Statistical Charts](#512-statistical-charts)
      - [5.1.3 3D Charts](#513-3d-charts)
      - [5.1.4 Specialized Charts](#514-specialized-charts)
      - [5.1.5 Geographic Charts](#515-geographic-charts)
    - [5.2 Property Editing System](#52-property-editing-system)
      - [5.2.1 Trace Properties (35+ Properties)](#521-trace-properties-35-properties)
      - [5.2.2 Layout Properties](#522-layout-properties)
      - [5.2.3 Annotation Properties](#523-annotation-properties)
      - [5.2.4 Shape Properties](#524-shape-properties)
    - [5.3 Drawing and Annotation Tools](#53-drawing-and-annotation-tools)
      - [5.3.1 Drawing Mode Toolbar](#531-drawing-mode-toolbar)
      - [5.3.2 Adding Text Annotations](#532-adding-text-annotations)
      - [5.3.3 Adding Arrows](#533-adding-arrows)
      - [5.3.4 Adding Images](#534-adding-images)
    - [5.4 Session Management](#54-session-management)
      - [5.4.1 Saving Sessions](#541-saving-sessions)
      - [5.4.2 Loading Sessions](#542-loading-sessions)
      - [5.4.3 Session Compatibility](#543-session-compatibility)
    - [5.5 Code Export](#55-code-export)
      - [5.5.1 Export Formats](#551-export-formats)
      - [5.5.2 Export Process](#552-export-process)
    - [5.6 Undo/Redo System](#56-undoredo-system)
      - [5.6.1 Supported Operations](#561-supported-operations)
      - [5.6.2 Usage](#562-usage)
      - [5.6.3 Behavior Notes](#563-behavior-notes)
    - [5.7 Template System](#57-template-system)
      - [5.7.1 Available Templates](#571-available-templates)
      - [5.7.2 Applying Templates](#572-applying-templates)
  - [6. Deployment Guide](#6-deployment-guide)
    - [6.1 Two Deployment Versions](#61-two-deployment-versions)
    - [6.2 Local Development Setup](#62-local-development-setup)
      - [6.2.1 Prerequisites](#621-prerequisites)
      - [6.2.2 Running Locally](#622-running-locally)
      - [6.2.3 Development Mode Features](#623-development-mode-features)
    - [6.3 PythonAnywhere Deployment](#63-pythonanywhere-deployment)
      - [6.3.1 Account Setup](#631-account-setup)
      - [6.3.2 File Upload](#632-file-upload)
      - [6.3.3 Web App Configuration](#633-web-app-configuration)
      - [6.3.4 WSGI Configuration](#634-wsgi-configuration)
      - [6.3.5 Virtual Environment (Recommended)](#635-virtual-environment-recommended)
      - [6.3.6 Reload and Test](#636-reload-and-test)
      - [6.3.7 Troubleshooting PythonAnywhere](#637-troubleshooting-pythonanywhere)
    - [6.4 Google Colab Deployment](#64-google-colab-deployment)
      - [6.4.1 Colab-Specific Setup](#641-colab-specific-setup)
      - [6.4.2 Colab Run Modes](#642-colab-run-modes)
      - [6.4.3 Colab Limitations](#643-colab-limitations)
    - [6.5 Alternative Deployment Platforms](#65-alternative-deployment-platforms)
      - [6.5.1 Heroku Deployment](#651-heroku-deployment)
      - [6.5.2 AWS Elastic Beanstalk](#652-aws-elastic-beanstalk)
      - [6.5.3 Docker Deployment](#653-docker-deployment)
    - [6.6 Production Considerations](#66-production-considerations)
      - [6.6.1 Security](#661-security)
      - [6.6.2 Performance Optimization](#662-performance-optimization)
      - [6.6.3 Monitoring](#663-monitoring)

---

## 1. Introduction

### 1.1 Project Background and Motivation

The genesis of this project stems from a direct requirement articulated by Professor Puneet Rana during the MATH 4710 course. On October 29, 2024, Professor Rana presented a reference video demonstrating MATLAB's Figure Tool capabilities and posed the following challenge:

> *"Hello Tony, I need similar Editable framework using Python... After you construct the image in Python, you should have some option on the graph to Edit graph."*
> — Professor Puneet Rana, October 29, 2024

This request highlighted a significant gap in the Python data science ecosystem. While Python has become the de facto standard for data analysis and machine learning, its visualization workflow remains fundamentally different from MATLAB's interactive approach. In MATLAB, users can create a figure and then use the Figure Tool to:

- Zoom and pan interactively
- Click on data points to see their values (datatips)
- Add text annotations and arrows
- Modify colors, line styles, and markers
- Adjust layout and axis properties
- Save the modified figure

In contrast, traditional Python visualization workflows follow a "code-execute-view" paradigm where any modification requires editing source code and re-running the script. This creates friction in the data exploration process and presents a barrier for users transitioning from MATLAB to Python.

The MATLAB Figure Tool reference video ([YouTube Link](https://www.youtube.com/watch?v=owKwqPyg5bk)) demonstrates the following key capabilities that served as design targets for this project:

| Timestamp | Feature | Description |
|-----------|---------|-------------|
| 0:00 | Figure Tool Overview | Introduction to the interactive editing interface |
| 0:20 | Save Figure | Exporting the current visualization |
| 0:47 | Zoom by Scrolling | Mouse wheel zoom functionality |
| 0:56 | Restore Home View | Reset to original viewport |
| 1:01 | Zoom In and Out | Dedicated zoom controls |
| 1:17 | Pan with Hand | Click-and-drag navigation |
| 1:24 | Datatips | Hover to show data point values |
| 1:51 | Adjust Plot Layout | Modify figure dimensions and margins |
| 2:34 | Insert Plot Features | Add annotations, shapes, and text |

Professor Rana further clarified the requirements in subsequent communications:

> *"No.. after I run python code.. Your graphics should be editable like MATLAB. It's like GUI - Graphic User Interface."*
> — Professor Puneet Rana, October 31, 2024

And provided a technical reference for GUI development approaches:

> *"There are many ways to make GUI.. you can choose best as per your level."*
> — Professor Puneet Rana, November 1, 2024 (with [StackOverflow reference](https://stackoverflow.com/questions/15507009/can-python-make-matlab-style-guis))

### 1.2 Problem Statement

The fundamental problem addressed by this project can be formally stated as follows:

**Primary Problem:** Python's standard visualization libraries (Matplotlib, Seaborn, Plotly) generate static or semi-interactive outputs that cannot be fully edited through a graphical interface after creation. Users must return to source code to make modifications, breaking the flow of data exploration.

**Secondary Problems:**

1. **Steep Learning Curve:** Each visualization library has its own API, requiring users to memorize extensive documentation to make even simple changes.

2. **No Unified Interface:** Different plot types often require different syntax, making it difficult to switch between visualizations.

3. **Limited Persistence:** There is no standard way to save an "editing session" that preserves all modifications made to a figure.

4. **Code-Visualization Disconnect:** Changes made through limited interactive features (like Plotly's built-in zoom) are not reflected back to code, making reproducibility challenging.

**Formal Requirements:**

The solution must provide:

| Requirement ID | Description | Priority |
|----------------|-------------|----------|
| R1 | Web-based GUI accessible through a browser | Critical |
| R2 | Support for multiple chart types (minimum 10) | Critical |
| R3 | Real-time property editing without code changes | Critical |
| R4 | Drawing and annotation capabilities | High |
| R5 | Undo/redo functionality | High |
| R6 | Session save/load functionality | High |
| R7 | Automatic Python code generation | Medium |
| R8 | Data import/export capabilities | Medium |
| R9 | Production deployment capability | Medium |

### 1.3 Project Objectives

The objectives of this project are organized into three tiers:

**Tier 1 - Core Objectives (Must Have):**

1. Develop a functional web-based GUI that allows users to create and edit Plotly figures interactively
2. Implement a property inspector that displays and allows modification of element properties
3. Provide at least 15 different chart types
4. Enable zoom, pan, and reset view controls
5. Deploy the application to a publicly accessible server

**Tier 2 - Enhanced Objectives (Should Have):**

1. Implement drawing tools for shapes (line, rectangle, circle)
2. Add text annotation capabilities with arrow support
3. Create an undo/redo system with history stack
4. Develop session save/load functionality
5. Build a data management interface for CSV import

**Tier 3 - Advanced Objectives (Nice to Have):**

1. Automatic Python code generation from current figure state
2. Smart column type inference for plot generation
3. Real-time data editing through DataTable integration
4. Background image overlay support
5. Multiple theme presets

### 1.4 Scope and Deliverables

**In Scope:**

- Web-based application using Dash/Plotly
- 2D and 3D chart types
- Geographic/map visualizations
- Statistical plots (box, violin, histogram)
- Financial charts (candlestick)
- Interactive property editing
- Shape drawing and annotation
- Session management
- Deployment to PythonAnywhere

**Out of Scope:**

- Real-time collaborative editing (multi-user)
- Mobile-native applications
- Offline desktop executables
- Integration with external databases
- Animation/video export
- Machine learning model integration

**Deliverables:**

| Deliverable | Description | Location |
|-------------|-------------|----------|
| D1 | Jupyter Notebook Implementation | `Final_Project_Implementation.ipynb` |
| D2 | Standalone Python Application | `app.py` |
| D3 | Technical Documentation | `README.md` (this document) |
| D4 | Live Deployment | https://zye.pythonanywhere.com/ |
| D5 | Source Code Repository | GitHub Repository |

---

## 2. Literature Review and Technical Foundation

This section provides a comprehensive analysis of the existing technologies, frameworks, and approaches that informed the design and implementation of PyFigureEditor. Understanding this foundation is essential for appreciating the architectural decisions made throughout the project.

### 2.1 Overview of Python Visualization Libraries

Python's ecosystem offers numerous visualization libraries, each with distinct philosophies, capabilities, and limitations. A thorough understanding of these options was necessary to select the most appropriate foundation for this project.

#### 2.1.1 Matplotlib: The Foundation

Matplotlib, created by John D. Hunter in 2003, is the foundational visualization library in Python. It follows an object-oriented architecture with two primary interfaces:

```python
# Pyplot Interface (MATLAB-like)
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title('Simple Plot')
plt.show()

# Object-Oriented Interface
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
ax.set_title('Simple Plot')
plt.show()
```

**Strengths:**
- Highly customizable with fine-grained control
- Extensive documentation and community support
- Integration with NumPy and Pandas
- Publication-quality static figures

**Limitations for This Project:**
- Primarily designed for static output
- Limited built-in interactivity
- Web integration requires additional libraries (mpld3, Bokeh backend)
- No native GUI for property editing

#### 2.1.2 Seaborn: Statistical Visualization

Seaborn, built on top of Matplotlib, provides a high-level interface for statistical graphics:

```python
import seaborn as sns
tips = sns.load_dataset("tips")
sns.boxplot(x="day", y="total_bill", data=tips)
```

**Strengths:**
- Beautiful default styles
- Built-in statistical aggregations
- Excellent for exploratory data analysis

**Limitations for This Project:**
- Inherits Matplotlib's static nature
- No interactive editing capabilities
- Limited chart type variety beyond statistics

#### 2.1.3 Bokeh: Interactive Web Visualizations

Bokeh is designed specifically for creating interactive visualizations for web browsers:

```python
from bokeh.plotting import figure, show
p = figure(title="Interactive Plot", tools="pan,wheel_zoom,box_zoom,reset")
p.circle([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], size=15)
show(p)
```

**Strengths:**
- Native web rendering
- Good interactivity (zoom, pan, hover)
- Bokeh Server for real-time updates
- Standalone HTML output

**Limitations for This Project:**
- Less extensive chart type library than Plotly
- Callback system less intuitive than Dash
- Property editing requires custom implementation

#### 2.1.4 Plotly: The Selected Foundation

Plotly emerged as the optimal choice for this project. It provides:

```python
import plotly.express as px
import plotly.graph_objects as go

# High-level Express API
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")

# Low-level Graph Objects API
fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6])])
fig.update_layout(title="My Plot")
```

**Strengths Critical for This Project:**
1. **Comprehensive Chart Library:** 40+ chart types including 3D, maps, and financial charts
2. **Native Interactivity:** Zoom, pan, hover, selection built-in
3. **JSON-Based Architecture:** Figures are serializable dictionaries
4. **Dash Integration:** First-class support for Dash applications
5. **Editable Mode:** Built-in `config={'editable': True}` option

**Plotly's Figure Architecture:**

Understanding Plotly's internal structure was crucial for implementing the property editor:

```python
fig.to_dict()
# Returns:
{
    "data": [
        {
            "type": "scatter",
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "mode": "lines+markers",
            "marker": {"color": "blue", "size": 10},
            "line": {"width": 2, "dash": "solid"},
            "name": "Trace 0"
        }
    ],
    "layout": {
        "title": {"text": "My Figure"},
        "xaxis": {"title": {"text": "X Axis"}},
        "yaxis": {"title": {"text": "Y Axis"}},
        "template": "plotly_white",
        "shapes": [...],
        "annotations": [...],
        "images": [...]
    }
}
```

This dictionary structure enables:
- Direct property access and modification
- JSON serialization for session save/load
- Programmatic figure reconstruction

#### 2.1.5 Comparative Analysis Summary

| Feature | Matplotlib | Seaborn | Bokeh | Plotly |
|---------|------------|---------|-------|--------|
| Web Native | ❌ | ❌ | ✅ | ✅ |
| Interactivity | Limited | Limited | Good | Excellent |
| Chart Types | Many | Statistical | Many | Most |
| JSON Serializable | ❌ | ❌ | Partial | ✅ |
| Dash Integration | Via Plotly | Via Plotly | Separate | Native |
| Learning Curve | Steep | Moderate | Moderate | Moderate |
| 3D Support | Basic | ❌ | Limited | Excellent |
| Geo/Maps | Basemap | ❌ | Good | Excellent |

**Conclusion:** Plotly's comprehensive feature set, JSON-based architecture, and native Dash integration made it the clear choice for implementing a MATLAB-style interactive figure editor.

### 2.2 MATLAB Figure Tool Analysis

To replicate MATLAB's functionality, a detailed analysis of the Figure Tool was conducted based on the reference video and documentation.

#### 2.2.1 MATLAB Figure Tool Architecture

MATLAB's Figure Tool operates on a hierarchical object model:

```
Figure (gcf)
├── Axes (gca)
│   ├── Line objects
│   ├── Scatter objects
│   ├── Bar objects
│   ├── Surface objects
│   └── Text objects
├── UI Controls
├── Legends
└── Colorbars
```

Each object has properties that can be accessed and modified:

```matlab
% MATLAB property access
h = plot(1:10, rand(1,10));
h.Color = 'red';
h.LineWidth = 2;
h.Marker = 'o';
```

#### 2.2.2 Feature Mapping: MATLAB to PyFigureEditor

| MATLAB Feature | MATLAB Implementation | PyFigureEditor Implementation |
|----------------|----------------------|------------------------------|
| Property Inspector | Built-in GUI panel | Custom Dash components |
| Zoom | `zoom on` command | Plotly native + buttons |
| Pan | `pan on` command | Plotly native + buttons |
| Data Cursor | `datacursormode on` | Plotly hover tooltips |
| Insert Text | `gtext()` function | Annotation modal |
| Insert Arrow | Arrow annotation | Annotation with `showarrow=True` |
| Insert Shape | Rectangle, ellipse tools | Drawing mode buttons |
| Save Figure | `saveas()`, `exportgraphics()` | JSON session export |
| Edit Plot | Property editor GUI | Dynamic property inspector |

#### 2.2.3 Interaction Paradigms

MATLAB uses a **modal** interaction system where the figure enters different "modes":
- Select mode (default)
- Zoom mode
- Pan mode
- Data cursor mode
- Edit mode

PyFigureEditor replicates this through Plotly's `dragmode` property:

```python
# Equivalent to MATLAB's zoom on
fig.update_layout(dragmode="zoom")

# Equivalent to MATLAB's pan on  
fig.update_layout(dragmode="pan")

# Drawing modes
fig.update_layout(dragmode="drawrect")
fig.update_layout(dragmode="drawline")
fig.update_layout(dragmode="drawcircle")
```

### 2.3 Web-Based GUI Frameworks for Python

Several frameworks were evaluated for building the web-based GUI component.

#### 2.3.1 Framework Comparison

| Framework | Type | Learning Curve | Plotly Integration | Real-time Updates |
|-----------|------|----------------|-------------------|-------------------|
| Flask + JS | Traditional | High | Manual | WebSocket |
| Django + JS | Traditional | High | Manual | Channels |
| Streamlit | Declarative | Low | Good | Automatic |
| Gradio | ML-focused | Low | Limited | Automatic |
| Panel | HoloViews | Moderate | Good | Good |
| **Dash** | Reactive | Moderate | **Native** | **Callbacks** |

#### 2.3.2 Why Dash Was Selected

**Dash** (by Plotly) was selected for the following reasons:

1. **Native Plotly Integration:** Dash is built by the same team that created Plotly. The `dcc.Graph` component accepts Plotly figures directly.

2. **Reactive Programming Model:** Dash's callback system provides a clean way to handle user interactions without writing JavaScript.

3. **Component Ecosystem:** Dash Bootstrap Components (dbc) provides pre-built UI elements that match modern design standards.

4. **Production Ready:** Dash applications can be deployed to standard WSGI servers.

5. **No JavaScript Required:** The entire application can be written in Python.

#### 2.3.3 Dash Architecture Overview

Dash applications consist of two parts:

**1. Layout (Declarative UI):**
```python
app.layout = html.Div([
    dcc.Graph(id='my-graph'),
    dcc.Slider(id='my-slider', min=0, max=10, value=5),
    html.Div(id='output')
])
```

**2. Callbacks (Reactive Logic):**
```python
@app.callback(
    Output('my-graph', 'figure'),
    Input('my-slider', 'value')
)
def update_graph(slider_value):
    # This function runs when slider changes
    return create_figure(slider_value)
```

This separation of concerns maps well to the Model-View-Controller (MVC) pattern:
- **Model:** Python data structures (DataFrames, FigureStore)
- **View:** Dash layout components
- **Controller:** Callback functions

### 2.4 Reactive Programming Paradigm

Understanding reactive programming is essential for comprehending how PyFigureEditor responds to user interactions.

#### 2.4.1 What is Reactive Programming?

Reactive programming is a declarative paradigm where the program specifies **what** should happen in response to events, rather than **how** to poll for changes.

**Traditional (Imperative) Approach:**
```python
while True:
    if button_clicked():
        update_display()
    if slider_moved():
        recalculate()
    sleep(0.1)  # Polling
```

**Reactive (Declarative) Approach:**
```python
@when(button.clicked)
def handle_click():
    update_display()

@when(slider.changed)
def handle_slider():
    recalculate()
```

#### 2.4.2 Dash's Callback System

Dash implements reactivity through **callbacks** with three types of dependencies:

| Dependency | Purpose | Triggers Callback |
|------------|---------|-------------------|
| `Input` | Values that trigger the callback | Yes |
| `State` | Values read but don't trigger | No |
| `Output` | Components updated by callback | N/A |

**Example with All Three:**
```python
@app.callback(
    Output('result', 'children'),      # What to update
    Input('submit-btn', 'n_clicks'),   # What triggers update
    State('input-field', 'value')      # Additional data needed
)
def process_form(n_clicks, input_value):
    if not n_clicks:
        raise PreventUpdate
    return f"Processed: {input_value}"
```

#### 2.4.3 Callback Graph and Execution Order

Dash automatically builds a **dependency graph** from all callbacks:

```
User clicks "Apply" button
        │
        ▼
┌─────────────────────────────────┐
│  Callback: apply_property_changes │
│  Inputs: btn-apply-props.n_clicks │
│  Outputs: main-graph.figure       │
└─────────────────────────────────┘
        │
        ▼ (figure changed)
┌─────────────────────────────────┐
│  Callback: update_element_options │
│  Inputs: main-graph.figure        │
│  Outputs: dd-element-select.options│
└─────────────────────────────────┘
        │
        ▼ (options changed)
┌─────────────────────────────────┐
│  Callback: update_inspector_controls│
│  Inputs: dd-element-select.value   │
│  Outputs: inspector-controls.children│
└─────────────────────────────────┘
```

This automatic chaining is fundamental to how PyFigureEditor maintains consistency between the figure and the property inspector.

#### 2.4.4 Handling Multiple Outputs

Dash 2.x introduced `allow_duplicate=True`, enabling multiple callbacks to write to the same output:

```python
# Callback 1: Apply properties
@app.callback(
    Output("main-graph", "figure", allow_duplicate=True),
    Input("btn-apply-props", "n_clicks"),
    ...
)

# Callback 2: Drawing tools
@app.callback(
    Output("main-graph", "figure", allow_duplicate=True),
    Input("btn-draw-line", "n_clicks"),
    ...
)

# Callback 3: Undo/Redo
@app.callback(
    Output("main-graph", "figure", allow_duplicate=True),
    Input("btn-undo", "n_clicks"),
    ...
)
```

This pattern is essential for PyFigureEditor, where many different user actions can modify the same figure.

---

## 3. System Architecture and Design

This chapter presents the architectural blueprint of PyFigureEditor, explaining how different components interact to create a cohesive, maintainable, and extensible system.

### 3.1 High-Level System Architecture

PyFigureEditor follows a **layered architecture** pattern that separates concerns and promotes modularity:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   Toolbar   │  │   Graph     │  │  Property   │  │   Modal     │ │
│  │   Panel     │  │   Display   │  │  Inspector  │  │   Dialogs   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     CALLBACK LAYER (Controller)                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  25+ Callback Functions Handling User Interactions            │   │
│  │  • Figure creation/modification  • Property updates           │   │
│  │  • Element selection            • Drawing operations          │   │
│  │  • Undo/Redo                    • Session management         │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     BUSINESS LOGIC LAYER (Model)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ FigureStore │  │HistoryStack │  │TraceDataset │  │CodeGenerator│ │
│  │   Class     │  │    Class    │  │    Class    │  │    Class    │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                       │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  Plotly Figure Dictionary  │  JSON Session Storage             ││
│  │  {data: [...], layout: {...}}  │  {figure, datasets, code}     ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Core Class Design

The business logic is encapsulated in four primary classes, each with a single responsibility.

#### 3.2.1 FigureStore Class

**Purpose:** Centralized management of the Plotly figure state.

**Design Pattern:** Facade Pattern - provides a simplified interface to the complex Plotly figure structure.

```python
class FigureStore:
    """
    Central store for managing the Plotly figure and its elements.
    
    Responsibilities:
    - Create and manage the main figure
    - Add/remove/update traces (data series)
    - Add/remove/update annotations (text labels)
    - Add/remove/update shapes (rectangles, circles, lines)
    - Add/remove/update images (background images)
    - Maintain consistency between visual elements and data
    """
    
    def __init__(self):
        self._figure = go.Figure()
        self._trace_counter = 0
        self._annotation_counter = 0
        self._shape_counter = 0
        self._image_counter = 0
```

**Key Methods:**

| Method | Purpose | Complexity |
|--------|---------|------------|
| `get_figure()` | Return current figure | O(1) |
| `set_figure(fig)` | Replace entire figure | O(1) |
| `add_trace(trace_type, dataset)` | Add new data series | O(n) |
| `update_trace(index, properties)` | Modify trace properties | O(1) |
| `remove_trace(index)` | Delete a trace | O(n) |
| `add_annotation(text, position)` | Add text annotation | O(1) |
| `update_annotation(index, props)` | Modify annotation | O(1) |
| `remove_annotation(index)` | Delete annotation | O(n) |
| `add_shape(shape_type, coords)` | Add shape | O(1) |
| `remove_shape(index)` | Delete shape | O(n) |
| `add_image(source, position)` | Add background image | O(1) |
| `remove_image(index)` | Delete image | O(n) |

**State Diagram:**

```
                    ┌─────────────┐
                    │    EMPTY    │
                    │   Figure    │
                    └──────┬──────┘
                           │ add_trace()
                           ▼
                    ┌─────────────┐
          ┌────────│   BASIC     │────────┐
          │        │   Figure    │        │
          │        └──────┬──────┘        │
          │               │               │
add_annotation()    update_*()      add_shape()
          │               │               │
          ▼               ▼               ▼
    ┌───────────┐  ┌───────────┐  ┌───────────┐
    │ ANNOTATED │  │  UPDATED  │  │  WITH     │
    │  Figure   │  │  Figure   │  │  SHAPES   │
    └───────────┘  └───────────┘  └───────────┘
          │               │               │
          └───────────────┼───────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │   COMPLEX   │
                   │   Figure    │
                   └─────────────┘
```

#### 3.2.2 HistoryStack Class

**Purpose:** Implement undo/redo functionality using the Memento design pattern.

**Design Pattern:** Memento Pattern - captures and stores the internal state of an object so it can be restored later.

```python
class HistoryStack:
    """
    Manages undo/redo history using a stack-based approach.
    
    Implementation Details:
    - Maintains two stacks: undo_stack and redo_stack
    - Each entry is a deep copy of the entire figure state
    - Maximum history depth configurable (default: 50)
    - Clear redo stack on new action (standard behavior)
    """
    
    def __init__(self, max_size: int = 50):
        self._undo_stack: List[Dict] = []
        self._redo_stack: List[Dict] = []
        self._max_size = max_size
```

**Operation Flow:**

```
User Action                    Stack State
───────────                    ───────────
Initial                        Undo: []          Redo: []

Create figure                  Undo: [S0]        Redo: []

Add trace                      Undo: [S0, S1]    Redo: []

Change color                   Undo: [S0,S1,S2]  Redo: []

Click UNDO                     Undo: [S0, S1]    Redo: [S2]
  → Restore S1

Click UNDO                     Undo: [S0]        Redo: [S2, S1]
  → Restore S0

Click REDO                     Undo: [S0, S1]    Redo: [S2]
  → Restore S1

New action (add shape)         Undo: [S0,S1,S3]  Redo: []  ← Cleared!
```

**Memory Optimization:**

Since storing complete figure states can be memory-intensive, the implementation includes optimizations:

1. **Shallow Copy Where Safe:** Layout properties that rarely change
2. **Deep Copy for Data:** Trace data and modifications
3. **Pruning Strategy:** Remove oldest states when limit exceeded

#### 3.2.3 TraceDataset Class

**Purpose:** Encapsulate data parsing and validation logic for different input formats.

**Design Pattern:** Strategy Pattern - allows the algorithm for data parsing to vary based on input format.

```python
class TraceDataset:
    """
    Handles data input, parsing, and validation for plot traces.
    
    Supported Input Formats:
    - Direct Python lists/arrays
    - NumPy arrays
    - Pandas DataFrames (column selection)
    - CSV-style comma-separated values
    - JSON data import
    
    Validation Rules:
    - X and Y must have same length
    - Numeric values only for most trace types
    - String values allowed for categorical axes
    """
    
    @staticmethod
    def parse_input(text_input: str, input_type: str) -> Tuple[List, List, Optional[List]]:
        """
        Parse user input into x, y, (optional z) data arrays.
        
        Args:
            text_input: Raw text from input field
            input_type: One of 'xy_pairs', 'separate_xy', 'csv_columns', 'function'
            
        Returns:
            Tuple of (x_data, y_data, z_data or None)
        """
```

**Input Format Examples:**

| Format | Input Example | Parsed Result |
|--------|---------------|---------------|
| XY Pairs | `(1,2), (3,4), (5,6)` | x=[1,3,5], y=[2,4,6] |
| Separate XY | `x: 1,2,3 y: 4,5,6` | x=[1,2,3], y=[4,5,6] |
| CSV Columns | `1,4\n2,5\n3,6` | x=[1,2,3], y=[4,5,6] |
| Function | `x=linspace(0,10,50)` | Generated array |

#### 3.2.4 CodeGenerator Class

**Purpose:** Generate reproducible Python code from the current figure state.

**Design Pattern:** Template Method Pattern - defines the skeleton of code generation algorithm.

```python
class CodeGenerator:
    """
    Generates Python/Plotly code that recreates the current figure.
    
    Output Formats:
    - Plotly Express (high-level, concise)
    - Plotly Graph Objects (detailed, explicit)
    - Matplotlib (for users preferring static output)
    
    Code Quality:
    - Properly formatted (PEP 8 compliant)
    - Commented for readability
    - Executable without modification
    """
    
    def generate(self, figure: go.Figure, style: str = 'graph_objects') -> str:
        """Generate Python code for the figure."""
```

**Generated Code Example:**

```python
# Auto-generated by PyFigureEditor
# Date: 2024-12-01

import plotly.graph_objects as go

fig = go.Figure()

# Add scatter trace
fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[2, 4, 1, 5, 3],
    mode='lines+markers',
    name='Series 1',
    line=dict(color='#1f77b4', width=2),
    marker=dict(size=8, symbol='circle')
))

# Update layout
fig.update_layout(
    title=dict(text='My Figure'),
    xaxis=dict(title=dict(text='X Axis')),
    yaxis=dict(title=dict(text='Y Axis')),
    template='plotly_white'
)

fig.show()
```

### 3.3 Component Interaction Diagram

The following sequence diagram illustrates how components interact when a user changes a trace property:

```
User          UI Layer        Callback Layer    FigureStore     HistoryStack
 │                │                │                │                │
 │  Click Apply   │                │                │                │
 │───────────────>│                │                │                │
 │                │  n_clicks +1   │                │                │
 │                │───────────────>│                │                │
 │                │                │  get_figure()  │                │
 │                │                │───────────────>│                │
 │                │                │<───────────────│                │
 │                │                │                │                │
 │                │                │  push(current) │                │
 │                │                │────────────────────────────────>│
 │                │                │<────────────────────────────────│
 │                │                │                │                │
 │                │                │ update_trace() │                │
 │                │                │───────────────>│                │
 │                │                │<───────────────│                │
 │                │                │                │                │
 │                │  new figure    │                │                │
 │                │<───────────────│                │                │
 │  Updated graph │                │                │                │
 │<───────────────│                │                │                │
```

### 3.4 Layout Architecture

The visual layout follows a **responsive grid system** using Dash Bootstrap Components:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HEADER / TOOLBAR                              │
│  [New] [Load] [Save] [Undo] [Redo] [Export PNG] [Export Code]       │
│  Plot Type: [Dropdown ▼]   Template: [Dropdown ▼]                   │
└─────────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────┬────────────────────────────────┐
│                                    │                                │
│                                    │      PROPERTY INSPECTOR        │
│                                    │  ┌──────────────────────────┐  │
│                                    │  │ Element: [Dropdown ▼]   │  │
│        MAIN GRAPH AREA             │  ├──────────────────────────┤  │
│                                    │  │ ● Title: [__________]   │  │
│        (dcc.Graph)                 │  │ ● Color: [__________]   │  │
│                                    │  │ ● Size:  [__________]   │  │
│        Width: ~70%                 │  │ ● Style: [Dropdown ▼]   │  │
│                                    │  │ ● ...                    │  │
│                                    │  │                          │  │
│                                    │  │ [Apply] [Reset] [Delete]│  │
│                                    │  └──────────────────────────┘  │
│                                    │                                │
│                                    │      Width: ~30%              │
├────────────────────────────────────┴────────────────────────────────┤
│                        DATA INPUT PANEL                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Input Type: [Dropdown ▼]                                     │   │
│  │ ┌────────────────────────────────────────────────────────┐   │   │
│  │ │ Enter your data here...                                │   │   │
│  │ │ (1, 2), (3, 4), (5, 6)                                 │   │   │
│  │ └────────────────────────────────────────────────────────┘   │   │
│  │ [Add to Plot]                                                │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│                        DRAWING TOOLBAR                               │
│  Mode: [Select] [Zoom] [Pan] [Draw Line] [Draw Rect] [Draw Circle] │
│  [Add Text] [Add Arrow] [Add Image]                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.5 Data Flow Architecture

Understanding how data flows through the system is crucial for debugging and extending the application:

```
                           USER INPUT
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
         Data Entry      Property Edit    Drawing Action
              │                │                │
              ▼                ▼                ▼
    ┌─────────────────────────────────────────────────┐
    │              INPUT VALIDATION                    │
    │  • Data format checking                         │
    │  • Type conversion                              │
    │  • Range validation                             │
    └─────────────────────────────────────────────────┘
              │                │                │
              ▼                ▼                ▼
    ┌─────────────────────────────────────────────────┐
    │              STATE UPDATE                        │
    │  • FigureStore modification                     │
    │  • History push                                 │
    │  • Code regeneration                           │
    └─────────────────────────────────────────────────┘
                               │
                               ▼
    ┌─────────────────────────────────────────────────┐
    │              UI REFRESH                          │
    │  • Graph re-render                              │
    │  • Inspector update                             │
    │  • Element dropdown refresh                     │
    └─────────────────────────────────────────────────┘
                               │
                               ▼
                         VISUAL OUTPUT
```

### 3.6 Design Decisions and Trade-offs

Several key design decisions shaped the architecture:

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| Single-page application | Simpler state management | Limited URL routing |
| Figure state in callbacks | No external database needed | Limited scalability |
| Full figure copy for undo | Simple implementation | Higher memory usage |
| Client-side validation | Faster feedback | Duplicate validation logic |
| No user authentication | Simplifies deployment | Single-user sessions |
| JSON-based sessions | Human-readable, portable | Larger file sizes |

### 3.7 Extensibility Points

The architecture includes several extension points for future development:

1. **New Trace Types:** Add to `TRACE_TYPE_OPTIONS` dictionary and `add_trace()` method
2. **New Properties:** Extend `EDITABLE_PROPERTIES` configuration
3. **Export Formats:** Add new generators in `CodeGenerator` class
4. **Theme Support:** Extend `TEMPLATE_OPTIONS` with custom themes
5. **Plugin System:** Callbacks can be added dynamically

---

## 4. Core Implementation Details

This chapter provides an in-depth examination of the actual implementation, with extensive code analysis and explanations suitable for understanding every aspect of the system.

### 4.1 Application Initialization and Configuration

The application begins with essential imports and configuration:

```python
# =============================================================================
# IMPORTS
# =============================================================================
import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL, MATCH
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import json
import base64
import io
from datetime import datetime
```

**Import Analysis:**

| Import | Purpose | Critical Functions |
|--------|---------|-------------------|
| `dash` | Core framework | App creation, callbacks |
| `dcc` | Dash Core Components | Graph, Dropdown, Input, Store |
| `html` | HTML components | Div, Button, Span, H1-H6 |
| `dbc` | Bootstrap Components | Card, Row, Col, Modal, Tabs |
| `go` | Graph Objects | Figure, Scatter, Bar, etc. |
| `px` | Plotly Express | Quick chart creation |
| `np` | NumPy | Numerical operations |
| `pd` | Pandas | DataFrame operations |
| `json` | JSON handling | Session save/load |
| `base64` | Encoding | Image handling |
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

### 4.2 Layout Implementation

The application layout is built using Dash components organized in a hierarchical structure:

#### 4.2.1 Main Layout Structure

```python
app.layout = dbc.Container([
    # Hidden stores for state management
    dcc.Store(id='store-figure', data=None),
    dcc.Store(id='store-history', data={'undo': [], 'redo': []}),
    dcc.Store(id='store-datasets', data={}),
    
    # Header Section
    create_header(),
    
    # Main Content Area
    dbc.Row([
        # Left Panel: Graph Display
        dbc.Col([
            create_graph_panel()
        ], width=8, className='pe-2'),
        
        # Right Panel: Property Inspector
        dbc.Col([
            create_property_inspector()
        ], width=4, className='ps-2'),
    ], className='mb-3'),
    
    # Bottom Panel: Data Input
    create_data_input_panel(),
    
    # Drawing Toolbar
    create_drawing_toolbar(),
    
    # Modal Dialogs
    create_annotation_modal(),
    create_image_modal(),
    create_code_modal(),
    create_save_load_modal(),
    
], fluid=True, className='py-3')
```

#### 4.2.2 Header Component Implementation

```python
def create_header():
    """Create the application header with toolbar buttons."""
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                # Logo and Title
                dbc.Col([
                    html.H4([
                        html.I(className='fas fa-chart-line me-2'),
                        'PyFigureEditor'
                    ], className='mb-0 text-primary')
                ], width='auto'),
                
                # Main Toolbar Buttons
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button([
                            html.I(className='fas fa-file me-1'),
                            'New'
                        ], id='btn-new', color='outline-primary', size='sm'),
                        
                        dbc.Button([
                            html.I(className='fas fa-folder-open me-1'),
                            'Load'
                        ], id='btn-load', color='outline-primary', size='sm'),
                        
                        dbc.Button([
                            html.I(className='fas fa-save me-1'),
                            'Save'
                        ], id='btn-save', color='outline-primary', size='sm'),
                    ], className='me-3'),
                    
                    dbc.ButtonGroup([
                        dbc.Button([
                            html.I(className='fas fa-undo')
                        ], id='btn-undo', color='outline-secondary', size='sm',
                           title='Undo (Ctrl+Z)'),
                        
                        dbc.Button([
                            html.I(className='fas fa-redo')
                        ], id='btn-redo', color='outline-secondary', size='sm',
                           title='Redo (Ctrl+Y)'),
                    ], className='me-3'),
                    
                    dbc.ButtonGroup([
                        dbc.Button([
                            html.I(className='fas fa-image me-1'),
                            'PNG'
                        ], id='btn-export-png', color='outline-success', size='sm'),
                        
                        dbc.Button([
                            html.I(className='fas fa-code me-1'),
                            'Code'
                        ], id='btn-export-code', color='outline-success', size='sm'),
                    ]),
                ], width='auto'),
                
                # Plot Type and Template Selectors
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label('Plot Type:', className='mb-0 me-2'),
                            dcc.Dropdown(
                                id='dd-plot-type',
                                options=[{'label': v['label'], 'value': k} 
                                        for k, v in TRACE_TYPE_OPTIONS.items()],
                                value='scatter',
                                clearable=False,
                                style={'width': '150px'}
                            )
                        ], width='auto', className='d-flex align-items-center'),
                        
                        dbc.Col([
                            dbc.Label('Template:', className='mb-0 me-2'),
                            dcc.Dropdown(
                                id='dd-template',
                                options=[{'label': t, 'value': t} for t in TEMPLATE_OPTIONS],
                                value='plotly_white',
                                clearable=False,
                                style={'width': '150px'}
                            )
                        ], width='auto', className='d-flex align-items-center'),
                    ])
                ], className='ms-auto'),
            ], align='center'),
        ])
    ], className='mb-3')
```

#### 4.2.3 Graph Panel Implementation

```python
def create_graph_panel():
    """Create the main graph display panel."""
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className='fas fa-chart-area me-2'),
                'Figure Preview'
            ], className='mb-0')
        ]),
        dbc.CardBody([
            dcc.Graph(
                id='main-graph',
                figure=go.Figure(layout={
                    'template': 'plotly_white',
                    'title': {'text': 'New Figure'},
                    'xaxis': {'title': {'text': 'X Axis'}},
                    'yaxis': {'title': {'text': 'Y Axis'}},
                }),
                config={
                    'editable': True,              # Enable built-in editing
                    'modeBarButtonsToAdd': [
                        'drawline', 'drawopenpath', 'drawclosedpath',
                        'drawcircle', 'drawrect', 'eraseshape'
                    ],
                    'displaylogo': False,          # Hide Plotly logo
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': 'pyfigureeditor_export',
                        'height': 800,
                        'width': 1200,
                        'scale': 2                 # High resolution
                    }
                },
                style={'height': '500px'}
            )
        ], className='p-2')
    ])
```

#### 4.2.4 Property Inspector Implementation

```python
def create_property_inspector():
    """Create the property inspector panel."""
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className='fas fa-sliders-h me-2'),
                'Property Inspector'
            ], className='mb-0')
        ]),
        dbc.CardBody([
            # Element Selection Dropdown
            dbc.Label('Select Element:', className='fw-bold'),
            dcc.Dropdown(
                id='dd-element-select',
                placeholder='Select an element to edit...',
                className='mb-3'
            ),
            
            html.Hr(),
            
            # Dynamic Property Controls Container
            html.Div(id='inspector-controls', children=[
                html.P('Select an element from the dropdown above to edit its properties.',
                       className='text-muted')
            ]),
            
            html.Hr(),
            
            # Action Buttons
            dbc.Row([
                dbc.Col([
                    dbc.Button([
                        html.I(className='fas fa-check me-1'),
                        'Apply'
                    ], id='btn-apply-props', color='primary', className='w-100')
                ], width=4),
                dbc.Col([
                    dbc.Button([
                        html.I(className='fas fa-sync me-1'),
                        'Reset'
                    ], id='btn-reset-props', color='secondary', className='w-100')
                ], width=4),
                dbc.Col([
                    dbc.Button([
                        html.I(className='fas fa-trash me-1'),
                        'Delete'
                    ], id='btn-delete-element', color='danger', className='w-100')
                ], width=4),
            ])
        ], style={'maxHeight': '600px', 'overflowY': 'auto'})
    ])
```

### 4.3 Callback Implementation

Callbacks are the heart of the application's interactivity. This section analyzes key callbacks in detail.

#### 4.3.1 Add Trace Callback

This callback handles adding new data traces to the figure:

```python
@app.callback(
    Output('main-graph', 'figure', allow_duplicate=True),
    Output('store-history', 'data', allow_duplicate=True),
    Input('btn-add-trace', 'n_clicks'),
    State('main-graph', 'figure'),
    State('store-history', 'data'),
    State('dd-plot-type', 'value'),
    State('dd-input-type', 'value'),
    State('textarea-data', 'value'),
    prevent_initial_call=True
)
def add_trace(n_clicks, current_figure, history, plot_type, input_type, data_text):
    """
    Add a new trace to the figure based on user input.
    
    Process Flow:
    1. Validate inputs (prevent if empty/invalid)
    2. Parse data text into x, y arrays
    3. Create appropriate trace object
    4. Push current state to history
    5. Add trace to figure
    6. Return updated figure and history
    """
    # Guard clause: prevent if no click or empty data
    if not n_clicks or not data_text:
        raise PreventUpdate
    
    # Step 1: Parse the input data
    try:
        x_data, y_data, z_data = parse_data_input(data_text, input_type)
    except ValueError as e:
        # In production: show error toast notification
        raise PreventUpdate
    
    # Step 2: Create figure object from current state
    fig = go.Figure(current_figure)
    
    # Step 3: Push current state to undo history
    history = push_to_history(history, current_figure)
    
    # Step 4: Determine trace constructor and create trace
    trace_config = TRACE_TYPE_OPTIONS.get(plot_type, TRACE_TYPE_OPTIONS['scatter'])
    constructor = trace_config['constructor']
    
    # Build trace arguments
    trace_args = {
        'x': x_data,
        'y': y_data,
        'name': f'Trace {len(fig.data) + 1}'
    }
    
    # Add mode for scatter-based traces
    if constructor == go.Scatter:
        trace_args['mode'] = trace_config.get('mode', 'lines+markers')
        if 'fill' in trace_config:
            trace_args['fill'] = trace_config['fill']
    
    # Handle 3D traces
    if z_data is not None and constructor in [go.Scatter3d, go.Surface, go.Mesh3d]:
        trace_args['z'] = z_data
    
    # Step 5: Add the trace
    fig.add_trace(constructor(**trace_args))
    
    # Step 6: Return updated figure and history
    return fig.to_dict(), history
```

**Callback Analysis:**

| Aspect | Implementation Detail |
|--------|----------------------|
| **Trigger** | `btn-add-trace.n_clicks` |
| **Outputs** | Figure (duplicate), History (duplicate) |
| **States** | 5 state values for context |
| **Error Handling** | PreventUpdate on invalid input |
| **History** | Push before modification |

#### 4.3.2 Element Selection Update Callback

This callback dynamically generates property controls based on the selected element:

```python
@app.callback(
    Output('inspector-controls', 'children'),
    Input('dd-element-select', 'value'),
    State('main-graph', 'figure')
)
def update_inspector_controls(selected_element, figure):
    """
    Generate property editing controls for the selected element.
    
    Element Types Handled:
    - 'trace-0', 'trace-1', etc. → Trace properties
    - 'layout' → Layout properties
    - 'annotation-0', etc. → Annotation properties
    - 'shape-0', etc. → Shape properties
    - 'image-0', etc. → Image properties
    """
    if not selected_element or not figure:
        return html.P('Select an element to edit its properties.', 
                     className='text-muted')
    
    # Parse element type and index
    element_parts = selected_element.split('-')
    element_type = element_parts[0]
    element_index = int(element_parts[1]) if len(element_parts) > 1 else None
    
    # Get current property values
    if element_type == 'trace':
        current_values = figure['data'][element_index]
        property_config = EDITABLE_PROPERTIES['trace']
    elif element_type == 'layout':
        current_values = figure['layout']
        property_config = EDITABLE_PROPERTIES['layout']
    elif element_type == 'annotation':
        current_values = figure['layout'].get('annotations', [])[element_index]
        property_config = EDITABLE_PROPERTIES['annotation']
    elif element_type == 'shape':
        current_values = figure['layout'].get('shapes', [])[element_index]
        property_config = EDITABLE_PROPERTIES['shape']
    elif element_type == 'image':
        current_values = figure['layout'].get('images', [])[element_index]
        property_config = EDITABLE_PROPERTIES['image']
    else:
        return html.P('Unknown element type.', className='text-danger')
    
    # Generate controls for each property
    controls = []
    for prop_key, prop_config in property_config.items():
        # Get nested property value (e.g., 'line.color' → figure.data[0].line.color)
        current_value = get_nested_value(current_values, prop_key)
        
        # Create appropriate input component
        control = create_property_control(
            prop_key=prop_key,
            prop_config=prop_config,
            current_value=current_value,
            element_id=selected_element
        )
        controls.append(control)
    
    return html.Div(controls)


def create_property_control(prop_key, prop_config, current_value, element_id):
    """Create a single property editing control."""
    control_type = prop_config['type']
    label = prop_config['label']
    
    # Create unique ID for this control
    control_id = {'type': 'prop-control', 'element': element_id, 'property': prop_key}
    
    if control_type == 'text':
        input_component = dbc.Input(
            id=control_id,
            type='text',
            value=current_value or '',
            placeholder=f'Enter {label}...'
        )
    
    elif control_type == 'number':
        input_component = dbc.Input(
            id=control_id,
            type='number',
            value=current_value,
            min=prop_config.get('min'),
            max=prop_config.get('max'),
            step=prop_config.get('step', 1)
        )
    
    elif control_type == 'color':
        input_component = dbc.Input(
            id=control_id,
            type='color',
            value=current_value or '#000000',
            style={'width': '100%', 'height': '38px'}
        )
    
    elif control_type == 'dropdown':
        input_component = dcc.Dropdown(
            id=control_id,
            options=[{'label': str(o), 'value': o} for o in prop_config['options']],
            value=current_value,
            clearable=False
        )
    
    elif control_type == 'checkbox':
        input_component = dbc.Checkbox(
            id=control_id,
            value=bool(current_value),
            label=''
        )
    
    else:
        input_component = html.Span('Unsupported type', className='text-danger')
    
    return dbc.Row([
        dbc.Col([
            dbc.Label(label, className='mb-0')
        ], width=5),
        dbc.Col([
            input_component
        ], width=7)
    ], className='mb-2 align-items-center')
```

#### 4.3.3 Apply Properties Callback

This callback applies edited properties back to the figure:

```python
@app.callback(
    Output('main-graph', 'figure', allow_duplicate=True),
    Output('store-history', 'data', allow_duplicate=True),
    Input('btn-apply-props', 'n_clicks'),
    State('dd-element-select', 'value'),
    State('main-graph', 'figure'),
    State('store-history', 'data'),
    State({'type': 'prop-control', 'element': ALL, 'property': ALL}, 'value'),
    State({'type': 'prop-control', 'element': ALL, 'property': ALL}, 'id'),
    prevent_initial_call=True
)
def apply_properties(n_clicks, selected_element, figure, history, 
                     control_values, control_ids):
    """
    Apply property changes from inspector to the figure.
    
    Pattern Matching Callback:
    - Uses ALL pattern to capture all property controls
    - control_ids contains metadata about each control
    - control_values contains current values
    """
    if not n_clicks or not selected_element:
        raise PreventUpdate
    
    # Safety check for callback trigger
    ctx = callback_context
    if not ctx.triggered_id:
        raise PreventUpdate
    
    # Create figure object
    fig = go.Figure(figure)
    
    # Push to history before changes
    history = push_to_history(history, figure)
    
    # Parse element info
    element_parts = selected_element.split('-')
    element_type = element_parts[0]
    element_index = int(element_parts[1]) if len(element_parts) > 1 else None
    
    # Build property updates dictionary
    updates = {}
    for control_id, value in zip(control_ids, control_values):
        if control_id['element'] == selected_element:
            prop_key = control_id['property']
            updates[prop_key] = value
    
    # Apply updates based on element type
    if element_type == 'trace':
        apply_trace_updates(fig, element_index, updates)
    elif element_type == 'layout':
        apply_layout_updates(fig, updates)
    elif element_type == 'annotation':
        apply_annotation_updates(fig, element_index, updates)
    elif element_type == 'shape':
        apply_shape_updates(fig, element_index, updates)
    
    return fig.to_dict(), history


def apply_trace_updates(fig, trace_index, updates):
    """Apply property updates to a specific trace."""
    for prop_key, value in updates.items():
        # Handle nested properties (e.g., 'line.color')
        keys = prop_key.split('.')
        
        if len(keys) == 1:
            # Simple property
            fig.data[trace_index][keys[0]] = value
        elif len(keys) == 2:
            # Nested property
            parent, child = keys
            if parent not in fig.data[trace_index]:
                fig.data[trace_index][parent] = {}
            fig.data[trace_index][parent][child] = value
```

#### 4.3.4 Delete Element Callback

This callback handles deletion of traces, annotations, shapes, and images:

```python
@app.callback(
    Output('main-graph', 'figure', allow_duplicate=True),
    Output('store-history', 'data', allow_duplicate=True),
    Output('dd-element-select', 'value', allow_duplicate=True),
    Input('btn-delete-element', 'n_clicks'),
    State('dd-element-select', 'value'),
    State('main-graph', 'figure'),
    State('store-history', 'data'),
    prevent_initial_call=True
)
def delete_element(n_clicks, selected_element, figure, history):
    """
    Delete the selected element from the figure.
    
    Handles:
    - Traces: Remove from fig.data
    - Annotations: Remove from fig.layout.annotations
    - Shapes: Remove from fig.layout.shapes
    - Images: Remove from fig.layout.images
    """
    # Safety check
    ctx = callback_context
    if not ctx.triggered_id or not n_clicks or not selected_element:
        raise PreventUpdate
    
    # Parse element info
    element_parts = selected_element.split('-')
    element_type = element_parts[0]
    
    if len(element_parts) < 2:
        raise PreventUpdate  # Can't delete 'layout'
    
    element_index = int(element_parts[1])
    
    # Create figure and push history
    fig = go.Figure(figure)
    history = push_to_history(history, figure)
    
    # Perform deletion based on type
    if element_type == 'trace':
        # Remove trace from data array
        data_list = list(fig.data)
        if 0 <= element_index < len(data_list):
            del data_list[element_index]
            fig.data = data_list
    
    elif element_type == 'annotation':
        annotations = list(fig.layout.annotations or [])
        if 0 <= element_index < len(annotations):
            del annotations[element_index]
            fig.update_layout(annotations=annotations)
    
    elif element_type == 'shape':
        shapes = list(fig.layout.shapes or [])
        if 0 <= element_index < len(shapes):
            del shapes[element_index]
            fig.update_layout(shapes=shapes)
    
    elif element_type == 'image':
        images = list(fig.layout.images or [])
        if 0 <= element_index < len(images):
            del images[element_index]
            fig.update_layout(images=images)
    
    # Clear selection and return
    return fig.to_dict(), history, None
```

### 4.4 Helper Functions

Several utility functions support the callback logic:

#### 4.4.1 Data Parsing Functions

```python
def parse_data_input(text, input_type):
    """
    Parse user text input into numerical arrays.
    
    Args:
        text: Raw text input from user
        input_type: One of 'xy_pairs', 'separate_xy', 'csv'
        
    Returns:
        Tuple of (x_list, y_list, z_list_or_None)
    """
    if input_type == 'xy_pairs':
        # Format: (1,2), (3,4), (5,6)
        return parse_xy_pairs(text)
    elif input_type == 'separate_xy':
        # Format: x: 1,2,3 y: 4,5,6
        return parse_separate_xy(text)
    elif input_type == 'csv':
        # Format: CSV with columns
        return parse_csv(text)
    elif input_type == 'function':
        # Format: Mathematical expression
        return parse_function(text)
    else:
        raise ValueError(f'Unknown input type: {input_type}')


def parse_xy_pairs(text):
    """Parse (x,y) pair format."""
    import re
    
    # Match patterns like (1,2) or (1.5, -2.3)
    pattern = r'\(([^,]+),([^)]+)\)'
    matches = re.findall(pattern, text)
    
    if not matches:
        raise ValueError('No valid (x,y) pairs found')
    
    x_data = [float(m[0].strip()) for m in matches]
    y_data = [float(m[1].strip()) for m in matches]
    
    return x_data, y_data, None


def parse_separate_xy(text):
    """Parse separate x: and y: format."""
    import re
    
    # Find x: ... and y: ... sections
    x_match = re.search(r'x\s*:\s*([^\n]+)', text, re.IGNORECASE)
    y_match = re.search(r'y\s*:\s*([^\n]+)', text, re.IGNORECASE)
    
    if not x_match or not y_match:
        raise ValueError('Could not find x: and y: sections')
    
    x_data = [float(v.strip()) for v in x_match.group(1).split(',')]
    y_data = [float(v.strip()) for v in y_match.group(1).split(',')]
    
    if len(x_data) != len(y_data):
        raise ValueError(f'X has {len(x_data)} values but Y has {len(y_data)}')
    
    return x_data, y_data, None
```

#### 4.4.2 History Management Functions

```python
def push_to_history(history, figure_state, max_size=50):
    """
    Push current state to undo history.
    
    Args:
        history: Current history dict with 'undo' and 'redo' keys
        figure_state: Current figure dictionary to save
        max_size: Maximum history depth
        
    Returns:
        Updated history dictionary
    """
    import copy
    
    # Deep copy to prevent reference issues
    state_copy = copy.deepcopy(figure_state)
    
    # Get current stacks
    undo_stack = history.get('undo', [])
    
    # Add to undo stack
    undo_stack.append(state_copy)
    
    # Prune if exceeds max size
    if len(undo_stack) > max_size:
        undo_stack = undo_stack[-max_size:]
    
    # Clear redo stack (standard behavior for new actions)
    return {
        'undo': undo_stack,
        'redo': []
    }


def undo_action(history):
    """
    Perform undo operation.
    
    Returns:
        Tuple of (restored_figure, updated_history)
    """
    undo_stack = history.get('undo', [])
    redo_stack = history.get('redo', [])
    
    if not undo_stack:
        return None, history  # Nothing to undo
    
    # Pop from undo, push to redo
    restored = undo_stack.pop()
    redo_stack.append(restored)
    
    # Get previous state (now at top of undo stack)
    previous = undo_stack[-1] if undo_stack else None
    
    return previous, {'undo': undo_stack, 'redo': redo_stack}
```

#### 4.4.3 Nested Property Access Functions

```python
def get_nested_value(obj, key_path, default=None):
    """
    Get value from nested dictionary using dot notation.
    
    Example:
        get_nested_value({'line': {'color': 'red'}}, 'line.color')
        → 'red'
    """
    keys = key_path.split('.')
    current = obj
    
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
        
        if current is None:
            return default
    
    return current


def set_nested_value(obj, key_path, value):
    """
    Set value in nested dictionary using dot notation.
    
    Example:
        set_nested_value({}, 'line.color', 'red')
        → {'line': {'color': 'red'}}
    """
    keys = key_path.split('.')
    current = obj
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value
    return obj
```

---

## 5. Complete Feature Documentation

This chapter provides comprehensive documentation of all features available in PyFigureEditor, organized by functional category. Each feature includes usage instructions, supported options, and practical examples.

### 5.1 Plot Types and Chart Creation

PyFigureEditor supports **26+ plot types**, covering virtually every common visualization need.

#### 5.1.1 Basic Charts

| Plot Type | Description | Best Use Cases |
|-----------|-------------|----------------|
| **Scatter** | Points plotted at x,y coordinates | Correlation analysis, point distributions |
| **Line** | Connected points showing trends | Time series, continuous data |
| **Bar** | Vertical bars for categorical data | Comparisons, rankings |
| **Horizontal Bar** | Horizontal bars | Long category names, rankings |
| **Area** | Filled line chart | Cumulative totals, proportions over time |

**Creating a Scatter Plot:**

```
Step 1: Select "Scatter" from Plot Type dropdown
Step 2: Enter data in the input area:
        Format: (1, 2), (3, 4), (5, 6), (7, 8)
Step 3: Click "Add to Plot"
Step 4: Customize using Property Inspector:
        - Marker Size: 10
        - Marker Color: #FF6B6B
        - Marker Symbol: circle
```

**Creating a Line Chart:**

```
Step 1: Select "Line" from Plot Type dropdown
Step 2: Enter data:
        x: 0, 1, 2, 3, 4, 5
        y: 0, 1, 4, 9, 16, 25
Step 3: Click "Add to Plot"
Step 4: Customize:
        - Line Width: 3
        - Line Color: #4ECDC4
        - Line Style: solid
```

#### 5.1.2 Statistical Charts

| Plot Type | Description | Data Requirements |
|-----------|-------------|-------------------|
| **Histogram** | Distribution of single variable | Single array of values |
| **Box Plot** | Statistical summary (quartiles) | One or more data arrays |
| **Violin Plot** | Distribution shape + statistics | One or more data arrays |
| **Heatmap** | 2D color-coded matrix | 2D array (matrix) |

**Creating a Box Plot:**

```
Step 1: Select "Box" from Plot Type dropdown
Step 2: Enter multiple data series:
        Series 1: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        Series 2: 2, 4, 6, 8, 10, 12, 14, 16
Step 3: Click "Add to Plot" for each series
Step 4: Customize:
        - Box Fill Color
        - Whisker Width
        - Show/Hide Points
```

#### 5.1.3 3D Charts

| Plot Type | Description | Data Requirements |
|-----------|-------------|-------------------|
| **3D Scatter** | Points in 3D space | x, y, z arrays |
| **3D Surface** | Continuous surface | 2D z matrix |
| **3D Mesh** | Triangulated surface | x, y, z + i, j, k indices |
| **3D Line** | Line in 3D space | x, y, z arrays |

**Creating a 3D Surface:**

```python
# Data format for 3D Surface:
# z values as 2D array, or generated from function

# Example: z = sin(x) * cos(y)
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

# Input the Z matrix in the data input area
```

#### 5.1.4 Specialized Charts

| Plot Type | Description | Use Cases |
|-----------|-------------|-----------|
| **Pie** | Proportional segments | Market share, budgets |
| **Donut** | Pie with center hole | Same as pie, modern look |
| **Funnel** | Progressive stages | Sales pipeline, conversion |
| **Waterfall** | Incremental changes | Financial analysis |
| **Candlestick** | OHLC financial data | Stock price analysis |
| **OHLC** | Open-High-Low-Close | Financial data |
| **Treemap** | Hierarchical rectangles | File sizes, organizational data |
| **Sunburst** | Hierarchical radial | Organizational hierarchy |

#### 5.1.5 Geographic Charts

| Plot Type | Description | Data Requirements |
|-----------|-------------|-------------------|
| **Scatter Geo** | Points on world map | lat, lon coordinates |
| **Choropleth** | Colored regions | Region codes + values |
| **Scatter Mapbox** | Points on Mapbox map | lat, lon + Mapbox token |

### 5.2 Property Editing System

The Property Inspector provides fine-grained control over every visual aspect of the figure.

#### 5.2.1 Trace Properties (35+ Properties)

**Line Properties:**

| Property | Type | Values | Description |
|----------|------|--------|-------------|
| `line.color` | Color | Hex/RGB/Named | Line color |
| `line.width` | Number | 0-20 | Line thickness in pixels |
| `line.dash` | Dropdown | solid, dot, dash, longdash, dashdot | Line style pattern |
| `line.shape` | Dropdown | linear, spline, hv, vh, hvh, vhv | Interpolation method |

**Marker Properties:**

| Property | Type | Values | Description |
|----------|------|--------|-------------|
| `marker.color` | Color | Hex/RGB/Named | Marker fill color |
| `marker.size` | Number | 1-50 | Marker diameter in pixels |
| `marker.symbol` | Dropdown | 40+ symbols | Marker shape |
| `marker.opacity` | Number | 0-1 | Marker transparency |
| `marker.line.color` | Color | Hex/RGB/Named | Marker border color |
| `marker.line.width` | Number | 0-10 | Marker border width |

**Available Marker Symbols:**

```
Basic:     circle, square, diamond, cross, x
Triangles: triangle-up, triangle-down, triangle-left, triangle-right
Extended:  star, hexagon, pentagon, octagon, hexagram
Open:      circle-open, square-open, diamond-open
Dot:       circle-dot, square-dot, diamond-dot
Combined:  circle-cross, circle-x, square-cross, square-x
```

**Text Properties (for traces with text):**

| Property | Type | Values | Description |
|----------|------|--------|-------------|
| `textposition` | Dropdown | top, bottom, left, right, etc. | Text placement |
| `textfont.size` | Number | 8-72 | Font size |
| `textfont.color` | Color | Hex/RGB/Named | Text color |
| `textfont.family` | Text | Font name | Font family |

#### 5.2.2 Layout Properties

**Title and Axes:**

| Property | Type | Description |
|----------|------|-------------|
| `title.text` | Text | Figure title |
| `title.font.size` | Number | Title font size |
| `title.font.color` | Color | Title color |
| `xaxis.title.text` | Text | X axis label |
| `yaxis.title.text` | Text | Y axis label |
| `xaxis.range` | Array | [min, max] for X axis |
| `yaxis.range` | Array | [min, max] for Y axis |

**Legend:**

| Property | Type | Description |
|----------|------|-------------|
| `showlegend` | Boolean | Show/hide legend |
| `legend.x` | Number | Legend X position (0-1) |
| `legend.y` | Number | Legend Y position (0-1) |
| `legend.orientation` | Dropdown | 'h' or 'v' |
| `legend.bgcolor` | Color | Legend background |

**Grid and Axes:**

| Property | Type | Description |
|----------|------|-------------|
| `xaxis.showgrid` | Boolean | Show X grid lines |
| `yaxis.showgrid` | Boolean | Show Y grid lines |
| `xaxis.gridcolor` | Color | X grid color |
| `yaxis.gridcolor` | Color | Y grid color |
| `xaxis.zeroline` | Boolean | Show X zero line |
| `yaxis.zeroline` | Boolean | Show Y zero line |

#### 5.2.3 Annotation Properties

| Property | Type | Description |
|----------|------|-------------|
| `text` | Text | Annotation text (supports HTML) |
| `x` | Number | X position |
| `y` | Number | Y position |
| `xref` | Dropdown | 'x', 'paper' - coordinate reference |
| `yref` | Dropdown | 'y', 'paper' - coordinate reference |
| `showarrow` | Boolean | Display arrow |
| `arrowhead` | Number | Arrow head style (0-8) |
| `arrowsize` | Number | Arrow head size multiplier |
| `arrowwidth` | Number | Arrow line width |
| `arrowcolor` | Color | Arrow color |
| `ax` | Number | Arrow X offset |
| `ay` | Number | Arrow Y offset |
| `font.size` | Number | Text font size |
| `font.color` | Color | Text color |
| `bgcolor` | Color | Background color |
| `bordercolor` | Color | Border color |
| `borderwidth` | Number | Border width |

#### 5.2.4 Shape Properties

| Property | Type | Description |
|----------|------|-------------|
| `type` | Dropdown | rect, circle, line, path |
| `x0`, `y0` | Number | Start coordinates |
| `x1`, `y1` | Number | End coordinates |
| `line.color` | Color | Shape outline color |
| `line.width` | Number | Outline width |
| `line.dash` | Dropdown | Line style |
| `fillcolor` | Color | Fill color |
| `opacity` | Number | Transparency (0-1) |
| `layer` | Dropdown | 'above', 'below' traces |

### 5.3 Drawing and Annotation Tools

PyFigureEditor provides MATLAB-style drawing tools for adding visual elements.

#### 5.3.1 Drawing Mode Toolbar

| Button | Mode | Function |
|--------|------|----------|
| **Select** | `select` | Default selection mode |
| **Zoom** | `zoom` | Click-drag to zoom region |
| **Pan** | `pan` | Click-drag to move view |
| **Draw Line** | `drawline` | Draw straight lines |
| **Draw Rect** | `drawrect` | Draw rectangles |
| **Draw Circle** | `drawcircle` | Draw circles/ellipses |
| **Draw Path** | `drawopenpath` | Draw freeform paths |
| **Eraser** | `eraseshape` | Delete drawn shapes |

#### 5.3.2 Adding Text Annotations

**Method 1: Via Modal Dialog**

```
1. Click "Add Text" button in toolbar
2. Modal dialog opens with fields:
   - Text: Enter your annotation text
   - X Position: Click graph or enter value
   - Y Position: Click graph or enter value
   - Show Arrow: Checkbox
   - Font Size: Number input
   - Font Color: Color picker
3. Click "Add Annotation"
4. Annotation appears on graph
```

**Method 2: Direct on Graph (Editable Mode)**

```
1. Double-click on the graph at desired location
2. Text input cursor appears
3. Type your annotation
4. Click outside to confirm
```

#### 5.3.3 Adding Arrows

```
1. Click "Add Text" button
2. Enable "Show Arrow" checkbox
3. Set arrow properties:
   - Arrow Head Style (0-8)
   - Arrow Size
   - Arrow Color
4. The annotation will display with an arrow
   pointing from (ax, ay) offset to (x, y) position
```

#### 5.3.4 Adding Images

```
1. Click "Add Image" button
2. Upload image file (PNG, JPG, SVG supported)
3. Configure placement:
   - X Position, Y Position
   - Width, Height (in axis units or paper fraction)
   - Layer: Above or Below traces
   - Opacity
4. Click "Add Image"
5. Image appears as figure background/overlay
```

### 5.4 Session Management

PyFigureEditor allows saving and restoring complete editing sessions.

#### 5.4.1 Saving Sessions

**What Gets Saved:**

```json
{
  "version": "1.0",
  "timestamp": "2024-12-01T10:30:00",
  "figure": {
    "data": [...],
    "layout": {...}
  },
  "datasets": {
    "trace-0": {"x": [...], "y": [...]},
    "trace-1": {"x": [...], "y": [...]}
  },
  "history": {
    "undo": [...],
    "redo": [...]
  },
  "metadata": {
    "name": "My Analysis",
    "author": "User",
    "description": "Quarterly sales analysis"
  }
}
```

**Saving Process:**

```
1. Click "Save" button in toolbar
2. Save dialog opens
3. Enter session name (optional)
4. Click "Download Session"
5. JSON file downloads to your computer
   Filename: pyfigureeditor_session_YYYYMMDD_HHMMSS.json
```

#### 5.4.2 Loading Sessions

```
1. Click "Load" button in toolbar
2. Load dialog opens
3. Click "Choose File" or drag-and-drop
4. Select a previously saved .json session file
5. Click "Load Session"
6. Figure and all data are restored
```

#### 5.4.3 Session Compatibility

| Version | Compatible With | Notes |
|---------|-----------------|-------|
| 1.0 | Current | Full support |
| Future | Backward compatible | Planned versioning |

### 5.5 Code Export

Generate reproducible Python code from your figure.

#### 5.5.1 Export Formats

**Plotly Graph Objects (Default):**

```python
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[2, 4, 1, 5, 3],
    mode='lines+markers',
    name='Data Series 1',
    line=dict(color='#1f77b4', width=2, dash='solid'),
    marker=dict(size=8, symbol='circle', color='#1f77b4')
))

fig.update_layout(
    title=dict(text='My Figure', font=dict(size=20)),
    xaxis=dict(title=dict(text='X Axis')),
    yaxis=dict(title=dict(text='Y Axis')),
    template='plotly_white',
    showlegend=True
)

fig.show()
```

**Plotly Express (Concise):**

```python
import plotly.express as px
import pandas as pd

df = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [2, 4, 1, 5, 3],
    'series': ['Data'] * 5
})

fig = px.scatter(df, x='x', y='y', title='My Figure')
fig.update_traces(marker=dict(size=8, color='#1f77b4'))
fig.show()
```

#### 5.5.2 Export Process

```
1. Create and customize your figure
2. Click "Export Code" button
3. Code modal opens with generated Python code
4. Options:
   - Copy to Clipboard: Click copy button
   - Download: Click download button (.py file)
5. Code is ready to run in any Python environment
```

### 5.6 Undo/Redo System

Full undo/redo support for all operations.

#### 5.6.1 Supported Operations

| Operation | Undoable | Notes |
|-----------|----------|-------|
| Add trace | ✅ | Removes added trace |
| Delete trace | ✅ | Restores deleted trace |
| Property change | ✅ | Reverts property value |
| Add annotation | ✅ | Removes annotation |
| Add shape | ✅ | Removes shape |
| Template change | ✅ | Reverts template |
| Drawing | ✅ | Removes drawn shape |

#### 5.6.2 Usage

```
Undo: Click "Undo" button or press Ctrl+Z
Redo: Click "Redo" button or press Ctrl+Y

History Depth: Up to 50 operations
Memory: ~5-10KB per state (depends on figure complexity)
```

#### 5.6.3 Behavior Notes

- **New Action After Undo:** Clears redo stack (standard behavior)
- **Load Session:** Replaces history (session has its own history)
- **New Figure:** Clears all history

### 5.7 Template System

PyFigureEditor supports 10 built-in Plotly templates.

#### 5.7.1 Available Templates

| Template | Description | Best For |
|----------|-------------|----------|
| `plotly` | Default Plotly style | General use |
| `plotly_white` | White background | Publications, reports |
| `plotly_dark` | Dark theme | Presentations, dashboards |
| `ggplot2` | R ggplot2 style | Statistical graphics |
| `seaborn` | Seaborn style | Data science |
| `simple_white` | Minimal white | Clean publications |
| `presentation` | Large fonts | Slideshows |
| `xgridoff` | No vertical grid | Time series |
| `ygridoff` | No horizontal grid | Bar charts |
| `gridon` | Full grid | Technical plots |

#### 5.7.2 Applying Templates

```
1. Select template from "Template" dropdown in toolbar
2. Template applies immediately to current figure
3. All existing traces inherit new template colors
4. Can be changed at any time without data loss
```

---

## 6. Deployment Guide

This chapter provides comprehensive instructions for deploying PyFigureEditor in different environments, from local development to production servers.

### 6.1 Two Deployment Versions

PyFigureEditor exists in two forms optimized for different use cases:

| Version | File | Purpose | Best For |
|---------|------|---------|----------|
| **Jupyter Notebook** | `Final_Project_Implementation.ipynb` | Interactive development | Learning, experimentation, Google Colab |
| **Server Application** | `app.py` | Production deployment | PythonAnywhere, Heroku, AWS |

### 6.2 Local Development Setup

#### 6.2.1 Prerequisites

**System Requirements:**

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8+ | 3.10+ |
| RAM | 2 GB | 4 GB |
| Storage | 100 MB | 500 MB |
| Browser | Modern (Chrome, Firefox, Edge) | Chrome 90+ |

**Required Packages:**

```bash
# Core dependencies
pip install dash>=2.17.0
pip install plotly>=5.20.0
pip install dash-bootstrap-components>=1.5.0
pip install pandas>=2.0.0
pip install numpy>=1.24.0

# Or install all at once
pip install dash plotly dash-bootstrap-components pandas numpy
```

#### 6.2.2 Running Locally

**From Jupyter Notebook:**

```python
# In Final_Project_Implementation.ipynb
# Run all cells in order
# The app will start on http://127.0.0.1:8050

# To change port:
app.run(debug=True, port=8051)
```

**From Python Script:**

```bash
# Navigate to project directory
cd "Final Project"

# Run the application
python app.py

# Output:
# Dash is running on http://127.0.0.1:8050/
# * Running on http://127.0.0.1:8050
# * Debug mode: on
```

#### 6.2.3 Development Mode Features

When running with `debug=True`:

| Feature | Description |
|---------|-------------|
| **Hot Reload** | Automatic restart on code changes |
| **Error Display** | Detailed error messages in browser |
| **Component Inspector** | Dash DevTools for debugging |
| **Callback Graph** | Visualize callback dependencies |

### 6.3 PythonAnywhere Deployment

PythonAnywhere is the recommended platform for this project. The live deployment is available at:

**🌐 https://zye.pythonanywhere.com/**

#### 6.3.1 Account Setup

```
1. Go to https://www.pythonanywhere.com
2. Sign up for a free account (or paid for more resources)
3. Verify your email address
4. Log in to your dashboard
```

#### 6.3.2 File Upload

**Method 1: Web Interface**

```
1. Go to "Files" tab in PythonAnywhere
2. Navigate to /home/yourusername/
3. Create a new directory: mkdir myproject
4. Upload files:
   - app.py
   - requirements.txt (if you have one)
```

**Method 2: Git Clone**

```bash
# In PythonAnywhere Bash console:
cd ~
git clone https://github.com/yourusername/pyfigureeditor.git
cd pyfigureeditor
```

#### 6.3.3 Web App Configuration

```
1. Go to "Web" tab
2. Click "Add a new web app"
3. Select "Manual configuration"
4. Choose Python version (3.10 recommended)
5. Note the path to your web app configuration
```

#### 6.3.4 WSGI Configuration

Edit the WSGI configuration file (e.g., `/var/www/yourusername_pythonanywhere_com_wsgi.py`):

```python
# =============================================================================
# WSGI Configuration for PythonAnywhere
# =============================================================================

import sys
import os

# Add your project directory to the path
project_home = '/home/yourusername/myproject'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Set the working directory
os.chdir(project_home)

# Import your Dash app
from app import app

# Expose the Flask server for WSGI
application = app.server
```

#### 6.3.5 Virtual Environment (Recommended)

```bash
# In PythonAnywhere Bash console:

# Create virtual environment
mkvirtualenv --python=/usr/bin/python3.10 myenv

# Activate it
workon myenv

# Install dependencies
pip install dash plotly dash-bootstrap-components pandas numpy

# Verify installation
pip list
```

Then update Web app settings to use the virtual environment:

```
Virtualenv: /home/yourusername/.virtualenvs/myenv
```

#### 6.3.6 Reload and Test

```
1. Go to "Web" tab
2. Click green "Reload" button
3. Visit your-username.pythonanywhere.com
4. Your app should be live!
```

#### 6.3.7 Troubleshooting PythonAnywhere

| Issue | Solution |
|-------|----------|
| 502 Bad Gateway | Check WSGI file for syntax errors |
| Module not found | Install in correct virtualenv |
| App not updating | Click "Reload" button |
| Slow performance | Consider paid tier for more CPU |
| Static files not loading | Configure static file mappings |

**Checking Error Logs:**

```
1. Go to "Web" tab
2. Click "Error log" link
3. Check for Python tracebacks
4. Fix issues and reload
```

### 6.4 Google Colab Deployment

For users who want to run PyFigureEditor directly in Google Colab:

#### 6.4.1 Colab-Specific Setup

```python
# Cell 1: Install dependencies
!pip install dash plotly dash-bootstrap-components pandas numpy jupyter-dash

# Cell 2: Import and configure
from jupyter_dash import JupyterDash
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# Use JupyterDash instead of dash.Dash
app = JupyterDash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# ... rest of app code ...

# Cell N: Run the app
app.run_server(mode='external')  # or mode='inline' for embedded view
```

#### 6.4.2 Colab Run Modes

| Mode | Command | Description |
|------|---------|-------------|
| `external` | `app.run_server(mode='external')` | Opens in new tab |
| `inline` | `app.run_server(mode='inline')` | Embeds below cell |
| `jupyterlab` | `app.run_server(mode='jupyterlab')` | JupyterLab tab |

#### 6.4.3 Colab Limitations

- Session timeout after inactivity
- No persistent storage (use Google Drive mount)
- Public URL expires when runtime disconnects
- Limited CPU/RAM on free tier

### 6.5 Alternative Deployment Platforms

#### 6.5.1 Heroku Deployment

**Required Files:**

`Procfile`:
```
web: gunicorn app:server
```

`requirements.txt`:
```
dash>=2.17.0
plotly>=5.20.0
dash-bootstrap-components>=1.5.0
pandas>=2.0.0
numpy>=1.24.0
gunicorn>=21.0.0
```

`runtime.txt`:
```
python-3.10.12
```

**Deployment Commands:**

```bash
# Install Heroku CLI, then:
heroku login
heroku create pyfigureeditor-app
git push heroku main
heroku open
```

#### 6.5.2 AWS Elastic Beanstalk

**application.py** (rename app.py):

```python
# AWS EB looks for 'application' variable
from app import app
application = app.server
```

**Deployment:**

```bash
eb init -p python-3.10 pyfigureeditor
eb create pyfigureeditor-env
eb deploy
eb open
```

#### 6.5.3 Docker Deployment

**Dockerfile:**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8050

CMD ["python", "app.py"]
```

**docker-compose.yml:**

```yaml
version: '3.8'
services:
  pyfigureeditor:
    build: .
    ports:
      - "8050:8050"
    environment:
      - DASH_DEBUG=false
```

**Run:**

```bash
docker-compose up -d
# Access at http://localhost:8050
```

### 6.6 Production Considerations

#### 6.6.1 Security

| Concern | Recommendation |
|---------|----------------|
| Input validation | Validate all user inputs server-side |
| File uploads | Limit file size, validate file types |
| HTTPS | Always use HTTPS in production |
| Debug mode | Never use `debug=True` in production |

#### 6.6.2 Performance Optimization

```python
# Production settings
app.run(
    debug=False,              # Disable debug mode
    dev_tools_hot_reload=False,  # Disable hot reload
    threaded=True,            # Enable threading
)

# For heavy loads, use gunicorn:
# gunicorn app:server -w 4 -b 0.0.0.0:8050
```

#### 6.6.3 Monitoring

- **PythonAnywhere:** Built-in CPU/bandwidth monitoring
- **Heroku:** Heroku Metrics dashboard
- **Custom:** Add logging to track usage

---
