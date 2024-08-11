import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import cdist
from scipy.interpolate import splprep, splev
import svgwrite
import cairosvg
import os
import matplotlib.colors as mcolors

# Function to read CSV files
def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []

    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)

    return path_XYs

# Function to plot polylines
def plot(paths_XYs):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)

    ax.set_aspect('equal')
    plt.show()

# Functions to identify basic geometric shapes
def is_straight_line(XY):
    if len(XY) < 2:
        return False
    X = XY[:, 0].reshape(-1, 1)
    y = XY[:, 1]
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    return np.allclose(y, y_pred, atol=1e-2)

def is_circle(XY):
    center = XY.mean(axis=0)
    distances = cdist(XY, [center])
    return np.allclose(distances, distances.mean(), atol=1e-2)

def is_rectangle(XY):
    if len(XY) != 4:
        return False
    edges = np.diff(XY, axis=0, append=XY[:1])
    return np.allclose(np.linalg.norm(edges, axis=1), np.linalg.norm(edges, axis=1).mean(), atol=1e-2)

def regularize_shapes(paths_XYs):
    regular_shapes = {'lines': [], 'circles': [], 'rectangles': []}
    
    for path in paths_XYs:
        for XY in path:
            if is_straight_line(XY):
                regular_shapes['lines'].append(XY)
            elif is_circle(XY):
                regular_shapes['circles'].append(XY)
            elif is_rectangle(XY):
                regular_shapes['rectangles'].append(XY)
    
    return regular_shapes

# Function to find symmetry line
def find_symmetry_line(XY):
    center = XY.mean(axis=0)
    distances = cdist(XY, [center])
    symmetrical_pairs = []

    for i, point in enumerate(XY):
        closest_point_idx = np.argmin(distances[i])
        symmetrical_pairs.append((point, XY[closest_point_idx]))

    return symmetrical_pairs

# Function to plot symmetry
def plot_symmetry(paths_XYs):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
            pairs = find_symmetry_line(XY)
            for p1, p2 in pairs:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', linewidth=1)
    
    ax.set_aspect('equal')
    plt.show()

# Function to complete curve
def complete_curve(XY):
    # Check if XY has at least four points (for cubic splines)
    if len(XY) < 4:
        print("Not enough points to create a spline, skipping this path.")
        return XY
    
    # Check if XY is a 2D array with two columns
    if XY.shape[1] != 2:
        print("Invalid shape for input data, expected a 2D array with two columns.")
        return XY
    
    # Ensure the data is numeric
    if not np.issubdtype(XY.dtype, np.number):
        print("Non-numeric data found, skipping this path.")
        return XY

    # Try creating the spline
    try:
        tck, u = splprep([XY[:, 0], XY[:, 1]], s=0, per=True)
        new_points = splev(np.linspace(0, 1, 1000), tck)
        return np.column_stack(new_points)
    except Exception as e:
        print(f"Error in creating spline: {e}")
        return XY


def complete_incomplete_curves(paths_XYs):
    completed_paths = []

    for path in paths_XYs:
        completed_path = [complete_curve(XY) for XY in path]
        completed_paths.append(completed_path)

    return completed_paths

# Function to generate SVG and PNG
def polylines2svg(paths_XYs, svg_path):
    colours = list(mcolors.CSS4_COLORS.values())  # Use valid CSS4 color values
    W, H = 0, 0
    
    for path_XYs in paths_XYs:
        for XY in path_XYs:
            W, H = max(W, np.max(XY[:, 0])), max(H, np.max(XY[:, 1]))

    padding = 0.1
    W, H = int(W + padding * W), int(H + padding * H)

    dwg = svgwrite.Drawing(svg_path, profile='tiny', shape_rendering='crispEdges')
    group = dwg.g()

    for i, path in enumerate(paths_XYs):
        path_data = []
        c = colours[i % len(colours)]
        for XY in path:
            path_data.append(("M", (XY[0, 0], XY[0, 1])))
            for j in range(1, len(XY)):
                path_data.append(("L", (XY[j, 0], XY[j, 1])))
            if not np.allclose(XY[0], XY[-1]):
                path_data.append(("Z", None))
        group.add(dwg.path(d=path_data, fill=c, stroke='none', stroke_width=2))
    
    dwg.add(group)
    dwg.save()

    png_path = svg_path.replace('.svg', '.png')
    fact = max(1, 1024 // min(H, W))
    cairosvg.svg2png(url=svg_path, write_to=png_path, parent_width=W, parent_height=H, output_width=fact * W, output_height=fact * H, background_color='white')

# Process and visualize polylines for all CSV files in the 'problems' directory
problems_dir = "machineavengers\problems" 

csv_files = [f for f in os.listdir(problems_dir) if f.endswith('.csv')]

for csv_file in csv_files:
    csv_path = os.path.join(problems_dir, csv_file)
    paths_XYs = read_csv(csv_path)
    
    # Plot original polylines
    print(f"Plotting polylines from {csv_file}")
    plot(paths_XYs)
    
    # Regularize shapes
    regular_shapes = regularize_shapes(paths_XYs)
    print(f"Shapes in {csv_file}: { {shape_type: len(shapes) for shape_type, shapes in regular_shapes.items()} }")
    
    # Plot symmetry
    print(f"Plotting symmetry for {csv_file}")
    plot_symmetry(paths_XYs)
    
    # Complete incomplete curves
    completed_paths_XYs = complete_incomplete_curves(paths_XYs)
    print(f"Plotting completed curves for {csv_file}")
    plot(completed_paths_XYs)
    
    # Generate SVG and PNG
    svg_path = csv_path.replace('.csv', '.svg')
    polylines2svg(paths_XYs, svg_path)
    print(f"Generated {svg_path} and corresponding PNG file")