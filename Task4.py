import numpy as np
import matplotlib.pyplot as plt
import svgwrite
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from flask import Flask, request, send_file, render_template_string
import os
import io

app = Flask(__name__)

def read_csv(csv_file):
    np_path_XYs = np.genfromtxt(io.StringIO(csv_file.decode('utf-8')), delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def plot(paths_XYs):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def polylines2svg(paths_XYs):
    W, H = 0, 0
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for path_XYs in paths_XYs:
        for XY in path_XYs:
            W, H = max(W, np.max(XY[:, 0])), max(H, np.max(XY[:, 1]))
    padding = 0.1
    W, H = int(W + padding * W), int(H + padding * H)
    dwg = svgwrite.Drawing(profile='tiny', width=W, height=H)
    group = dwg.g()
    for i, path_XYs in enumerate(paths_XYs):
        path_data = []
        c = colours[i % len(colours)]
        for XY in path_XYs:
            path_str = f"M {XY[0, 0]},{XY[0, 1]} "
            for j in range(1, len(XY)):
                path_str += f"L {XY[j, 0]},{XY[j, 1]} "
            if not np.allclose(XY[0], XY[-1]):
                path_str += "Z"
            group.add(dwg.path(d=path_str, fill=c, stroke='none', stroke_width=2))
    dwg.add(group)
    buf = io.BytesIO()
    dwg.saveas(buf)
    buf.seek(0)
    return buf

def svg_to_png(svg_file):
    drawing = svg2rlg(io.BytesIO(svg_file.read()))
    buf = io.BytesIO()
    renderPM.drawToFile(drawing, buf, fmt='PNG')
    buf.seek(0)
    return buf

@app.route('/')
def index():
    return render_template_string('''
        <h1>File Upload</h1>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <label for="file">Upload file:</label>
            <input type="file" name="file" id="file" required>
            <br>
            <label for="file_type">File type:</label>
            <select name="file_type" id="file_type" required>
                <option value="csv">CSV</option>
                <option value="svg">SVG</option>
            </select>
            <br>
            <input type="submit" value="Upload">
        </form>
    ''')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    file_type = request.form['file_type']
    
    if file_type == 'csv':
        path_XYs = read_csv(file.read())
        buf_svg = polylines2svg(path_XYs)
        return send_file(buf_svg, mimetype='image/svg+xml', as_attachment=True, download_name='output.svg')
    
    elif file_type == 'svg':
        buf_png = svg_to_png(file)
        return send_file(buf_png, mimetype='image/png', as_attachment=True, download_name='output.png')
    
    return "Invalid file type. Please upload a CSV or SVG file."

if __name__ == "__main__":
    app.run(debug=True)
    