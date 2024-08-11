import cv2
import numpy as np

# Only PNG Format Files are Supported

def compute_centroid(points):
    points = np.array(points)
    return np.mean(points, axis=0)

def check_vertical_symmetry(points, line_x):
    reflected_points = [(2 * line_x - x, y) for (x, y) in points]
    reflected_points = sorted([tuple(np.round(p).astype(int)) for p in reflected_points])
    points = sorted([tuple(np.round(p).astype(int)) for p in points])
    return reflected_points == points

def check_horizontal_symmetry(points, line_y):
    reflected_points = [(x, 2 * line_y - y) for (x, y) in points]
    reflected_points = sorted([tuple(np.round(p).astype(int)) for p in reflected_points])
    points = sorted([tuple(np.round(p).astype(int)) for p in points])
    return reflected_points == points

def check_diagonal_symmetry(points, line_x, line_y):
    reflected_points_45 = [(line_x - (y - line_y), line_y + (x - line_x)) for (x, y) in points]
    reflected_points_135 = [(line_x + (y - line_y), line_y - (x - line_x)) for (x, y) in points]
    reflected_points_45 = sorted([tuple(np.round(p).astype(int)) for p in reflected_points_45])
    reflected_points_135 = sorted([tuple(np.round(p).astype(int)) for p in reflected_points_135])
    points = sorted([tuple(np.round(p).astype(int)) for p in points])
    return reflected_points_45 == points, reflected_points_135 == points

image = cv2.imread('AdobeR2Final\circle.png', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error loading image")
    exit()

edges = cv2.Canny(image, 50, 150)

circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=100)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

for contour in contours:
    points = contour[:, 0, :]
    centroid = compute_centroid(points)
    centroid_x, centroid_y = centroid

    if check_vertical_symmetry(points, centroid_x):
        cv2.line(output_image, (int(centroid_x), 0), (int(centroid_x), image.shape[0]), (0, 0, 255), 2)

    if check_horizontal_symmetry(points, centroid_y):
        cv2.line(output_image, (0, int(centroid_y)), (image.shape[1], int(centroid_y)), (0, 0, 255), 2)
    
    diag1, diag2 = check_diagonal_symmetry(points, centroid_x, centroid_y)
    if diag1:
        cv2.line(output_image, (0, image.shape[1]), (image.shape[1], 0), (0, 0, 255), 2)
    if diag2:
        cv2.line(output_image, (0, 0), (image.shape[1], image.shape[1]), (0, 0, 255), 2)

    cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(output_image, (x, y), r, (0, 255, 0), 2)
        cv2.circle(output_image, (x, y), 2, (0, 0, 255), 3)
        num_lines = 8
        for i in range(num_lines):
            angle = 2 * np.pi * i / num_lines
            x_end = int(x + r * np.cos(angle))
            y_end = int(y + r * np.sin(angle))
            cv2.line(output_image, (x, y), (x_end, y_end), (0, 0, 255), 2)

cv2.imshow('Symmetry Detection', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
