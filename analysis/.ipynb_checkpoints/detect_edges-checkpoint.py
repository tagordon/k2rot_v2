from scipy.ndimage import generate_binary_structure, binary_erosion, label
import numpy as np
from gradkde import kde
from gradkde import kernels
from scipy import ndimage as ndi

def detect_edges_scatter(x, y, kernel_width=0.05, gridsize=(1000, 1000)):
    data = np.vstack((x, y)).T
    gauss = kernels.Gaussian()
    dkde = kde.KDE(gauss, kernel_width, nderiv=1)
    dkde.condition(data)
    gridx = np.linspace(x.min(), x.max(), gridsize[0])
    gridy = np.linspace(y.min(), y.max(), gridsize[1])
    grid = np.meshgrid(gridx, gridy)
    points = np.concatenate(np.array(grid).T)
    z = dkde.evaluate_on_grid(grid, separate_vectors=True)
    return gridx, gridy, detect_extrema(z).T

def detect_extrema(z):
    
    '''
    from scki-kit image implementation of canny edge detection.
    '''

    gradx = z[0]
    grady = z[1]
    magnitude = np.linalg.norm(z, axis=0)
    mask = np.ones(magnitude.shape, dtype=bool)
    s = generate_binary_structure(2, 2)
    eroded_mask = binary_erosion(mask, s, border_value=0)
    eroded_mask = eroded_mask & (magnitude > 0)

    local_maxima = np.zeros(magnitude.shape, bool)
    pts_plus = (gradx >= 0) & (grady >= 0) & (np.abs(gradx) >= np.abs(grady))
    pts_minus = (gradx <= 0) & (grady <= 0) & (np.abs(gradx) >= np.abs(grady))
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts

    c1 = magnitude[1:, :][pts[:-1, :]]
    c2 = magnitude[1:, 1:][pts[:-1, :-1]]
    m = magnitude[pts]
    w = np.abs(grady)[pts] / np.abs(gradx)[pts]
    c_plus = c2 * w + c1 * (1 - w) <= m
    c1 = magnitude[:-1, :][pts[1:, :]]
    c2 = magnitude[:-1, :-1][pts[1:, 1:]]
    c_minus = c2 * w + c1 * (1 - w) <= m
    local_maxima[pts] = c_plus & c_minus

    pts_plus = (gradx >= 0) & (grady >= 0) & (np.abs(gradx) <= np.abs(grady))
    pts_minus = (gradx <= 0) & (grady <= 0) & (np.abs(gradx) <= np.abs(grady))
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    c1 = magnitude[:, 1:][pts[:, :-1]]
    c2 = magnitude[1:, 1:][pts[:-1, :-1]]
    m = magnitude[pts]
    w = np.abs(gradx)[pts] / np.abs(grady)[pts]
    c_plus = c2 * w + c1 * (1 - w) <= m
    c1 = magnitude[:, :-1][pts[:, 1:]]
    c2 = magnitude[:-1, :-1][pts[1:, 1:]]
    c_minus = c2 * w + c1 * (1 - w) <= m
    local_maxima[pts] = c_plus & c_minus

    pts_plus = (gradx <= 0) & (grady >= 0) & (np.abs(gradx) <= np.abs(grady))
    pts_minus = (gradx >= 0) & (grady <= 0) & (np.abs(gradx) <= np.abs(grady))
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    c1a = magnitude[:, 1:][pts[:, :-1]]
    c2a = magnitude[:-1, 1:][pts[1:, :-1]]
    m = magnitude[pts]
    w = np.abs(gradx)[pts] / np.abs(grady)[pts]
    c_plus = c2a * w + c1a * (1.0 - w) <= m
    c1 = magnitude[:, :-1][pts[:, 1:]]
    c2 = magnitude[1:, :-1][pts[:-1, 1:]]
    c_minus = c2 * w + c1 * (1.0 - w) <= m
    local_maxima[pts] = c_plus & c_minus

    pts_plus = (gradx <= 0) & (grady >= 0) & (np.abs(gradx) >= np.abs(grady))
    pts_minus = (gradx >= 0) & (grady <= 0) & (np.abs(gradx) >= np.abs(grady))
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    c1 = magnitude[:-1, :][pts[1:, :]]
    c2 = magnitude[:-1, 1:][pts[1:, :-1]]
    m = magnitude[pts]
    w = np.abs(grady)[pts] / np.abs(gradx)[pts]
    c_plus = c2 * w + c1 * (1 - w) <= m
    c1 = magnitude[1:, :][pts[:-1, :]]
    c2 = magnitude[1:, :-1][pts[:-1, 1:]]
    c_minus = c2 * w + c1 * (1 - w) <= m
    local_maxima[pts] = c_plus & c_minus

    high_threshold = 0.2
    low_threshold = 0.1
    high_mask = local_maxima & (magnitude >= high_threshold)
    low_mask = local_maxima & (magnitude >= low_threshold)

    strel = np.ones((3, 3), bool)
    labels, count = label(low_mask, strel)
    if count == 0:
        return low_mask

    sums = (np.array(ndi.sum(high_mask, labels,
                         np.arange(count, dtype=np.int32) + 1),
                 copy=False, ndmin=1))
    good_label = np.zeros((count + 1,), bool)
    good_label[1:] = sums > 0
    output_mask = good_label[labels]
    return output_mask