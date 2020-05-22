import os

import numpy
import pylab
import scipy.ndimage

import mmorph


def main():
    #######################################################
    bg = 600  # background level
    br = 1.  # blur radius in pixels
    frame = numpy.loadtxt(
        os.path.join('//file/e24/Projects/ReinhardLab/data_setup_nv1/170306_phase_plates_rabi_echo_pro',
                     'proem_009.txt'))
    gscan = frame.copy()
    pylab.imshow(frame)
    pylab.savefig("original.jpg", dpi=500)
    pylab.clf()
    gscan = numpy.array(256 * gscan / gscan.max(), dtype=numpy.uint8)
    gscan = scipy.ndimage.grey_erosion(gscan, size=(2, 2))  # - 50000#, structure=kernel)
    gscan = scipy.ndimage.gaussian_filter(gscan, br)
    ws = mmorph.watershed(255 - gscan)
    pylab.imshow(scipy.ndimage.gaussian_filter(gscan, br), cmap=pylab.cm.gray)
    pylab.hold(True)
    pylab.contour(ws, levels=numpy.arange(ws.max()))
    pylab.savefig("camera_contour.png", dpi=1200)
    xs = numpy.arange(frame.shape[1])
    ys = numpy.arange(frame.shape[0])
    x, y = numpy.meshgrid(xs, ys)

    result = []
    nvlist = []
    As = []
    targets = numpy.array([0, 0, 0])
    z0 = 0
    a_max = 5000  # Maximum Flourescence allowed
    a_min = 200  # Min Fl. needed to be NV
    for i in range(ws.max()):
        data = scipy.ndimage.gaussian_filter(frame, br)
        cell = (ws == i) * (data - bg) * (data > bg)
        if pylab.sum(cell) > 0:
            A = cell.max()
            x0 = pylab.sum(x * cell) / pylab.sum(cell)
            y0 = pylab.sum(y * cell) / pylab.sum(cell)
            # print(A)
            if A > a_min and A < a_max:  # and score < 1.5:
                result += [A, x0, y0]
                targets = numpy.vstack((targets, numpy.array((x0, y0, z0))))
                nvlist.append(i)
                As += [A]
            else:
                pass
    all_cropped = numpy.zeros((ws.shape[0], ws.shape[1]))
    for nv in nvlist:
        full = (ws == nv)
        maxi = (full * (frame - bg)).max()
        cropped = ((full * (frame - bg)) > maxi * 0.5)
        cropped = scipy.ndimage.binary_erosion(cropped)
        for i in range(2):
            cropped = scipy.ndimage.binary_dilation(cropped.copy())
        cropped = (full * cropped)
        all_cropped += cropped
    nvlist_copy = nvlist
    # render nv positions
    sigma = 2.5  # bin = 1
    p = [bg, sigma] + result
    nvs = __render_nvs(p, (xs.min(), xs.max(), ys.min(), ys.max()), frame.shape[1], frame.shape[0])
    all_cropped = (all_cropped > 0)
    all_cropped = all_cropped * (nvs > (nvs.max() - nvs.min()) * 0.1 + nvs.min())
    for i in range(2):
        all_cropped = scipy.ndimage.binary_erosion(all_cropped)
    for i in range(2):
        all_cropped = scipy.ndimage.binary_dilation(all_cropped)
    wsi = ws.copy()
    ws = all_cropped * wsi
    count = 0
    for i in range(1, ws.max() + 1):
        if i in ws:
            count += 1
    print('detected %d NVs' % (count))
    pylab.clf()
    pylab.imshow(ws)
    numpy.savetxt("ws.txt", ws)
    pylab.savefig("ws.jpg", dpi=300)
    pylab.clf()
    pylab.imshow((ws > 0) * frame)
    numpy.savetxt("roi.txt", (ws > 0) * frame)
    pylab.savefig("roi.jpg", dpi=300)
    # end of new image processing code - find NVs automatically
    ###############################################################################


def __render_nvs(p, extent, w, h):
    result = p[0] * numpy.ones((h, w))  # background
    xs = numpy.linspace(extent[0], extent[1], w)
    ys = numpy.linspace(extent[2], extent[3], h)
    x, y = numpy.meshgrid(xs, ys)
    sigma = p[1]
    for i in range(len(p[2:]) / 3):
        A = p[2 + 3 * i]
        x0 = p[2 + 3 * i + 1]
        y0 = p[2 + 3 * i + 2]
        nv = A * pylab.exp((-(x - x0) ** 2 - (y - y0) ** 2) / 2 / sigma ** 2)
        result += nv
    return result


if __name__ == '__main__':
    main()
