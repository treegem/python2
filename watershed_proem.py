import os

import numpy
import pylab
import scipy.ndimage

import mmorph  # part of the pymorph package on PyPI


class AutomatedNvFinder:

    def __init__(self):
        self.frame = self.__load_frame()
        self.blur_radius = 1.
        self.background_level = 600
        self.watershed = self.__perform_watershed()

    def determine_nvs_and_plot(self):

        self.__plot_original()
        self.__plot_all_ws(self.watershed)
        self.__plot_camera_contour(self.__smooth_frame(), self.watershed)

        xs = numpy.arange(self.frame.shape[1])
        ys = numpy.arange(self.frame.shape[0])
        x, y = numpy.meshgrid(xs, ys)

        nv_brightnesses_and_positions, suitable_nv_indices = self.__find_suitable_nvs(self.watershed, x, y)

        nv_areas = self.__determine_nv_areas(suitable_nv_indices)

        # render nv positions
        sigma = 2.5
        p = [self.background_level, sigma] + nv_brightnesses_and_positions

        nvs = self.__render_nvs(p, (xs.min(), xs.max(), ys.min(), ys.max()), self.frame.shape[1], self.frame.shape[0])
        nv_areas = (nv_areas > 0)
        nv_areas = nv_areas * (nvs > (nvs.max() - nvs.min()) * 0.1 + nvs.min())
        for current_ws_index in range(2):
            nv_areas = scipy.ndimage.binary_erosion(nv_areas)
        for current_ws_index in range(2):
            nv_areas = scipy.ndimage.binary_dilation(nv_areas)
        wsi = self.watershed.copy()
        self.watershed = nv_areas * wsi
        count = 0
        for current_ws_index in range(1, self.watershed.max() + 1):
            if current_ws_index in self.watershed:
                count += 1
        print('detected %d NVs' % count)
        pylab.clf()
        pylab.imshow(self.watershed)
        numpy.savetxt("ws.txt", self.watershed)
        pylab.savefig("ws.jpg", dpi=300)
        pylab.clf()
        pylab.imshow((self.watershed > 0) * self.frame)
        numpy.savetxt("roi.txt", (self.watershed > 0) * self.frame)
        pylab.savefig("roi.jpg", dpi=300)

    def __determine_nv_areas(self, suitable_nv_indices):
        all_cropped = numpy.zeros((self.watershed.shape[0], self.watershed.shape[1]))
        for nv in suitable_nv_indices:
            contour_area = (self.watershed == nv)
            brightest_part = self.__determine_brightest_part_within_contour(contour_area)
            smoothed_brightest_part = self.__smoothen_area_contour(brightest_part, contour_area)
            all_cropped += smoothed_brightest_part
        return all_cropped

    @staticmethod
    def __smoothen_area_contour(brightest_part, contour_area):
        brightest_part = scipy.ndimage.binary_erosion(brightest_part)
        for i in range(2):
            brightest_part = scipy.ndimage.binary_dilation(brightest_part.copy())
        brightest_part = (contour_area * brightest_part)
        return brightest_part

    def __determine_brightest_part_within_contour(self, contour_area):
        max_luminescence_above_background = (contour_area * self.__luminescence_above_background()).max()
        optically_active_area_within_contour = (
                                                       contour_area * self.__luminescence_above_background()) > max_luminescence_above_background * 0.5
        return optically_active_area_within_contour

    def __luminescence_above_background(self):
        return self.frame - self.background_level

    @staticmethod
    def __load_frame():
        path = '//file/e24/Projects/ReinhardLab/data_setup_nv1/170306_phase_plates_rabi_echo_pro'
        return numpy.loadtxt(
            os.path.join(path, 'proem_009.txt'))

    def __perform_watershed(self):
        return mmorph.watershed(255 - self.__smooth_frame())

    def __find_suitable_nvs(self, watershed, x, y):
        nv_brightness_and_position = []
        suitable_nv_indices = []
        a_max = 5000  # maximum fluorescence allowed
        a_min = 200  # minimum fluorescence needed to be NV
        data = scipy.ndimage.gaussian_filter(self.frame, self.blur_radius)
        for current_ws_index in range(watershed.max()):
            cell = (watershed == current_ws_index) * (data - self.background_level) * (data > self.background_level)
            if pylab.sum(cell) > 0:
                cell_max = cell.max()
                x0 = pylab.sum(x * cell) / pylab.sum(cell)
                y0 = pylab.sum(y * cell) / pylab.sum(cell)
                if a_min < cell_max < a_max:
                    nv_brightness_and_position += [cell_max, x0, y0]
                    suitable_nv_indices.append(current_ws_index)
        return nv_brightness_and_position, suitable_nv_indices

    def __smooth_frame(self):
        smoothed_frame = self.frame.copy()
        smoothed_frame = numpy.array(256 * smoothed_frame / smoothed_frame.max(), dtype=numpy.uint8)
        smoothed_frame = scipy.ndimage.grey_erosion(smoothed_frame, size=(2, 2))
        smoothed_frame = scipy.ndimage.gaussian_filter(smoothed_frame, self.blur_radius)
        return smoothed_frame

    @staticmethod
    def __plot_all_ws(ws):
        pylab.imshow(ws)
        pylab.savefig('all_ws.png')
        pylab.clf()

    def __plot_camera_contour(self, smoothed_frame, watershed):
        pylab.imshow(scipy.ndimage.gaussian_filter(smoothed_frame, self.blur_radius), cmap=pylab.cm.gray)
        pylab.hold(True)
        pylab.contour(watershed, levels=numpy.arange(watershed.max()))
        pylab.savefig("camera_contour.png", dpi=500)

    def __plot_original(self):
        pylab.imshow(self.frame)
        pylab.savefig("original.jpg", dpi=500)
        pylab.clf()

    @staticmethod
    def __render_nvs(p, extent, w, h):
        result = p[0] * numpy.ones((h, w))  # background
        xs = numpy.linspace(extent[0], extent[1], w)
        ys = numpy.linspace(extent[2], extent[3], h)
        x, y = numpy.meshgrid(xs, ys)
        sigma = p[1]
        for i in range(len(p[2:]) / 3):
            a = p[2 + 3 * i]
            x0 = p[2 + 3 * i + 1]
            y0 = p[2 + 3 * i + 2]
            nv = a * pylab.exp((-(x - x0) ** 2 - (y - y0) ** 2) / 2 / sigma ** 2)
            result += nv
        return result


if __name__ == '__main__':
    plotter = AutomatedNvFinder()
    plotter.determine_nvs_and_plot()
