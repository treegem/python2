import os

import numpy
import pylab
import scipy.ndimage

import mmorph  # part of the pymorph package on PyPI


class AutomatedNvFinder:

    def __init__(self, file_path):
        self.frame = self.__load_frame(file_path)
        self.blur_radius = 1.
        self.background_level = 600
        self.watershed = self.__perform_watershed()

    def determine_nvs_and_plot(self):

        self.__plot_original()
        self.__plot_all_watershed_areas()
        self.__save_and_plot_camera_contour()

        nv_areas = self.__determine_round_and_clean_nv_areas()

        self.watershed = self.__remove_empty_watershed_areas(nv_areas)
        self.__count_nvs()
        self.__save_and_plot_watershed_areas_of_suitable_nvs()
        self.__save_and_plot_roi()

    def __plot_original(self):
        pylab.imshow(self.frame)
        pylab.savefig("original.jpg", dpi=500)
        pylab.clf()

    def __plot_all_watershed_areas(self):
        numpy.savetxt('all_ws.txt', self.watershed)
        pylab.imshow(self.watershed)
        pylab.savefig('all_ws.png')
        pylab.clf()

    def __save_and_plot_camera_contour(self):
        smoothed_frame = scipy.ndimage.gaussian_filter(self.__smooth_frame(), self.blur_radius)
        numpy.savetxt('smoothed_frame.txt', smoothed_frame)
        pylab.imshow(smoothed_frame, cmap=pylab.cm.gray)
        pylab.hold(True)
        pylab.contour(self.watershed, levels=numpy.arange(self.watershed.max()))
        pylab.savefig("camera_contour.png", dpi=500)
        pylab.clf()

    def __smooth_frame(self):
        smoothed_frame = self.frame.copy()
        smoothed_frame = numpy.array(256 * smoothed_frame / smoothed_frame.max(), dtype=numpy.uint8)
        smoothed_frame = scipy.ndimage.grey_erosion(smoothed_frame, size=(2, 2))
        smoothed_frame = scipy.ndimage.gaussian_filter(smoothed_frame, self.blur_radius)
        return smoothed_frame

    def __determine_round_and_clean_nv_areas(self):
        xs, ys = numpy.meshgrid(numpy.arange(self.frame.shape[1]), numpy.arange(self.frame.shape[0]))
        nv_brightnesses_and_positions, suitable_nv_indices = self.__find_suitable_nvs(self.watershed, xs, ys)
        nv_areas = self.__determine_nv_areas(suitable_nv_indices)
        nv_areas = self.__rounden_nv_areas(nv_areas, nv_brightnesses_and_positions, xs, ys)
        nv_areas = self.__clean_up_nv_areas(nv_areas)
        return nv_areas

    def __find_suitable_nvs(self, watershed, xs, ys):
        nv_brightness_and_position = []
        suitable_nv_indices = []
        a_max = 5000  # maximum fluorescence allowed
        a_min = 200  # minimum fluorescence needed to be NV
        data = scipy.ndimage.gaussian_filter(self.frame, self.blur_radius)
        for current_ws_index in range(watershed.max()):
            cell = (watershed == current_ws_index) * (data - self.background_level) * (data > self.background_level)
            if pylab.sum(cell) > 0:
                cell_max = cell.max()
                x0 = pylab.sum(xs * cell) / pylab.sum(cell)
                y0 = pylab.sum(ys * cell) / pylab.sum(cell)
                if a_min < cell_max < a_max:
                    nv_brightness_and_position += [cell_max, x0, y0]
                    suitable_nv_indices.append(current_ws_index)
        return nv_brightness_and_position, suitable_nv_indices

    def __determine_nv_areas(self, suitable_nv_indices):
        nv_areas = numpy.zeros((self.watershed.shape[0], self.watershed.shape[1]))
        for nv in suitable_nv_indices:
            contour_area = (self.watershed == nv)
            brightest_part = self.__determine_brightest_part_within_contour(contour_area)
            smoothed_brightest_part = self.__smoothen_area_contour(brightest_part, contour_area)
            nv_areas += smoothed_brightest_part
        return nv_areas

    def __determine_brightest_part_within_contour(self, contour_area):
        luminescence_within_contour = self.__luminescence_within_contour(contour_area)
        max_luminescence_above_background = luminescence_within_contour.max()
        bright_area_within_contour = luminescence_within_contour > max_luminescence_above_background * 0.5
        return bright_area_within_contour

    def __luminescence_within_contour(self, contour_area):
        return contour_area * self.__luminescence_above_background()

    def __luminescence_above_background(self):
        return self.frame - self.background_level

    @staticmethod
    def __smoothen_area_contour(brightest_part, contour_area):
        brightest_part = scipy.ndimage.binary_erosion(brightest_part)
        for i in range(3):
            brightest_part = scipy.ndimage.binary_dilation(brightest_part.copy())
        brightest_part = (contour_area * brightest_part)
        return brightest_part

    def __rounden_nv_areas(self, nv_areas, nv_brightnesses_and_positions, xs, ys):
        gaussian_nvs = self.__render_gaussian_at_nv_positions(nv_brightnesses_and_positions, xs, ys)
        nv_areas = (nv_areas > 0)
        nv_areas = nv_areas * (gaussian_nvs > (gaussian_nvs.max() - gaussian_nvs.min()) * 0.1 + gaussian_nvs.min())
        return nv_areas

    def __render_gaussian_at_nv_positions(self, nv_brightnesses_and_positions, x, y):
        result = self.background_level * numpy.ones_like(self.frame)  # background
        sigma = 2.5  # standard deviation
        for i in range(len(nv_brightnesses_and_positions) / 3):
            nv = self.render_gaussian_at_nv_position(i, nv_brightnesses_and_positions, sigma, x, y)
            result += nv
        return result

    @staticmethod
    def render_gaussian_at_nv_position(i, nv_brightnesses_and_positions, sigma, x, y):
        brightness = nv_brightnesses_and_positions[3 * i]
        x0 = nv_brightnesses_and_positions[3 * i + 1]
        y0 = nv_brightnesses_and_positions[3 * i + 2]
        nv = brightness * pylab.exp((-(x - x0) ** 2 - (y - y0) ** 2) / 2 / sigma ** 2)
        return nv

    @staticmethod
    def __clean_up_nv_areas(nv_areas):
        for repetition in range(2):
            nv_areas = scipy.ndimage.binary_erosion(nv_areas)
        for repetition in range(2):
            nv_areas = scipy.ndimage.binary_dilation(nv_areas)
        return nv_areas

    def __remove_empty_watershed_areas(self, nv_areas):
        return nv_areas * self.watershed

    def __count_nvs(self):
        count = 0
        for current_ws_index in range(1, self.watershed.max() + 1):
            if current_ws_index in self.watershed:
                count += 1
        print('detected %d NVs' % count)

    def __save_and_plot_watershed_areas_of_suitable_nvs(self):
        pylab.clf()
        pylab.imshow(self.watershed)
        numpy.savetxt("ws.txt", self.watershed)
        pylab.savefig("ws.jpg", dpi=300)

    def __save_and_plot_roi(self):
        pylab.clf()
        pylab.imshow((self.watershed > 0) * self.frame)
        numpy.savetxt("roi.txt", (self.watershed > 0) * self.frame)
        pylab.savefig("roi.jpg", dpi=300)

    @staticmethod
    def __load_frame(file_path):
        return numpy.loadtxt(file_path)

    def __perform_watershed(self):
        return mmorph.watershed(255 - self.__smooth_frame())


if __name__ == '__main__':
    folder = '//file/e24/Projects/ReinhardLab/data_setup_nv1/170306_phase_plates_rabi_echo_pro'
    full_file_path = os.path.join(folder, 'proem_010.txt')
    plotter = AutomatedNvFinder(full_file_path)
    plotter.determine_nvs_and_plot()
