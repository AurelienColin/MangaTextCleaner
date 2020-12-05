import os
import glob
from PIL import Image, ImageDraw
import numpy as np
from skimage.transform import resize
from skimage.morphology import binary_erosion

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import fire

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

OUTPUT_FOLDER = "./cleaned"
COLOR = 'tab:blue'
FIGSIZE = (10, 8)

from deprecation_warnings import filter_warnings
filter_warnings()

from keras.models import load_model as keras_load_model

model = keras_load_model('model.h5', compile=False)


class Annotater:
    def update_legend(self):
        if np.isscalar(self.selected_color):
            self.selected_color = (self.selected_color, self.selected_color, self.selected_color)
        selected_color = [color / [255, 1][color < 1] for color in self.selected_color]

        plt.legend(handles=[mpatches.Patch(color=selected_color, label="Selected Color"),
                            mpatches.Patch(color=(1, 1, 1),
                                           label=f"Rectangle mode is {('off', 'on')[self.rectangle_mode]}. "
                                           "Toogle with 'r'")],
                   loc='center left', bbox_to_anchor=(-0.5, 1.1))
        plt.draw()

    def button_press_event(self, event):
        def left_click_event():
            if event.xdata is None or event.ydata is None:
                print('Error: "central click" should add a point, but you clicked outside the figure')
                return

            (x1, y1) = (int(event.xdata), int(event.ydata))
            if self.rectangle_mode and len(self.mask_points) == 1:
                x0, y0 = self.mask_points[0]
                for point in [(x0, y1), (x1, y1), (x1, y0), (x0, y0)]:
                    self.mask_points.append(point)
                    self.add_point(point)
            else:
                self.mask_points.append((x1, y1))
                self.add_point((x1, y1))

        def central_click_event():
            if event.xdata is None or event.ydata is None:
                print('Error: "left click" should select a color, but you clicked outside the figure')
                return
            self.selected_color = np.array(self.output_image)[int(event.ydata), int(event.xdata)]
            self.update_legend()

        def right_click_event():
            """
            If points were plotted, remove the last point.
            Else, if a mask was drawn, remove the last mask.
            Else, go to the previous picture.
            """
            if not self.plotted_points:
                if self.history:
                    self.output_image = self.history.pop(-1)

                    self.plotted_image[1].remove()
                    self.plotted_image[1] = self.axes[1].imshow(self.output_image, interpolation="bilinear")
                    plt.draw()
                elif self.index:
                    print('go to previous image')
                    self.index -= 1
                    self.new_image()
            else:
                self.remove_point()
                self.mask_points.pop(-1)

        _ = (left_click_event, central_click_event, right_click_event)[event.button.value - 1]()

    def key_press_event(self, event):
        def p_press_event():
            self.mask_points = []
            if self.filenames:
                self.index += 1
                self.new_image()
            else:
                plt.close()

        def enter_press_event():
            self.draw_masks()
            self.mask_points = []

            if self.filenames:
                self.index += 1
                self.new_image()
            else:
                plt.close()

        def zero_press_event():
            """
            Can be done after adding at least three points to a polygon, validate the current polygon.
            """
            if len(self.mask_points) < 3:
                print('Error: Not enough point to create a polygon')
                return
            self.add_mask()
            self.mask_points = []

        def r_press_event():
            """
            Can be done when no class is selected, toogle the rectangle mode
            """
            if len(self.mask_points) > 1:
                print("Error: Can't switch to rectangle mode when more than one point was drawn")
            else:
                self.rectangle_mode = False if self.rectangle_mode else True
                print(f'Rectangle Mode is now at {self.rectangle_mode}')
                self.update_legend()

        if event.key == "p":
            p_press_event()
        if event.key == "enter":
            enter_press_event()
        elif event.key == "0":
            zero_press_event()
        elif event.key == 'r':
            r_press_event()

    def new_image(self, border=0.05):
        self.history = []
        filename = self.filenames[self.index]

        self.current_filename = filename

        self.input_image = np.array(Image.open(filename))
        if len(self.input_image.shape) == 2:
            self.input_image = np.expand_dims(self.input_image, axis=-1)
            self.input_image = self.input_image[:, :, [0, 0, 0]]
        self.output_image = self.input_image.copy()

        self.current_mask = Image.new('L', self.input_image.shape[:2])
        self.current_mask_draw = ImageDraw.Draw(self.current_mask)

        self.process_image(self.input_image)

        if self.plotted_image[1] is not None:
            self.plotted_image[1].remove()

        self.plotted_image[1] = self.axes[1].imshow(self.output_image, interpolation="bilinear", cmap='gray')

        x_border = int(border * self.input_image.shape[1])
        y_border = int(border * self.input_image.shape[0])
        self.axes[0].set_xlim((-x_border, self.input_image.shape[1] + x_border))
        self.axes[0].set_ylim((self.input_image.shape[0] + y_border, -y_border))
        self.axes[1].set_xlim((-x_border, self.input_image.shape[1] + x_border))
        self.axes[1].set_ylim((self.input_image.shape[0] + y_border, -y_border))

        self.update_legend()

    def add_mask(self):
        manual_mask = Image.new('RGB', (self.input_image.shape[1], self.input_image.shape[0]))
        ImageDraw.Draw(manual_mask).polygon(self.mask_points, fill=(1, 1, 1))
        manual_mask = np.array(manual_mask)

        self.history.append(self.output_image)
        self.output_image = self.output_image.copy()
        np.putmask(self.output_image, np.logical_and(self.model_mask, manual_mask), self.selected_color)

        self.plotted_image[1].remove()
        self.plotted_image[1] = self.axes[1].imshow(self.output_image, interpolation="bilinear")

        while self.plotted_points:
            self.remove_point()
        plt.draw()

    def add_point(self, point, point_marker='+', linestyle='--'):
        self.plotted_points.append(self.axes[0].scatter(point[0], point[1], c=COLOR, marker=point_marker))
        if len(self.mask_points) > 1:
            (x1, y1), (x2, y2) = self.mask_points[-2:]
            self.plotted_lines.extend(self.axes[0].plot((x1, x2), (y1, y2), c=COLOR, linestyle=linestyle))
        plt.draw()

    def remove_point(self):
        self.plotted_points.pop(-1).remove()
        if self.plotted_lines:
            self.plotted_lines.pop(-1).remove()
        plt.draw()

    def process_image(self, im, threshold=0.5, opening_radius=5, opening_iteration=1):
        input_shape = model.input_shape[1:]
        im = resize(im, input_shape[:2], anti_aliasing=True)
        self.model_mask = model.predict(np.expand_dims(im, axis=0))[0, :, :, 0]

        self.model_mask = np.where(self.model_mask < threshold, 0, 1)

        for _ in range(opening_iteration):
            self.model_mask = binary_erosion(self.model_mask, np.ones((opening_radius, opening_radius)))

        self.model_mask = resize(self.model_mask, self.input_image.shape[:2], anti_aliasing=True)

        adjusted_mask = np.multiply(self.model_mask, self.input_image[:, :, 1]).astype('uint8') // 3
        self.input_image[:, :, 0] -= adjusted_mask
        self.input_image[:, :, 2] -= adjusted_mask

        if self.plotted_image[0] is not None:
            self.plotted_image[0].remove()
        self.plotted_image[0] = self.axes[0].imshow(self.input_image, interpolation="bilinear")

        self.model_mask = np.expand_dims(self.model_mask, axis=-1)[:, :, [0, 0, 0]]

    def draw_masks(self):
        current_filename = self.filenames[self.index]
        new_filename = os.path.join(self.output_folder, os.path.split(current_filename)[-1])
        os.makedirs(os.path.split(new_filename)[0], exist_ok=True)
        Image.fromarray(self.output_image).save(new_filename)

    def __init__(self, path, output_folder, figsize=FIGSIZE):
        self.index = 0
        self.filenames = glob.glob(path)
        self.output_folder = output_folder
        self.rectangle_mode = True
        self.selected_color = (255, 255, 255)

        self.fig, self.axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        plt.tight_layout(rect=(0, 0, 1, 0.8))

        self.axes[0].set_anchor('N')
        self.axes[1].set_anchor('N')

        self.axes[0].set_xticks([])
        self.axes[1].set_xticks([])
        self.axes[0].set_yticks([])
        self.axes[1].set_yticks([])
        self.connexions = {
            "button_press_event": self.fig.canvas.mpl_connect("button_press_event", self.button_press_event),
            "key_press_event": self.fig.canvas.mpl_connect("key_press_event", self.key_press_event)
        }

        self.plotted_points = []
        self.plotted_lines = []
        self.plotted_image = [None, None]
        self.current_mask = None

        self.mask_points = []

        self.new_image()
        plt.show()


def main(path="./to_clean/*.jpg", output_folder=OUTPUT_FOLDER, figsize=FIGSIZE):
    print("Here a small manual that explains how to use this app          \r\n"
          "---------------------------------------------------------------\r\n"
          "                     HOW TO LAUNCH?                            \r\n"
          "To run it, you can use the default command:                    \r\n"
          "\t\"python MangaTextCleaner.py\"                               \r\n"
          "The app will be launched with the defaults parameters.         \r\n"
          "These parameters are:                                          \r\n"
          "-path: You can use it to specify the path to your file.        \r\n"
          "\tThe default value is \"./to_clean/*.jpg\"                    \r\n"
          "-output_folder: You can use it change the output folder.       \r\n"
          "\tThe default value is \"./cleaned\"                           \r\n"
          "---------------------------------------------------------------\r\n"
          "                     HOW TO RUN?                               \r\n"
          "Six actions are possible:                                      \r\n"
          "- The left click adds points to the annotater.                 \r\n"
          "\tBy default, it draw rectangles (see below) so that only      \r\n"
          "\ttwo clicks are needed to draw a polygon.                     \r\n"
          "- The center click copies a color on the input image.          \r\n"
          "\tThis color will be used to fill the removed areas.           \r\n"
          "\tBy default, the selected color is white, but this            \r\n"
          "\taction is usefull if the cell to clean is black.             \r\n"
          "- The right click removes points. If no points are drawn,      \r\n"
          "\tit will remove the last polygon. If neither polygons         \r\n"
          "\tnor points are added, it will go to the previous picture.    \r\n"
          "- The 0 key of the numpad is used to validate a polygon.       \r\n"
          "\tAfter validation, the content of the segmented region        \r\n"
          "\twill be erased on the left panel. Then you can proceed       \r\n"
          "\tto add more polygons.                                        \r\n"
          "- The Enter key validates the picture and go the next one.     \r\n"
          "\tAt this moment, the output is generated and all recorded     \r\n"
          "\tpolygons are discared, so it is to be used with caution.     \r\n"
          "- The 'r' key toogle the rectangle mode. In this mode, the     \r\n"
          "\tsecond click put three points at the same time to form a     \r\n"
          "\trectangle."
          )
    
    Annotater(path, output_folder, figsize=figsize)


if __name__ == '__main__':
    fire.Fire(main)
