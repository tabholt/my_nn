import os
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("agg")


class ANN(object):
    def __init__(self, model, error_function, printer, input_pixel_range=[np.nan, np.nan]):
        self.layers = model
        self.error_function = error_function
        self.printer = printer
        self.error_history = []
        self.n_iter_train = int(5e5)
        self.n_iter_evaluate = int(2e5)
        self.viz_interval = int(1e5)
        self.reporting_bin_size = int(1e3)
        self.report_min = -3
        self.report_max = 0

        self.input_pixel_range = input_pixel_range

        self.reports_path = 'reports'
        self.report_name = 'performance_history.png'

        try:
            os.mkdir(self.reports_path)
        except:
            pass

    def train(self, training_set):
        for i in range(self.n_iter_train):
            x = next(training_set()).ravel()
            x = self.normalize(x)
            y = self.forward_prop(x)
            error = self.error_function.calc(y)
            self.error_history.append(error)
            de_dy = self.error_function.calc_d(y)
            self.back_prop(de_dy)

            if (i+1) % self.viz_interval == 0:
                self.report()
                self.printer.render(
                    self, x, name='iter_%d_vis' % (i + 1))

    def evaluate(self, evaluation_set):
        for i in range(self.n_iter_evaluate):
            x = next(evaluation_set()).ravel()
            x = self.normalize(x)
            y = self.forward_prop(x, evaluating=True)
            error = self.error_function.calc(y)
            self.error_history.append(error)

            if (i+1) % self.viz_interval == 0:
                self.report()
                self.printer.render(
                    self, x, name='iter_%d_vis' % (i + 1))

    def normalize(self, pic):
        """
        Transform the input picture so that all pixel values fall 
        between the desired normalized_pixel_range
        """
        if np.isnan(self.input_pixel_range[0]):
            high = np.max(pic)
            low = np.min(pic)
            dist = high-low
        else:
            high = self.input_pixel_range[1]
            low = self.input_pixel_range[0]
            dist = high-low
        pic_percent = (pic - low) / dist
        pic_normalized = pic_percent * \
            (self.normalized_pixel_range[1]-self.normalized_pixel_range[0]
             ) + self.normalized_pixel_range[0]
        return pic_normalized

    def denormalize(self, pic, original_range=[0, 1]):
        high = np.max(pic)
        low = np.min(pic)
        dist = high-low
        pic_percent = (pic - low) / dist
        pic_normalized = pic_percent * \
            (original_range[1]-original_range[0]) + original_range[0]
        return pic_normalized

    def forward_pass(self, x, evaluating=False, i_start_layer=None, i_stop_layer=None):
        if i_start_layer is None:
            i_start_layer = 0
        if i_stop_layer is None:
            i_stop_layer = len(self.layers)
        if i_stop_layer <= i_start_layer:
            return x

        for layer in self.layers:
            layer.reset()

        self.layers[i_start_layer].x += x.ravel()[np.newaxis, :]

        for layer in self.layers[i_start_layer: i_stop_layer]:
            layer.forward_pass(evaluating=evaluating)

        return layer.y.ravel()

    def backward_pass(self, de_dy):
        self.layers[-1].de_dy += de_dy
        for layer in self.layers[::-1]:
            layer.backward_pass()

    def report(self):
        n_bins = int(len(self.error_history)) // self.reporting_bin_size
        smoothed_history = []
        for i_bin in range(n_bins):
            smoothed_history.append(np.mean(self.error_history[
                i_bin * self.reporting_bin_size:
                (i_bin+1) * self.reporting_bin_size
            ]))
        error_history = np.log10(np.array(smoothed_history)+1e-10)
        ymin = np.minimum(self.report_min, np.min(error_history))
        ymax = np.maximum(self.report_max, np.max(error_history))

        fig = plt.figure()
        ax = plt.gca()
        ax.plot(error_history)
        ax.set_xlabel('x%d iterations' % (self.reporting_bin_size))
        ax.set_ylabel('log error')
        ax.set_ylim(ymin, ymax)
        ax.grid()
        fig.savefig(os.path.join(self.reports_path, self.report_name))
        plt.close()
