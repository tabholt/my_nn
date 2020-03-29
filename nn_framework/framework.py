import os
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("agg")


class ANN(object):
    def __init__(self, model, error_function, printer, normalized_pixel_range=[-.5, .5], input_pixel_range=[np.nan, np.nan]):
        self.layers = model
        self.error_function = error_function
        self.printer = printer
        self.error_history = []
        self.n_iter_train = int(1e6)
        self.n_iter_evaluate = int(1e6)
        self.viz_interval = int(1e5)
        self.reporting_bin_size = int(1e3)
        self.report_min = -3
        self.report_max = 0

        self.normalized_pixel_range = normalized_pixel_range
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
            error = self.error_function.calc(x, y)
            rms_error = (np.mean(error**2))**.5
            self.error_history.append(rms_error)
            de_dy = self.error_function.calc_d(x, y)

            if (i+1) % self.viz_interval == 0:
                self.report()
                self.printer.render(
                    self, x, name='iter_%d_vis' % (i + 1))

    def evaluate(self, evaluation_set):
        for i in range(self.n_iter_evaluate):
            x = next(evaluation_set()).ravel()
            x = self.normalize(x)
            y = self.forward_prop(x)
            error = self.error_function.calc(x, y)
            rms_error = np.sqrt(np.mean(error))
            self.error_history.append(rms_error)

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

    def forward_prop(self, x):
        # convert inputs into 2d array of right shape
        y = x.ravel()[np.newaxis, :]
        for layer in self.layers:
            y = layer.forward_prop(y)
        return y.ravel()

    def forward_prop_to_layer(self, x, i_layer):
        y = x.ravel()[np.newaxis, :]
        for layer in self.layers[:i_layer]:
            y = layer.forward_prop(y)
        return y.ravel()

    def forward_prop_from_layer(self, x, i_layer):
        y = x.ravel()[np.newaxis, :]
        for layer in self.layers[i_layer:]:
            y = layer.forward_prop(y)
        return y.ravel()

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
