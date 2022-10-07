import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


def smooth_scatter(x, y, log=False, resolution=50):
    JointDistribution(x, y, log).fit().draw(resolution)


class JointDistribution(object):
    def __init__(self, x1, x2, log=False):
        self.x1 = x1
        self.x2 = x2
        self.log = log
        self.joint_estimate = None

    def fit(self):
        x1 = np.log10(self.x1) if self.log else self.x1
        x2 = np.log10(self.x2) if self.log else self.x2

        self.joint_estimate = gaussian_kde(np.vstack([x1, x2]))
        return self

    def draw(self, resolution=100):
        resolution *= 1j
        x1 = np.log10(self.x1) if self.log else self.x1
        x2 = np.log10(self.x2) if self.log else self.x2

        xmin, xmax = np.percentile(x1, [1, 99])
        ymin, ymax = np.percentile(x2, [1, 99])
        X, Y = np.mgrid[xmin:xmax:resolution, ymin:ymax:resolution]
        positions = np.vstack([X.ravel(), Y.ravel()])

        Z = np.reshape(self.joint_estimate(positions), X.shape).T
        plt.imshow(Z, interpolation='nearest', origin='lower')

        locs = np.arange(0, int(resolution.imag), int(resolution.imag) // 6)
        if self.log:
          plt.xticks(locs, [f'{10**e:.02f}' for e in X[locs, 0].squeeze()])
          plt.yticks(locs, [f'{10**e:.02f}' for e in Y[0, locs].squeeze()])
        else:
          plt.xticks(locs, [f'{e:.02f}' for e in X[locs, 0].squeeze()])
          plt.yticks(locs, [f'{e:.02f}' for e in Y[0, locs].squeeze()])
