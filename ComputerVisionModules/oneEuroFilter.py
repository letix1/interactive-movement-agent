import math


class OneEuroFilter:
    """
    A low-pass filter that adapts to the signal speed:
    - when the joint moves slow, cutoff drops -> heavy smoothing (kills jitter)
    - when the joint moves fast, cutoff rises      -> low lag (keeps responsiveness)

    Parameters:
        min_cutoff : minimum cutoff freq (Hz). Lower = smoother signal, more lag.
        beta       : speed coefficient. Higher = cutoff rises faster with motion,
                     reducing lag during fast movements.
        d_cutoff   : cutoff used when low-passing the derivative (rarely tuned).
        angular    : if True, unwrap input around ±π before filtering so that
                     the filter doesn't sweep through 360° at wrap boundaries.

    Starting values for MediaPipe joint angles @ ~15-30 FPS:
        min_cutoff = 1.0, beta = 0.007  (see Casiez et al., "1€ Filter", CHI 2012).
    """

    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0, angular=False):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.angular = angular
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x

        if self.angular:
            n = round((self.x_prev - x) / (2 * math.pi))
            x = x + n * 2 * math.pi

        dt = t - self.t_prev if t != self.t_prev else 1e-6
        dx = (x - self.x_prev) / dt

        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)