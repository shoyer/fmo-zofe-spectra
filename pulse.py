import numpy as np
from numpy import exp, mod, ceil, sqrt, log
try:
    from qspectra.constants import CM_FS
except ImportError:
    from numpy import pi
    # multiply by this constant to convert from fs to linear cm^-1:
    CM_FS_LINEAR = 2.99792458e-5
    # multiply by this constant to convert from fs to angular cm^-1:
    CM_FS = pi * 2 * CM_FS_LINEAR


class PulseError(Exception):
    pass


class ShapedPulse(object):
    def __init__(self, t_delay=0, shape_func=None, shaping=None,
                 num_int_pts=100000, freq_repeat=1, overlap_error_bound=0.01,
                 cutoff_bound=5e-3, cutoff_at_t_final=False, scale=0.1,
                 suppress_errors=False, rw_default=12500, carrier_freq=12500,
                 freq_fwhm=467.87434):
        """
        shape_func should be a function yielding the desired phase (a real number)
        as a function of frequncy. The imput frequency to this function is scaled
        so that the pulse width std. dev. (225 cm^-1) is equal to 1.

        Alternatve, the shaping can be specified directly by supplying the shaping
        argument.

        Calling pulse(t, rw_freq) returns the amplitude at time t for the given
        rw_freq (default rw_freq & carrier freq is 12500 cm^-1).

        NOTE: This class is a bit of hack and not how I would design this today,
        but I'm not going to revise it and introduce more bugs.
        """

        if shaping is None:
            self.set_shaping(shape_func, t_delay, freq_repeat)
        else:
            self.shaping = shaping
        self.freq_fwhm = freq_fwhm
        self.freq_sigma_squared = (freq_fwhm/(2*log(2*sqrt(2))))**2
        self.scale = scale
        self.set_time_domain(num_int_pts, freq_repeat)

        self.cutoff_at_t_final = cutoff_at_t_final

        self.t_init, self.t_final = self.calc_t_limits(overlap_error_bound,
                                                       cutoff_bound)
        if not suppress_errors and self.t_final - self.t_init > 5000:
            raise PulseError("Pulse is longer than 5ps: check pulse shaping")

        self.amp_t_cache = {}
        self.amp_f_cache = {}

        self.carrier_freq = carrier_freq
        self.rw_default = rw_default

        self.call_default = self.interpolate_amp_t

    def set_shaping(self, shape_func, t_delay, freq_repeat):
        if shape_func is None:
            shape_func = lambda x: 0
        scaled_shape_func = lambda x: shape_func(x / 225) - CM_FS * t_delay * x
        df = 2.25 / freq_repeat
        f = df * np.arange(-300 * freq_repeat + .5, 300 * freq_repeat + 0.5)
        self.shaping = np.array(map(scaled_shape_func, f))

    def set_time_domain(self, num_int_pts, freq_repeat):
        N = ceil(num_int_pts / 2) * 2
        df = 2.25 / freq_repeat
        f = df * np.arange(-N / 2 + 0.5, N / 2 + 0.5)
        amp_f = self.scale * exp(-f ** 2 / (2 * self.freq_sigma_squared),
                                 dtype=complex)

        start_shaping = (N / 2 - 300 * freq_repeat)
        end_shaping = (N / 2 + 300 * freq_repeat)
        amp_f[:start_shaping] = 0
        amp_f[start_shaping:end_shaping] *= self.shaping
        amp_f[end_shaping:] = 0

        self.f_range = f
        self.amp_f = amp_f

        dt = 1 / (N * df * 3e-5)
        self.t_range = dt * np.arange(-N / 2, N / 2)
        self.amp_t = np.roll(np.fft.ifft(np.roll(amp_f, int(N / 2))), int(N / 2))

    def calc_t_limits(self, overlap_error_bound, cutoff_bound):
        max_amp = np.max(np.abs(self.amp_t))
        amp_t_zeroout = (np.abs(self.amp_t) < cutoff_bound * max_amp)
        amp_t = self.amp_t.copy()
        amp_t[amp_t_zeroout] = 0

        amp_acc = np.add.accumulate(np.abs(amp_t))
        keep = ((amp_acc < (1 - overlap_error_bound) * amp_acc[-1]) &
                (amp_acc > overlap_error_bound * amp_acc[-1]))

        return self.t_range[keep][0], self.t_range[keep][-1]

    def interpolate_amp_t(self, t, rw_freq):
        if (t, rw_freq) in self.amp_t_cache:
            return self.amp_t_cache[t, rw_freq]
        elif t < self.t_range[0] or t > self.t_range[-1]:
            return 0
        elif self.cutoff_at_t_final and (t > self.t_final):
            return 0
        else:
            N = self.t_range.shape[0]
            dt = self.t_range[1] - self.t_range[0]
            steps = t/dt
            w = mod(steps, 1)
            i = int(steps + N/2)
            amp = (exp(1j*CM_FS*(self.carrier_freq - rw_freq)*t) *
                   ((1-w)*self.amp_t[i] + w*self.amp_t[i+1]))
            self.amp_t_cache[t, rw_freq] = amp
            return amp

    def interpolate_amp_f(self, f0, rw_freq):
        f = f0 - self.carrier_freq + rw_freq
        if (f, rw_freq) in self.amp_f_cache:
            return self.amp_f_cache[f, rw_freq]
        elif f < self.f_range[0] or f > self.f_range[-1]:
            return 0
        else:
            N = self.f_range.shape[0]
            df = self.f_range[1] - self.f_range[0]
            steps = f/df - 0.5
            w = mod(steps, 1)
            i = int(steps + N/2)
            amp = (1-w)*self.amp_f[i] + w*self.amp_f[i+1]
            self.amp_f_cache[f, rw_freq] = amp
            return amp

    def __call__(self, x, rw_freq=None):
        if rw_freq is None:
            rw_freq = self.rw_default

        if np.iterable(x):
            return np.array([self.interpolate_amp_t(x0, rw_freq) for x0 in x])
        else:
            return self.interpolate_amp_t(x, rw_freq)


def poly_pulse(args, **pulse_args):
    sf = lambda x: exp(1j * np.sum(args * x ** np.arange(2, len(args) + 2)))
    return ShapedPulse(shape_func=sf, suppress_errors=True, carrier_freq=12422,
                       **pulse_args)


def poly_pulse_scaled(args, **pulse_args):
    args = (np.array(args) * np.array([1. / ((n + 2) * 2 ** (n + 1))
                                       for n in range(len(args))]))
    return poly_pulse(args, **pulse_args)
