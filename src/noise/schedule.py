import math
import torch


eps = 1e-11


def log(t):
    return torch.log(t.clamp(min = eps))


def logsnr_schedule_cosine(t, logsnr_min = -15, 
                              logsnr_max = 15):

    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))

    return -2 * log(torch.tan(t_min + t * (t_max - t_min)))


def logsnr_schedule_shifted(fn, image_d, noise_d):
    shift = 2 * math.log(noise_d / image_d)

    @wraps(fn)
    def inner(*args, **kwargs):
        nonlocal shift
        return fn(*args, **kwargs) + shift

    return inner


def logsnr_schedule_interpolated(fn, image_d, noise_d_low, noise_d_high):
    logsnr_high_fn = logsnr_schedule_shifted(fn, image_d, noise_d_high)
    logsnr_low_fn = logsnr_schedule_shifted(fn, image_d, noise_d_low)

    @wraps(fn)
    def inner(t, *args, **kwargs):
        nonlocal logsnr_low_fn
        nonlocal logsnr_high_fn

        return (1-t) * logsnr_high_fn(t, *args, **kwargs) + \
                  t  *  logsnr_low_fn(t, *args, **kwargs)

    return inner


