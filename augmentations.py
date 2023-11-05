from scipy.misc import imresize

import utils
import numpy as np

from scipy import ndimage
from utils import to_int


def get_random_transform_params(input_shape, rotation_range = 0., height_shift_range = 0., width_shift_range = 0.,
                                shear_range = 0., zoom_range = (1, 1), horizontal_flip = False, resize_range = None,
                                distortion_prob = 0., additive_gaussian_noise_range = None, multiplication_gaussian = 0,
                                transform_colorspace_param = None, transform_colorspace_bounds = (-1, 1)):
    

    def get_instance(rnd):
        U = rnd.uniform
        N = rnd.normal

        rr = rotation_range
        hs = height_shift_range
        ws = width_shift_range
        sr = shear_range
        agn = additive_gaussian_noise_range
        dp = distortion_prob
        mg = multiplication_gaussian
        tcp = transform_colorspace_param
        tcb = transform_colorspace_bounds

        return {
            'input_shape': input_shape,
            'theta': np.pi / 180 * U(-rr, rr) if rr else 0,
            'ty': U(-hs, hs) * input_shape[0] if hs else 0,
            'tx': U(-ws, ws) * input_shape[1] if ws else 0,
            'shear': U(-sr, sr) if shear_range else 0,
            'z': U(zoom_range[0], zoom_range[1]) if zoom_range != (1, 1) else 1,
            'h_flip': rnd.rand() < 0.5 if horizontal_flip else False,
            'add_noise': N(0, U(agn[0], agn[1]), input_shape) if agn is not None else None,
            'resize': U(*resize_range) if U(0, 1) < dp else None,
            'resize_smooth': U(0, 1) < 0.5,
            'mul': N(1, mg) if mg > 0 else None,
            'color_m': utils.crop_value(N(tcp[0], tcp[1], (3, 3)), tcb) if tcp is not None else None,
            'agn': agn
        }

    return get_instance


def assemble_transformation_matrix(input_shape, theta = 0, tx = 0, ty = 0, shear = 0, z = 1):

    def transform_matrix_offset_center(matrix, x, y):

        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        t_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return t_matrix

    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    translation_matrix = np.array([[1, 0, ty],
                                   [0, 1, tx],
                                   [0, 0, 1]])

    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    zoom_matrix = np.array([[z, 0, 0],
                            [0, z, 0],
                            [0, 0, 1]])

    transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

    h, w = input_shape[0], input_shape[1]
    transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)

    return transform_matrix


def transform(v, t_matrix, h_flip = False, add_noise = None, resize = None, resize_smooth = None,
              mul = None, color_m = None):

    def apply_transform(x, transform_matrix, channel_index = 0, fill_mode = 'nearest', cval = 0.):
        x = np.rollaxis(x, channel_index, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]
        channel_images = [ndimage.interpolation.affine_transform(x_channel, final_affine_matrix,
                                                                 final_offset, order = 2, mode = fill_mode,
                                                                 cval = cval)
                          for x_channel in x]
        x = np.stack(channel_images, axis = 0)
        x = np.rollaxis(x, 0, channel_index + 1)

        return x

    def flip_axis(x, axis):
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        return x

    v = apply_transform(v, t_matrix, 2)

    if h_flip:
        v = flip_axis(v, 1)

    if color_m is not None or mul is not None or add_noise is not None:
        v = v.astype(np.float32)
        shape = v.shape

        if mul is not None:
            v *= mul

        if color_m is not None:
            v = np.reshape(v, [-1, 3])
            v = np.matmul(v, color_m)
            v = np.reshape(v, shape)

        if add_noise is not None:
            v += add_noise

        if resize is not None:
            interpolation = 'bilinear' if resize_smooth else 'nearest'

            v = imresize(v, (resize * np.array(shape[:2])).astype(np.uint16), interpolation)
            v = imresize(v, shape[:2], interpolation)

        v = utils.crop_value(v, [np.zeros(shape), np.ones(shape) * 255])

    return v.astype(np.uint8)


def crop_data(img, labels, new_img_size, new_label_size = None, crop_label = True):

    img_size = img.shape[-3]
    r = to_int((img_size - new_img_size) / 2)

    img = img[..., r:r + new_img_size, r:r + new_img_size, :]

    if crop_label:
        labels -= r

    if new_label_size is not None:
        labels = np.array((labels / new_img_size) * new_label_size, dtype = np.int32)

    return img, labels


def flip_body_joints(points):

    def swap(a, b):
        points[:, [a, b]] = points[:, [b, a]]

    # Leg
    swap(0, 5)
    swap(1, 4)
    swap(2, 3)

    # Arm
    swap(10, 15)
    swap(11, 14)
    swap(12, 13)


def generate_random__transformation(X, rseed = 0, t_params_f = None):
    rnd = np.random.RandomState(rseed)

    if not t_params_f:
        raise Exception('No attributes given!')

    n = X.shape[0]
    X_t = []

    t_params = t_params_f(rnd)
    h_flip = t_params.pop('h_flip')
    add_noise = t_params.pop('add_noise')
    resize = t_params.pop('resize')
    mul = t_params.pop('mul')
    agn = t_params.pop('agn')
    color_m = t_params.pop('color_m')
    resize_smooth = t_params.pop('resize_smooth')

    t_matrix = assemble_transformation_matrix(**t_params)

    for k in range(n):
        inp = np.squeeze(X[k])

        if agn is not None:
            gauss = rnd.normal(0, rnd.uniform(agn[0], agn[1]), inp.shape)
        else:
            gauss = None

        x_t = transform(inp, t_matrix, h_flip, gauss, resize, resize_smooth, mul, color_m)
        X_t.append(x_t)

    return np.array(X_t)


def generate_random_sequences(X, Y, sequence_size = 32, shift = 16, rseed = 0, final_size = None,
                              t_params_f = None, final_heatmap_size = None):
    rnd = np.random.RandomState(rseed)

    if not t_params_f:
        raise Exception('No attributes given!')

    if final_size is None:
        final_size = min(X.shape[2], X.shape[3])

    n = X.shape[0]
    perm = rnd.permutation(range(0, n, shift))
    perm_n = perm.shape[0]

    for idx in range(perm_n):
        b = range(perm[idx], min(perm[idx] + sequence_size, n))

        X_t = []
        Y_t = []

        t_params = t_params_f(rnd)
        h_flip = t_params.pop('h_flip')
        add_noise = t_params.pop('add_noise')
        resize = t_params.pop('resize')
        mul = t_params.pop('mul')
        agn = t_params.pop('agn')
        color_m = t_params.pop('color_m')
        resize_smooth = t_params.pop('resize_smooth')

        t_matrix = assemble_transformation_matrix(**t_params)

        for k in b:
            inp = np.squeeze(X[k])

            if agn is not None:
                gauss = rnd.normal(0, rnd.uniform(agn[0], agn[1]), inp.shape)
            else:
                gauss = None

            x_t = transform(inp, t_matrix, h_flip, gauss, resize, resize_smooth, mul, color_m)
            y_t = utils.get_affine_transform(np.squeeze(Y[k]), np.linalg.inv(t_matrix)) if Y is not None else None

            x_t, y_t = crop_data(x_t, y_t, final_size, final_heatmap_size)

            X_t.append(x_t)
            if Y is not None:
                if h_flip:
                    y_t[1, :] = (final_size if final_heatmap_size is None else final_heatmap_size) - y_t[1, :]
                    flip_body_joints(y_t)

                Y_t.append(y_t)

        if Y is not None:
            yield np.array(X_t), np.array(Y_t), idx
        else:
            yield np.array(X_t), idx


def generate_minibatches(X, Y = None, batch_size = 32, rseed = 0,
                         final_size = None, t_params_f = None, final_heatmap_size = None):

    rnd = np.random.RandomState(rseed)

    if not t_params_f:
        raise Exception('No attributes given!')

    if final_size is None:
        final_size = min(X.shape[2], X.shape[3])

    n = X.shape[0]
    perm = rnd.permutation(n)

    for idx in range(0, n, batch_size):
        b = perm[idx:min(idx + batch_size, n)]

        X_t = []
        Y_t = []

        for k in b:
            t_params = t_params_f(rnd)
            h_flip = t_params.pop('h_flip')
            add_noise = t_params.pop('add_noise')
            resize = t_params.pop('resize')
            mul = t_params.pop('mul')
            color_m = t_params.pop('color_m')
            agn = t_params.pop('agn')
            resize_smooth = t_params.pop('resize_smooth')

            t_matrix = assemble_transformation_matrix(**t_params)

            x_t = transform(np.squeeze(X[k]), t_matrix, h_flip, add_noise, resize, resize_smooth, mul, color_m)
            y_t = utils.get_affine_transform(np.squeeze(Y[k]), np.linalg.inv(t_matrix)) if Y is not None else None

            x_t, y_t = crop_data(x_t, y_t, final_size, final_heatmap_size)

            X_t.append(x_t)
            if Y is not None:
                if h_flip:
                    y_t[1, :] = (final_size if final_heatmap_size is None else final_heatmap_size) - y_t[1, :]
                    flip_body_joints(y_t)

                Y_t.append(y_t)

        if Y is not None:
            yield np.array(X_t), np.array(Y_t), b
        else:
            yield np.array(X_t), b
