"""
Collection of constants.
"""
import numpy

__all__ = ['SPECIAL_PIXELS']

def _make_num(num, dtype):
    return numpy.fromstring(num, dtype=dtype)[0]

# ISIS special pixels 
SPECIAL_PIXELS = {
    'UnsignedByte': {
        'Min': 1,
        'Null': 0,
        'Lrs': 0,
        'Lis': 0,
        'His': 255,
        'Hrs': 255,
        'Max': 254
    },
    'UnsignedWord': {
        'Min': 3,
        'Null': 0,
        'Lrs': 1,
        'Lis': 2,
        'His': 65534,
        'Hrs': 65535,
        'Max': 65523
    },
    'SignedWord': {
        'Min': -32763,
        'Null': -32768,
        'Lrs': -32767,
        'Lis': -32766,
        'His': -32765,
        'Hrs': -32764,
        'Max': 32767
    },
    'SignedInteger': {
        'Min': -8388608,
        'Null': -8388613,
        'Lrs': -8388612,
        'Lis': -8388611,
        'His': -8388610,
        'Hrs': -8388609,
        'Max': 2147483647
    },
    'Real': {
        'Min': _make_num(b'\xFF\x7F\xFF\xFA', '>f4'),
        'Null': _make_num(b'\xFF\x7F\xFF\xFB', '>f4'),
        'Lrs': _make_num(b'\xFF\x7F\xFF\xFC', '>f4'),
        'Lis': _make_num(b'\xFF\x7F\xFF\xFD', '>f4'),
        'His': _make_num(b'\xFF\x7F\xFF\xFE', '>f4'),
        'Hrs': _make_num(b'\xFF\x7F\xFF\xFF', '>f4'),
        'Max': numpy.finfo('f4').max
    },
    'Double': {
        'Min': _make_num(b'\xFF\xEF\xFF\xFF\xFF\xFF\xFF\xFA', '>f8'),
        'Null': _make_num(b'\xFF\xEF\xFF\xFF\xFF\xFF\xFF\xFB', '>f8'),
        'Lrs': _make_num(b'\xFF\xEF\xFF\xFF\xFF\xFF\xFF\xFC', '>f8'),
        'Lis': _make_num(b'\xFF\xEF\xFF\xFF\xFF\xFF\xFF\xFD', '>f8'),
        'His': _make_num(b'\xFF\xEF\xFF\xFF\xFF\xFF\xFF\xFE', '>f8'),
        'Hrs': _make_num(b'\xFF\xEF\xFF\xFF\xFF\xFF\xFF\xFF', '>f8'),
        'Max': numpy.finfo('f8').max
    }
}

# Dictionary mapping ISIS datatypes to NumPy datatypes
PIXEL_TYPES = {
    'UnsignedByte': np.dtype('uint8'),
    'SignedByte': np.dtype('int8'),
    'UnsignedWord': np.dtype('uint16'),
    'SignedWord': np.dtype('int16'),
    'UnsignedInteger': np.dtype('uint32'),
    'SignedInteger': np.dtype('int32'),
    'Real': np.dtype('float32'),
    'Double': np.dtype('float64')
}

# Dictionary mapping ISIS byte orders to NumPy byte orders
BYTE_ORDERS = {
    'NoByteOrder': '=',  # system
    'Lsb': '<',          # little-endian
    'Msb': '>'           # big-endian
}