# -*- coding: utf-8 -*-

import numpy as np
import pvl

from six import string_types
from six.moves import range

from .constants import PIXEL_TYPES, BYTE_ORDERS, SPECIAL_PIXELS


class CubeFile(object):
    """A Isis Cube file reader."""

    @classmethod
    def open(cls, filename):
        """
        Read an Isis Cube file from disk.

        Parameters
        ----------
            filename : string

        Returns
        ----------
            IsisCube
                A new instance of the IsisCube class loaded from file.

        """
        with open(filename, 'rb') as fp:
            return cls(fp, filename)

    def __init__(self, stream_or_fname, filename=None):
        """
        Create an Isis Cube file.
        
        Parameters
        ----------
            stream : file object
                An open Isis .cub file object
            filename : string, optionsl
        
        """
        if isinstance(stream_or_fname, string_types):
            self.filename = stream_or_fname
            stream = open(stream_or_fname, 'rb')
        else:
            #: The filename if given, otherwise none.
            self.filename = filename
            stream = stream_or_fname

        #: The parsed label header in dictionary form.
        self.label = self._parse_label(stream)

        #: A numpy array representing the image data.
        self.data = self._parse_data(stream)

    def apply_scaling(self, copy=True):
        """
        Scale pixel values to their true DN.

        Parameters
        ----------
            copy : boolean 
                scale and return a copy of the array?

        Returns
        ----------
            ndarray
                a scaled version of the pixel data
        
        To-Do
        ----------
        This currently breaks on my machine (Mac OSX 10.11.5) when copy=False.
        The data array in the test case is an int16 array and it complains
        about broadcasting a float (when scaling by the multiplier) to an
        int16 array.
        """
        if copy:
            return self.multiplier * self.data + self.base

        if self.multiplier != 1:
            self.data *= self.multiplier

        if self.base != 0:
            self.data += self.base

        return self.data

    def apply_numpy_specials(self, copy=True):
        """
        Convert isis special pixel values to numpy special pixel values.
        
        Parameters
        ----------
            copy : boolean 
                scale and return a copy of the array?

        Returns
        ----------
            ndarray
                an array with ISIS special values converted to numpy's NaN.
        
        Notes
        ----------
        Comparison table:
            =======  =======
             Isis     NumPy
            =======  =======
            Null     nan
            Lrs      -inf
            Lis      -inf
            His      inf
            Hrs      inf
            =======  =======

        """
        if copy:
            data = self.data.astype(np.float64)

        elif self.data.dtype != np.float64:
            data = self.data = self.data.astype(np.float64)

        else:
            data = self.data

        data[data == self.specials['Null']] = np.nan
        data[data < self.specials['Min']] = np.NINF
        data[data > self.specials['Max']] = np.inf

        return data

    def specials_mask(self):
        """
        Create a pixel map for special pixels.

        Returns
        ----------
            array_like
                 boolean array, False if value is a special pixel, True 
                 otherwise. 

        """
        mask = self.data >= self.specials['Min']
        mask &= self.data <= self.specials['Max']
        return mask

    def get_image_array(self):
        """
        Create an array for use in making an image.

        Creates a linear stretch of the image and scales it to between `0` and
        `255`. `Null`, `Lis` and `Lrs` pixels are set to `0`. `His` and `Hrs`
        pixels are set to `255`.

        Returns
        ----------
            ndarray
                A copy of the data array as type uint8

        Examples
        ----------
            >>>>from pysis import CubeFile
            >>>>from PIL import Image
            >>>>image = CubeFile.open('test.cub')
            >>>>data = image.get_image_array()
            >>>>Image.fromarray(data[0]).save('test.png')

        """
        specials_mask = self.specials_mask()
        data = self.data.copy()

        data[specials_mask] -= data[specials_mask].min()
        data[specials_mask] *= 255 / data[specials_mask].max()

        data[data == self.specials['His']] = 255
        data[data == self.specials['Hrs']] = 255

        return data.astype(np.uint8)

    @property
    def bands(self):
        """Number of image bands."""
        return self.label['IsisCube']['Core']['Dimensions']['Bands']

    @property
    def lines(self):
        """Number of lines per band."""
        return self.label['IsisCube']['Core']['Dimensions']['Lines']

    @property
    def samples(self):
        """Number of samples per line."""
        return self.label['IsisCube']['Core']['Dimensions']['Samples']

    @property
    def tile_lines(self):
        """Number of lines per tile."""
        if self.format != 'Tile':
            return None
        return self.label['IsisCube']['Core']['TileLines']

    @property
    def tile_samples(self):
        """Number of samples per tile."""
        if self.format != 'Tile':
            return None
        return self.label['IsisCube']['Core']['TileSamples']

    @property
    def format(self):
        return self.label['IsisCube']['Core']['Format']

    @property
    def dtype(self):
        """Pixel data type."""
        pixels_group = self.label['IsisCube']['Core']['Pixels']
        byte_order = BYTE_ORDERS[pixels_group['ByteOrder']]
        pixel_type = PIXEL_TYPES[pixels_group['Type']]
        return pixel_type.newbyteorder(byte_order)

    @property
    def specials(self):
        pixel_type = self.label['IsisCube']['Core']['Pixels']['Type']
        return self.SPECIAL_PIXELS[pixel_type]

    @property
    def base(self):
        """An additive factor by which to offset pixel DN."""
        return self.label['IsisCube']['Core']['Pixels']['Base']

    @property
    def multiplier(self):
        """A multiplicative factor by which to scale pixel DN."""
        return self.label['IsisCube']['Core']['Pixels']['Multiplier']

    @property
    def start_byte(self):
        """Index of the start of the image data (zero indexed)."""
        return self.label['IsisCube']['Core']['StartByte'] - 1

    @property
    def shape(self):
        """Tuple of images bands, lines and samples."""
        return (self.bands, self.lines, self.samples)

    @property
    def size(self):
        """Total number of pixels."""
        return self.bands * self.lines * self.samples

    def _parse_label(self, stream):
        """Parses the label of an ISIS cube, returns a pvl module"""
        return pvl.load(stream)

    def _parse_data(self, stream):
        """Parses the image data of an ISIS cube"""
        stream.seek(self.start_byte)

        if self.format == 'BandSequential':
            return self._parse_band_sequential_data(stream)

        if self.format == 'Tile':
            return self._parse_tile_data(stream)

        raise Exception('Unkown Isis Cube format (%s)' % self.format)

    def _parse_band_sequential_data(self, stream):
        """Parses the image data of a band sequential ISIS cube"""
        data = np.fromfile(stream, self.dtype, self.size)
        return data.reshape(self.shape)

    def _parse_tile_data(self, stream):
        """Parses the image data of a tiled ISIS cube"""
        tile_lines = self.tile_lines
        tile_samples = self.tile_samples
        tile_size = tile_lines * tile_samples

        lines = range(0, self.lines, self.tile_lines)
        samples = range(0, self.samples, self.tile_samples)

        dtype = self.dtype
        data = np.empty(self.shape, dtype=dtype)

        for band in data:
            for line in lines:
                for sample in samples:
                    sample_end = sample + tile_samples
                    line_end = line + tile_lines
                    chunk = band[line:line_end, sample:sample_end]

                    tile = np.fromfile(stream, dtype, tile_size)
                    tile = tile.reshape((tile_lines, tile_samples))

                    chunk_lines, chunk_samples = chunk.shape
                    chunk[:] = tile[:chunk_lines, :chunk_samples]

        return data
