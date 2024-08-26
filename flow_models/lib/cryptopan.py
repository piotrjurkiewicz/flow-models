# https://github.com/FFM/pycryptopan

#   pycryptopan - a python module implementing the CryptoPAn algorithm
#   Copyright (C) 2013 - the CONFINE project

#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3 of the
#   License, or (at your option) any later version.

#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.

#   You should have received a copy of the GNU Lesser General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

from functools import reduce

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


class CryptoPan:
    def __init__(self, key):
        if len(key) != 32:
            raise ValueError("Key must be a 32 byte long string")
        self.aes = Cipher(algorithms.AES(key[0:16]), modes.ECB()).encryptor()  # noqa: S305
        self.pad = self.aes.update(key[16:32])
        f4 = self.pad[0:4]
        f4bp = self.toint(f4)
        self.masks = [(mask, f4bp & (~ mask)) for mask in (0xFFFFFFFF >> (32 - p) << (32 - p) for p in range(32))]

    def toint(self, array):
        return array[0] << 24 | array[1] << 16 | array[2] << 8 | array[3]

    def toarray(self, n):
        for i in range(3, -1, -1):
            yield (n >> (i * 8)) & 0xFF

    def anonymize(self, ip):
        address = ip

        def calc(a):
            a_array = self.toarray(a)
            inp = bytes(a_array)
            inp += self.pad[4:]
            rin_output = self.aes.update(inp)
            out = rin_output[0]
            return out >> 7

        addresses = ((address & mask[0]) | mask[1] for mask in self.masks)
        result = reduce(lambda x, y: x << 1 | y, (calc(a) for a in addresses), 0)
        return result ^ address
