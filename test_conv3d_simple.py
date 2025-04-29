import torch
import torch.nn as nn
import unittest

class TestConvTranspose3d(unittest.TestCase):
    def test_basic(self):
        conv = nn.ConvTranspose3d(2, 3, kernel_size=3)
        x = torch.randn(1, 2, 4, 4, 4)
        output = conv(x)
        self.assertEqual(output.shape, (1, 3, 6, 6, 6))

    def test_padding_zeros(self):
        conv = nn.ConvTranspose3d(2, 3, kernel_size=3, padding_mode='zeros')
        x = torch.randn(1, 2, 4, 4, 4)
        output = conv(x)
        self.assertEqual(output.shape, (1, 3, 6, 6, 6))

    def test_padding_invalid(self):
        with self.assertRaises(ValueError):
            nn.ConvTranspose3d(2, 3, kernel_size=3, padding_mode='reflect')

if __name__ == '__main__':
    unittest.main() 