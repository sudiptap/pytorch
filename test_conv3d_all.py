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

    def test_stride(self):
        conv = nn.ConvTranspose3d(2, 3, kernel_size=3, stride=2)
        x = torch.randn(1, 2, 4, 4, 4)
        output = conv(x)
        self.assertEqual(output.shape, (1, 3, 9, 9, 9))

    def test_dilation(self):
        conv = nn.ConvTranspose3d(2, 3, kernel_size=3, dilation=2)
        x = torch.randn(1, 2, 4, 4, 4)
        output = conv(x)
        self.assertEqual(output.shape, (1, 3, 8, 8, 8))

    def test_groups(self):
        conv = nn.ConvTranspose3d(4, 6, kernel_size=3, groups=2)
        x = torch.randn(1, 4, 4, 4, 4)
        output = conv(x)
        self.assertEqual(output.shape, (1, 6, 6, 6, 6))
        self.assertEqual(conv.weight.shape, (4, 3, 3, 3, 3))

    def test_no_bias(self):
        conv = nn.ConvTranspose3d(2, 3, kernel_size=3, bias=False)
        self.assertIsNone(conv.bias)
        x = torch.randn(1, 2, 4, 4, 4)
        output = conv(x)
        self.assertEqual(output.shape, (1, 3, 6, 6, 6))

    def test_output_padding(self):
        conv = nn.ConvTranspose3d(2, 3, kernel_size=3, stride=2, output_padding=1)
        x = torch.randn(1, 2, 4, 4, 4)
        output = conv(x)
        self.assertEqual(output.shape, (1, 3, 10, 10, 10))

if __name__ == '__main__':
    unittest.main() 