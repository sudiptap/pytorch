import unittest
import torch
import torch.nn as nn
import itertools

class TestConvTranspose3d(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.in_channels = 3
        self.out_channels = 4
        self.kernel_size = 3
        self.input_size = (5, 6, 7)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtypes = [torch.float32, torch.float64]
        if self.device == 'cuda':
            self.dtypes.append(torch.half)

    def test_basic_functionality(self):
        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                m = nn.ConvTranspose3d(self.in_channels, self.out_channels, self.kernel_size).to(self.device, dtype)
                input = torch.randn(self.batch_size, self.in_channels, *self.input_size, device=self.device, dtype=dtype)
                output = m(input)
                self.assertEqual(output.shape[0], self.batch_size)
                self.assertEqual(output.shape[1], self.out_channels)
                for i in range(3):
                    self.assertEqual(output.shape[i + 2], self.input_size[i] + self.kernel_size - 1)

    def test_stride_and_padding(self):
        strides = [(1, 1, 1), (2, 2, 2)]
        paddings = [(0, 0, 0), (1, 1, 1)]
        for stride, padding in itertools.product(strides, paddings):
            with self.subTest(stride=stride, padding=padding):
                m = nn.ConvTranspose3d(
                    self.in_channels, self.out_channels, self.kernel_size,
                    stride=stride, padding=padding
                ).to(self.device)
                input = torch.randn(self.batch_size, self.in_channels, *self.input_size, device=self.device)
                output = m(input)
                expected_shape = [
                    (self.input_size[i] - 1) * stride[i] - 2 * padding[i] + self.kernel_size
                    for i in range(3)
                ]
                for i in range(3):
                    self.assertEqual(output.shape[i + 2], expected_shape[i])

    def test_output_padding(self):
        stride = (2, 2, 2)
        padding = (1, 1, 1)
        output_padding = (1, 1, 1)
        m = nn.ConvTranspose3d(
            self.in_channels, self.out_channels, self.kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        ).to(self.device)
        input = torch.randn(self.batch_size, self.in_channels, *self.input_size, device=self.device)
        output = m(input)
        expected_shape = [
            (self.input_size[i] - 1) * stride[i] - 2 * padding[i] + self.kernel_size + output_padding[i]
            for i in range(3)
        ]
        for i in range(3):
            self.assertEqual(output.shape[i + 2], expected_shape[i])

    def test_groups(self):
        groups = 2
        m = nn.ConvTranspose3d(4, 4, self.kernel_size, groups=groups).to(self.device)
        input = torch.randn(self.batch_size, 4, *self.input_size, device=self.device)
        output = m(input)
        self.assertEqual(output.shape[1], 4)

    def test_dilation(self):
        dilation = (2, 2, 2)
        m = nn.ConvTranspose3d(
            self.in_channels, self.out_channels, self.kernel_size,
            dilation=dilation
        ).to(self.device)
        input = torch.randn(self.batch_size, self.in_channels, *self.input_size, device=self.device)
        output = m(input)
        effective_kernel_size = (self.kernel_size - 1) * dilation[0] + 1
        expected_shape = [
            self.input_size[i] + effective_kernel_size - 1
            for i in range(3)
        ]
        for i in range(3):
            self.assertEqual(output.shape[i + 2], expected_shape[i])

    def test_size_1_kernel(self):
        kernel_size = 1
        m = nn.ConvTranspose3d(self.in_channels, self.out_channels, kernel_size).to(self.device)
        input = torch.randn(self.batch_size, self.in_channels, *self.input_size, device=self.device)
        output = m(input)
        for i in range(3):
            self.assertEqual(output.shape[i + 2], self.input_size[i])

    def test_empty_input(self):
        m = nn.ConvTranspose3d(self.in_channels, self.out_channels, self.kernel_size).to(self.device)
        input = torch.randn(0, self.in_channels, *self.input_size, device=self.device)
        output = m(input)
        self.assertEqual(output.shape[0], 0)
        self.assertEqual(output.shape[1], self.out_channels)

    def test_non_contiguous(self):
        m = nn.ConvTranspose3d(self.in_channels, self.out_channels, self.kernel_size).to(self.device)
        input = torch.randn(self.batch_size, self.in_channels, *self.input_size, device=self.device)
        input = input.permute(0, 2, 1, 3, 4).contiguous().permute(0, 2, 1, 3, 4)
        self.assertFalse(input.is_contiguous())
        output = m(input)  # Should handle non-contiguous input
        self.assertEqual(output.shape[1], self.out_channels)

    def test_padding_mode_zeros(self):
        m = nn.ConvTranspose3d(
            self.in_channels, self.out_channels, self.kernel_size,
            padding=1, padding_mode='zeros'
        ).to(self.device)
        input = torch.randn(self.batch_size, self.in_channels, *self.input_size, device=self.device)
        output = m(input)
        self.assertTrue(output is not None)  # Basic smoke test

    def test_padding_mode_invalid(self):
        with self.assertRaises(ValueError):
            nn.ConvTranspose3d(
                self.in_channels, self.out_channels, self.kernel_size,
                padding=1, padding_mode='invalid'
            )

if __name__ == '__main__':
    unittest.main() 