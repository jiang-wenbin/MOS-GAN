# generator, model_nlayer_nchannel_nfft

# generator
# model_nlayer_nchannel
UNet_6_128 = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    "encoder":
        [[  0,  32, (5, 2), (2, 1), (1, 1)],
         ( 32,  64, (5, 2), (2, 1), (2, 1)),
         ( 64, 128, (5, 2), (2, 1), (2, 1))],
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    "decoder":
        [(256,  64, (5, 2), (2, 1), (2, 0), (1, 0)),
         (128,  32, (5, 2), (2, 1), (2, 0), (1, 0)),
         [ 64,   0, (5, 2), (2, 1), (1, 0), (0, 0), True]]
}
