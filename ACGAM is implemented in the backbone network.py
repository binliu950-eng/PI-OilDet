from Innovation.ACGAM import AdaptiveGaborAttention


class MobileNetV2(nn.Module):
    def __init__(self, num_gabor_filters, downsample_factor=8, pretrained=True):
        # super(MobileNetV2, self).__init__()
        # from functools import partial
        #
        # model = mobilenetv2(pretrained)
        # self.features = model.features[:-1]
        #
        # self.total_idx = len(self.features)
        # self.down_idx = [2, 4, 7, 14]

        # ----------ACGAM模块------------#
        self.acgam_module = AdaptiveGaborAttention(
            in_channels=24,
            # num_filters=num_gabor_filters,
            num_filters=num_gabor_filters,
            kernel_size=15  # 核尺寸调整
        )

        # if downsample_factor == 8:
        #     for i in range(self.down_idx[-2], self.down_idx[-1]):
        #         self.features[i].apply(
        #             partial(self._nostride_dilate, dilate=2)
        #         )
        #     for i in range(self.down_idx[-1], self.total_idx):
        #         self.features[i].apply(
        #             partial(self._nostride_dilate, dilate=4)
        #         )
        # elif downsample_factor == 16:
        #     for i in range(self.down_idx[-1], self.total_idx):
        #         self.features[i].apply(
        #             partial(self._nostride_dilate, dilate=2)
        #         )

    # def _nostride_dilate(self, m, dilate):
    #     classname = m.__class__.__name__
    #     if classname.find('Conv') != -1:
    #         if m.stride == (2, 2):
    #             m.stride = (1, 1)
    #             if m.kernel_size == (3, 3):
    #                 m.dilation = (dilate // 2, dilate // 2)
    #                 m.padding = (dilate // 2, dilate // 2)
    #         else:
    #             if m.kernel_size == (3, 3):
    #                 m.dilation = (dilate, dilate)
    #                 m.padding = (dilate, dilate)

    def forward(self, x):
        feature1 = self.features[:4](x)
        low_level_features = feature1
        enhanced_feature = self.acgam_module(feature1)
        feature1 = self.features[4:](enhanced_feature)
        return low_level_features, feature1