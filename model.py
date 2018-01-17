class CaffeFCN(Network):
    """
    Implementation of CaffeNet-FCN model
    """
    def setup(self):
        # Add the input of the network to layers dict
        self.layers["inputs"] = self.inputs
        # conv1 -> relu1 -> pool1 -> lcrn1
        conv1 = self.conv(self.inputs, 11, 11, 96, 4, 4, "conv1", relu=True, padding="VALID", trainable=self.trainable)
        pool1 = self.max_pool(conv1, 3, 3, 2, 2, "pool1")
        lcrn1 = self.lrn(pool1, "norm1")
        # conv2 -> relu2 -> pool2 -> lcrn2
        conv2 = self.conv(lcrn1, 5, 5, 256, 1, 1, "conv2", relu=True, group=2, trainable=self.trainable)
        pool2 = self.max_pool(conv2, 3, 3, 2, 2, "pool2")
        lcrn2 = self.lrn(pool2, "norm2")
        # conv3 -> relu3
        conv3 = self.conv(lcrn2, 3, 3, 384, 1, 1, "conv3", relu=True, trainable=self.trainable)
        # conv4 -> relu4
        conv4 = self.conv(conv3, 3, 3, 384, 1, 1, "conv4", relu=True, group=2, trainable=self.trainable)
        # conv5 -> relu5 -> pool5
        conv5 = self.conv(conv4, 3, 3, 256, 1, 1, "conv5", relu=True, group=2, trainable=self.trainable)
        pool5 = self.max_pool(conv5, 3, 3, 2, 2, "pool5")
        # fc6 -> relu6 -> drop6
        fc6 = self.conv(pool5, 6, 6, 4096, 1, 1, "fc6", relu=True, trainable=self.trainable)
        drop6 = self.dropout(fc6, self.keep_p[0], "drop6")
        # fc7 -> relu7 -> drop7
        fc7 = self.conv(drop6, 1, 1, 4096, 1, 1, "fc7", relu=True, trainable=self.trainable)
        drop7 = self.dropout(fc7, self.keep_p[1], "drop7")
        # score
        score = self.conv(drop7, 1, 1, 40, 1, 1, "score_fr", relu=False, trainable=self.trainable)
        # upsample
        score_up = self.upsample(score, "outputs")
        