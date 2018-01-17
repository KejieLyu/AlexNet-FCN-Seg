form network import Network

class CaffeFCN(Network):
    """
    Implementation of CaffeNet-FCN model
    """
    def setup(self):
        # conv1 -> relu1 -> pool1 -> lcrn1
        self.conv1 = self.conv(self.inputs, 11, 11, 96, 4, 4, "conv1", relu=True, padding="VALID", trainable=self.trainable)
        self.pool1 = self.max_pool(self.conv1, 3, 3, 2, 2, "pool1")
        self.lcrn1 = self.lrn(self.pool1, "norm1")
        # conv2 -> relu2 -> pool2 -> lcrn2
        self.conv2 = self.conv(self.lcrn1, 5, 5, 256, 1, 1, "conv2", relu=True, group=2, trainable=self.trainable)
        self.pool2 = self.max_pool(self.conv2, 3, 3, 2, 2, "pool2")
        self.lcrn2 = self.lrn(self.pool2, "norm2")
        # conv3 -> relu3
        self.conv3 = self.conv(self.lcrn2, 3, 3, 384, 1, 1, "conv3", relu=True, trainable=self.trainable)
        # conv4 -> relu4
        self.conv4 = self.conv(self.conv3, 3, 3, 384, 1, 1, "conv4", relu=True, group=2, trainable=self.trainable)
        # conv5 -> relu5 -> pool5
        self.conv5 = self.conv(self.conv4, 3, 3, 256, 1, 1, "conv5", relu=True, group=2, trainable=self.trainable)
        self.pool5 = self.max_pool(self.conv5, 3, 3, 2, 2, "pool5")
        # fc6 -> relu6 -> drop6
        self.fc6 = self.conv(self.pool5, 6, 6, 4096, 1, 1, "fc6", relu=True, trainable=self.trainable)
        self.drop6 = self.dropout(self.fc6, self.keep_p[0], "drop6")
        # fc7 -> relu7 -> drop7
        self.fc7 = self.conv(self.drop6, 1, 1, 4096, 1, 1, "fc7", relu=True, trainable=self.trainable)
        self.drop7 = self.dropout(self.fc7, self.keep_p[1], "drop7")
        # score
        self.score = self.conv(self.drop7, 1, 1, 40, 1, 1, "score_fr", relu=False, trainable=self.trainable)
        # upsample
        self.score_up = self.upsample(self.score, "outputs")
        
