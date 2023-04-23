import keras.models
from keras.applications.resnet import ResNet50
from keras.models import load_model, Model
from keras.layers import *
import tensorflow as tf


class ResNet50Modify:
    def __init__(self, size=None):
        self.size = size


    def conv1_block(self, x):
        x = Conv2D(filters=64, kernel_size=(7,7), strides=(2, 2), padding='valid', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
        return x


    def identity_block(self, mid_c, out_c, dilation, x):
        fx = Conv2D(filters=mid_c, kernel_size=(1, 1), strides=(1, 1), use_bias=False)(x)
        fx = BatchNormalization()(fx)
        fx = ReLU()(fx)
        fx = Conv2D(filters=mid_c, kernel_size=(3, 3), strides=(1, 1), dilation_rate=dilation, padding='same',
                    use_bias=False)(fx)
        fx = BatchNormalization()(fx)
        fx = ReLU()(fx)
        fx = Conv2D(filters=out_c, kernel_size=(1, 1), strides=(1, 1), use_bias=False)(fx)
        fx = BatchNormalization()(fx)
        hx = fx + x
        hx = ReLU()(hx)
        return hx


    def convolution_block(self, mid_c, out_c, stride, dilation, padding, x):
        fx = Conv2D(filters=mid_c, kernel_size=(1, 1), strides=(1, 1), use_bias=False)(x)
        fx = BatchNormalization()(fx)
        fx = ReLU()(fx)
        fx = Conv2D(filters=mid_c, kernel_size=(3, 3), strides=stride, dilation_rate=dilation, padding=padding,
                    use_bias=False)(fx)
        fx = BatchNormalization()(fx)
        fx = ReLU()(fx)
        fx = Conv2D(filters=out_c, kernel_size=(1, 1), strides=(1, 1), use_bias=False)(fx)
        fx = BatchNormalization()(fx)
        if mid_c == 64:   # specially for conv2 (stage1)
            cx = Conv2D(filters=out_c, kernel_size=(1, 1), strides=(1, 1), use_bias=False)(x)
        else:
            cx = Conv2D(filters=out_c, kernel_size=(3, 3), strides=stride, dilation_rate=dilation, padding=padding,
                        use_bias=False)(x)
        cx = BatchNormalization()(cx)
        hx = fx + cx
        hx = ReLU()(hx)
        return hx


    def conv2_block(self, x):
        x = self.convolution_block(64, 256, stride=(1,1), dilation=1, padding='same', x=x)
        x = self.identity_block(64, 256, dilation=1, x=x)
        x = self.identity_block(64, 256, dilation=1, x=x)
        return x


    def conv3_block(self, x):
        x = self.convolution_block(128, 512, stride=(2, 2), dilation=1, padding='valid', x=x)
        x = self.identity_block(128, 512, dilation=1, x=x)
        x = self.identity_block(128, 512, dilation=1, x=x)
        x = self.identity_block(128, 512, dilation=1, x=x)
        return x


    def conv4_block(self, x):
        x = self.convolution_block(256, 1024, stride=(1, 1), dilation=1, padding='same', x=x)
        x = self.identity_block(256, 1024, dilation=2, x=x)
        x = self.identity_block(256, 1024, dilation=2, x=x)
        x = self.identity_block(256, 1024, dilation=2, x=x)
        x = self.identity_block(256, 1024, dilation=2, x=x)
        x = self.identity_block(256, 1024, dilation=2, x=x)
        return x


    def conv5_block(self, x):
        x = self.convolution_block(512, 2048, stride=(1, 1), dilation=2, padding='same', x=x)
        x = self.identity_block(512, 2048, dilation=4, x=x)
        x = self.identity_block(512, 2048, dilation=4, x=x)
        return x


    def neck(self, x):
        x = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), use_bias=False)(x)
        x = BatchNormalization()(x)
        return x


    def __call__(self):
        inp = Input(shape=(self.size, self.size, 3))
        x = self.conv1_block(inp)
        x = self.conv2_block(x)
        x3 = self.conv3_block(x)
        x4 = self.conv4_block(x3)
        x5 = self.conv5_block(x4)
        x3 = self.neck(x3)
        x4 = self.neck(x4)
        x5 = self.neck(x5)
        model = Model(inputs=inp, outputs=[x3, x4, x5], name='backbone0')
        return model
        # return x3, x4, x5



class DepthwiseCorr(Layer):
    def __init__(self, k_size, x_size):
        super().__init__()
        template = Input(shape=(k_size, k_size, 256))
        search = Input(shape=(x_size, x_size, 256))
        correlation = []
        for h in range(x_size-k_size+1):
            for w in range(x_size-k_size+1):
                a = search[:, h:h+k_size, w:w+k_size, :] * template[:, :, :, :]
                a = tf.reduce_sum(a, axis=1)
                a = tf.reduce_sum(a, axis=1)
                correlation.append(a)
        correlation = tf.convert_to_tensor(correlation)
        correlation = tf.transpose(correlation, (1, 0, 2))
        correlation = Reshape(target_shape=(x_size-k_size+1, x_size-k_size+1, correlation.shape[2]))(correlation)
        self.model = Model([search, template], correlation)

    def __call__(self, inputs):
        return self.model(inputs)



class SiamRPNpp:
    def __init__(self, template_size, search_size, temp_crop_size, num_anchors):
        self.template_size = template_size
        self.search_size = search_size
        self.temp_crop_size = temp_crop_size
        self.num_anchors = num_anchors


    def rpn(self, template, search, category='cls'):
        template = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), use_bias=False)(template)
        template = BatchNormalization()(template)
        template = ReLU()(template)
        template = CenterCrop(self.temp_crop_size, self.temp_crop_size)(template)

        search = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), use_bias=False)(search)
        search = BatchNormalization()(search)
        search = ReLU()(search)

        k_size, x_size = self.temp_crop_size, search.shape[1]
        correlation = DepthwiseCorr(k_size, x_size)([search, template])
        correlation = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), use_bias=False)(correlation)
        correlation = BatchNormalization()(correlation)
        correlation = ReLU()(correlation)
        if category == 'cls':
            correlation = Conv2D(filters=2*self.num_anchors, kernel_size=(1, 1), strides=(1, 1),
                                 use_bias=False)(correlation)
        else:
            assert category == 'loc', "False at anything"
            correlation = Conv2D(filters=4*self.num_anchors, kernel_size=(1, 1), strides=(1, 1),
                                 use_bias=False)(correlation)
        return correlation


    def __call__(self):
        backbone = ResNet50Modify()()
        inp_template = Input(shape=(self.template_size, self.template_size, 3))
        inp_search = Input(shape=(self.search_size, self.search_size, 3))
        template = backbone(inp_template)
        search = backbone(inp_search)

        cls0 = self.rpn(template[0], search[0], category='cls')
        cls1 = self.rpn(template[1], search[1], category='cls')
        cls2 = self.rpn(template[2], search[2], category='cls')
        cls = Add()([cls0, cls1, cls2])

        loc0 = self.rpn(template[0], search[0], category='loc')
        loc1 = self.rpn(template[1], search[1], category='loc')
        loc2 = self.rpn(template[2], search[2], category='loc')
        loc = Add()([loc0, loc1, loc2])

        model = Model(inputs=[inp_template, inp_search], outputs=[cls, loc])
        return model



if __name__ == "__main__":
    siamrpnpp = SiamRPNpp(127, 255, 7, 5)()
    siamrpnpp.summary()
    siamrpnpp.save("saved_models/demo.h5")

