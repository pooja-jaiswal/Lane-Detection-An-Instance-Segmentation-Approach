"""
Implement a full convolutional network decoding class
Paper used for this is https://arxiv.org/pdf/1411.4038.pdf
"""
import tensorflow as tf

import cnn_basenet
#import vgg_encoder


class FCNDecoder(cnn_basenet.CNNBaseModel):
    """
    Implement a full convolutional decoding class
    """
    def __init__(self, phase):
        """
        Initializing the FCN Decoder
        """
        super(FCNDecoder, self).__init__()
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()

    def _init_phase(self):
        """

        :return:
        """
        return tf.equal(self._phase, self._train_phase)

    def decode(self, input_tensor_dict, decode_layer_list, name):
        """
        Decoding feature information deconvolution reduction
        :param input_tensor_dict:
        :param decode_layer_list: The layer names that need to be decoded need to be written from deep to shallow
                                  eg. ['pool5', 'pool4', 'pool3']
        :param name:
        :return:
        """
        ret = dict()

        with tf.variable_scope(name):
            # score stage 1
            # Deepest encoded node i.e. "pool5" is pulled from tensor array for further processing
            input_tensor = input_tensor_dict[decode_layer_list[0]]['data']

            # Deconvolution is done i.e transpose of conv2d
            score = self.conv2d(inputdata=input_tensor, out_channel=64,
                                kernel_size=1, use_bias=False, name='score_origin')
            # New decoder list is prepared with ['pool4','pool3']
            decode_layer_list = decode_layer_list[1:]
            #Looping over the decoder list to perform decoding
            for i in range(len(decode_layer_list)):
                # Performing the transpose of conv2d on the score obtained till
                deconv = self.deconv2d(inputdata=score, out_channel=64, kernel_size=4,
                                       stride=2, use_bias=False, name='deconv_{:d}'.format(i + 1))
                # Fetching the next decoder index i.e. 'pool3' and using index fetching the tensor data
                input_tensor = input_tensor_dict[decode_layer_list[i]]['data']
                # Convolution is applied on the previously fetched tensor and update the score
                score = self.conv2d(inputdata=input_tensor, out_channel=64,
                                    kernel_size=1, use_bias=False, name='score_{:d}'.format(i + 1))
                # Bothe the score and the deconvolution result is fused  and become the updated score
                fused = tf.add(deconv, score, name='fuse_{:d}'.format(i + 1))
                score = fused
            
            # The result is again deconvolute and saved for binary segmentation
            deconv_final = self.deconv2d(inputdata=score, out_channel=64, kernel_size=16,
                                         stride=8, use_bias=False, name='deconv_final')

            # Convolutional Filter is appplied and the result is saved for instance segmentation
            score_final = self.conv2d(inputdata=deconv_final, out_channel=2,
                                      kernel_size=1, use_bias=False, name='score_final')

            # logits -> Binary operation
            ret['logits'] = score_final
            # deconv -> Instance Segmentation
            ret['deconv'] = deconv_final

        return ret
