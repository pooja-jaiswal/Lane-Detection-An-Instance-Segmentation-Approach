"""
Implement the LaneNet model
"""
import tensorflow as tf

import vgg_encoder
import fcn_decoder
import cnn_basenet
import discriminative_loss


class LaneNet(cnn_basenet.CNNBaseModel):
    """
    Implementing a semantic segmentation model
    """
    def __init__(self, phase, net_flag='vgg'):
        super(LaneNet, self).__init__()
        self._net_flag = net_flag
        self._phase = phase
        if self._net_flag == 'vgg':
            self._encoder = vgg_encoder.VGG16Encoder(phase=phase)
        self._decoder = fcn_decoder.FCNDecoder(phase=phase)
        return

    def __str__(self):
        info = 'Semantic Segmentation use {:s} as basenet to encode'.format(self._net_flag)
        return info

    def _build_model(self, input_tensor, name):
        """
        Forward propagation process
        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # Firstly Encoder Model is formed
            encode_ret = self._encoder.encode(input_tensor=input_tensor,
                                              name='encode')

            # Secondaly Decoder Model is formulated with the resultant model of Encoder
            if self._net_flag.lower() == 'vgg':
                decode_ret = self._decoder.decode(input_tensor_dict=encode_ret,
                                                  name='decode',
                                                  decode_layer_list=['pool5',
                                                                     'pool4',
                                                                     'pool3'])
                return decode_ret

    def compute_loss(self, input_tensor, binary_label, instance_label, name):
        """
        Calculate the loss function of the LaneNet model
        :param input_tensor:
        :param binary_label:
        :param instance_label:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # Forward propagation to get logits
            # ******Logits is a function that maps probabilities (Tensorflow "with logit": It means that you are applying a 
            # softmax function to logit numbers to normalize it. )********
            # Building an Encoder (using VGG16) and Decoder (using Fully Convolutional Network) Module 
            inference_ret = self._build_model(input_tensor=input_tensor, name='inference')
            
            #### Calculate the binary partition loss function ####
            # Fetching the binary results from CNN result array
            decode_logits = inference_ret['logits']

            # Method is reshaping the Tensor for computing Loss
            binary_label_plain = tf.reshape(
                binary_label,
                shape=[binary_label.get_shape().as_list()[0] *
                       binary_label.get_shape().as_list()[1] *
                       binary_label.get_shape().as_list()[2]])
                       
            # Join class weights
            # tf.unique_with_counts will return unique_labels of tensor binary_label_plain,  
            # unique_id of the labels corrosponding to the unique_labels and last o/p is the count of 
            # each element of unique_labels in counts
            unique_labels, unique_id, counts = tf.unique_with_counts(binary_label_plain)
            #Casting count to datatype float32
            counts = tf.cast(counts, tf.float32)

            #Calculating Inverse weight
            inverse_weights = tf.divide(1.0,
                                        tf.log(tf.add(tf.divide(tf.constant(1.0), counts),
                                                      tf.constant(1.02))))

            # Gather slices from inverse_weights according to binary_label and update inverse_weights
            inverse_weights = tf.gather(inverse_weights, binary_label)
            # Calculating the cross entropy loss on the basis of i/p binary_label, calculated decode_logits from CNN 
            # and calculated inverse_weights. Return the binary segmentation loss.
            binary_segmenatation_loss = tf.losses.sparse_softmax_cross_entropy(
                labels=binary_label, logits=decode_logits, weights=inverse_weights)
            # Computes the mean of elements across dimensions of a tensor (dimensions - x/y)
            binary_segmenatation_loss = tf.reduce_mean(binary_segmenatation_loss)

            #### Calculate the Instance partition loss using discriminative loss function ####
            # Calculate the discriminative loss function
            # Fetching the instance image results from CNN result array
            decode_deconv = inference_ret['deconv']
            # Pixel embedding
            #Applying Convolution layer  followed by activation layer ReLU
            pix_embedding = self.conv2d(inputdata=decode_deconv, out_channel=4, kernel_size=1,
                                        use_bias=False, name='pix_embedding_conv')
            pix_embedding = self.relu(inputdata=pix_embedding, name='pix_embedding_relu')
            # Calculate discriminative loss
            # Preparing Image Shape after applying conv and relu activation layer.
            image_shape = (pix_embedding.get_shape().as_list()[1], pix_embedding.get_shape().as_list()[2])
            #Invoking discriminative loss function which returns variance term, distance 
            # term and regularization term
            disc_loss, l_var, l_dist, l_reg = \
                discriminative_loss.discriminative_loss(
                    pix_embedding, instance_label, 4, image_shape, 0.5, 3.0, 1.0, 1.0, 0.001)

            # Consolidation loss
            # Calculating Total Loss  i.e binary seg. loss + disciminative loss and 
            # regularization loss
            l2_reg_loss = tf.constant(0.0, tf.float32)
            for vv in tf.trainable_variables():
                if 'bn' in vv.name:
                    continue
                else:
                    l2_reg_loss = tf.add(l2_reg_loss, tf.nn.l2_loss(vv))
            l2_reg_loss *= 0.001
            total_loss = 0.5 * binary_segmenatation_loss + 0.5 * disc_loss + l2_reg_loss

            #Collecting all parameters as an list and returning to the train_lanenet
            ret = {
                'total_loss': total_loss,
                'binary_seg_logits': decode_logits,
                'instance_seg_logits': pix_embedding,
                'binary_seg_loss': binary_segmenatation_loss,
                'discriminative_loss': disc_loss
            }

            return ret

    def inference(self, input_tensor, name):
        """
        Building model for extracting the features
        :param input_tensor:
        :param name:
        :return:Returns the binary and segmentation features
        """
        with tf.variable_scope(name):
            # Forward propagation to get logits
            inference_ret = self._build_model(input_tensor=input_tensor, name='inference')
            # Calculate the binary partition loss function
            decode_logits = inference_ret['logits']
            binary_seg_ret = tf.nn.softmax(logits=decode_logits)
            binary_seg_ret = tf.argmax(binary_seg_ret, axis=-1)
            # Computing pixel embedding
            decode_deconv = inference_ret['deconv']
            # Pixel embedding
            # Convoluting
            pix_embedding = self.conv2d(inputdata=decode_deconv, out_channel=4, kernel_size=1,
                                        use_bias=False, name='pix_embedding_conv')
            # Activation
            pix_embedding = self.relu(inputdata=pix_embedding, name='pix_embedding_relu')

            # Returning binary and instance tensor.
            return binary_seg_ret, pix_embedding
