import tensorflow as tf, pdb
import tensorflow_addons as tfa

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    assert tf.config.experimental.get_memory_growth(physical_devices[0])
except:
  # Invalid device or cannot modify virtual devices once initialized.
    pass


WEIGHTS_INIT_STDEV = .1

KERNEL_SIZE = 9 # default is 9

def net(image, data_format='NHWC', num_base_channels=32, evaluate=False):

    # transformer net
    conv1 = _conv_layer(image, num_base_channels, KERNEL_SIZE, 1, data_format=data_format, evaluate=evaluate)
    conv2 = _conv_layer(conv1, num_base_channels * 2, 3, 2, data_format=data_format, evaluate=evaluate)
    conv3 = _conv_layer(conv2, num_base_channels * 4, 3, 2, data_format=data_format, evaluate=evaluate)
    
    resid1 = _residual_block(conv3,  num_base_channels * 4, 3, data_format=data_format, evaluate=evaluate)
    resid2 = _residual_block(resid1, num_base_channels * 4, 3, data_format=data_format, evaluate=evaluate)
    resid3 = _residual_block(resid2, num_base_channels * 4, 3, data_format=data_format, evaluate=evaluate)
    resid4 = _residual_block(resid3, num_base_channels * 4, 3, data_format=data_format, evaluate=evaluate)
    resid5 = _residual_block(resid4, num_base_channels * 4, 3, data_format=data_format, evaluate=evaluate)

    # layers used in the original source

    conv_t1 = _conv_tranpose_layer(resid5, 64, 3, 2, data_format=data_format)
    conv_t2 = _conv_tranpose_layer(conv_t1, 32, 3, 2, data_format=data_format)
    conv_t3 = _conv_layer(conv_t2, 3, 9, 1, relu=False, data_format=data_format, evaluate=evaluate)
    preds = tf.nn.tanh(conv_t3) * 150 + 255./2

    '''
    
    # use upsample convolution to reduce checkerboard effects
    #     ref: http://distill.pub/2016/deconv-checkerboard/
    
    up2_1 = _upsample2 (resid5, num_base_channels * 2, kernel_size=3, stride=1, data_format=data_format, evaluate=evaluate)
    up2_2 = _upsample2 (up2_1,  num_base_channels,     kernel_size=3, stride=1, data_format=data_format, evaluate=evaluate)
    preds = _conv_layer(up2_2,  3, KERNEL_SIZE, 1, instanceNorm=False, relu=True, data_format=data_format, evaluate=evaluate)  #mcky, use relu to avoid negative values
   

    # use depthwise conv2d
    up2_1 = _upsample2_conv2d_depthwise (resid5, num_base_channels * 2, kernel_size=3, stride=1, data_format=data_format, evaluate=evaluate)
    up2_2 = _upsample2_conv2d_depthwise (up2_1,  num_base_channels,     kernel_size=3, stride=1, data_format=data_format, evaluate=evaluate)
    preds = _conv_depthwise_layer       (up2_2,  3, KERNEL_SIZE, 1, instanceNorm=False, relu=True, data_format=data_format, evaluate=evaluate)  #mcky, use relu to avoid negative values
    '''
    #mcky 2018/12/25, this is to workaround a bug in tf2onnx (currently v0.3.2 in pip and v0.4.0 in master)
    #        The bug does not export the last node correctly (the output node does not have 'shape' attribute)
    #        The output node name is 'add_37' (observed from tensorflow 'summarize_graph' tool
    #        Adding an Identity node to receive the output from 'add_37'as last node would workaround this tf2onnx bug.
    #        Specify 'dummy_output' in tf2onnx as it's the desired output node. (Step 2 from above)
    #
    #        this should be removed during training
    #
    #preds = tf.identity(preds, "dummy_output")
    #print("tf.identity: {}".format(preds))
    
    return  preds

def _conv_layer(net, num_channels, filter_size, strides, instanceNorm=True, relu=True, data_format='NHWC', evaluate=False):
    if data_format == 'NHWC':
        weights_init = _conv_init_vars(net, num_channels, filter_size, data_format=data_format)
        strides_shape = [1, strides, strides, 1]

        # padding to avoid 'if' checking in TFlite GPU delegate conv2D
        print ("net.shape:{}".format(net.shape))
        p = tf.constant([ [0, 0], [filter_size//2,filter_size//2], [filter_size//2,filter_size//2], [0, 0] ])
        print ("p.shape:{}".format(p.shape))
        net = tf.pad(tensor=net, paddings=p, mode="CONSTANT")
        print ("net.shape:{}".format(net.shape))

        net = tf.nn.conv2d(input=net, filters=weights_init, strides=strides_shape, padding='VALID',
            data_format=data_format
            )
        if instanceNorm:
            if evaluate:
                #net = _instance_norm_tflite_nnapi(net)
                net = _instance_norm_tflite_gpudelegate(net)  # avoid this for training.  it requires square images.
            else:
                net = _instance_norm (net)  # use this for training
    else:
        weights_init = _conv_init_vars(net, num_channels, filter_size, data_format=data_format)
        strides_shape = [1, 1, strides, strides]

        print ("net.shape:{}".format(net.shape))
        p = tf.constant([ [0, 0], [0, 0], [filter_size//2,filter_size//2], [filter_size//2,filter_size//2] ])
        print ("p.shape:{}".format(p.shape))
        net = tf.pad(tensor=net, paddings=p, mode="CONSTANT")
        print ("net.shape:{}".format(net.shape))

        net = tf.nn.conv2d(input=net, filters=weights_init, strides=strides_shape, padding='VALID',
            data_format=data_format
            )

        if instanceNorm:
            net = _instance_norm_nchw(net)
            # net = tf.layers.batch_normalization(net, axis=1, momentum=0.99, epsilon=0.001)

    if relu:
        net = tf.nn.relu(net)

    return net

#mcky, reference https://stackoverflow.com/questions/37092037/tensorflow-what-does-tf-nn-separable-conv2d-do
def _conv_depthwise_layer(net, num_channels, filter_size, strides, instanceNorm=True, relu=True, data_format='NHWC', evaluate=False):
    channel_multiplier = 1

    if data_format == 'NHWC':
        p = tf.constant([ [0, 0], [filter_size//2,filter_size//2], [filter_size//2,filter_size//2], [0, 0] ])
        print ("depthwise == net.shape: {}".format(net.shape))
        net = tf.pad(tensor=net, paddings=p, mode="CONSTANT")

        print ("net.shape: {}".format(net.shape))
        
        _, rows, cols, in_channels = [i for i in net.get_shape()]
        out_channels = num_channels
        
        depthwise_channels_shape = [filter_size, filter_size, in_channels, channel_multiplier]
        pointwise_channels_shape = [1, 1, in_channels*channel_multiplier, out_channels]

        depthwise_channels = tf.Variable(tf.random.truncated_normal(depthwise_channels_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)
        pointwise_channels = tf.Variable(tf.random.truncated_normal(pointwise_channels_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)

        #strides_shape = [1, strides, strides, 1]
        strides_shape = [1, 1, 1, 1]
        net = tf.nn.separable_conv2d(
                        input=net,
                        depthwise_filter=depthwise_channels,
                        pointwise_filter=pointwise_channels,
                        strides=strides_shape,
                        padding='VALID',
                        )
        
        print ("conv_depthwise.shape: {}".format(net.shape))

        '''
        print ("net.shape: {}".format(net.shape))
        weights_init = _conv_init_vars(net, num_channels, filter_size, data_format=data_format)
        strides_shape = [1, strides, strides, 1]

        # padding to avoid 'if' checking in TFlite GPU delegate conv2D
        print ("net.shape:{}".format(net.shape))
        p = tf.constant([ [0, 0], [filter_size//2,filter_size//2], [filter_size//2,filter_size//2], [0, 0] ])
        print ("p.shape:{}".format(p.shape))
        net = tf.pad(net, p, "CONSTANT")
        print ("net.shape:{}".format(net.shape))

        net = tf.nn.conv2d(net, weights_init, strides_shape, padding='VALID',
            data_format=data_format
            )
        print ("conv2d.shape: {}".format(net.shape))
        '''

        if instanceNorm:
            if evaluate:
                #net = _instance_norm_tflite_nnapi(net)
                net = _instance_norm_tflite_gpudelegate(net)  # avoid this for training.  it requires square images.
            else:
                net = _instance_norm (net)  # use this for training
        
    else:
        '''
        weights_init = _conv_init_vars(net, num_channels, filter_size, data_format=data_format)
        strides_shape = [1, 1, strides, strides]
        net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME',
            data_format=data_format
            )
        '''

        p = tf.constant([ [0, 0], [0, 0], [filter_size//2,filter_size//2], [filter_size//2,filter_size//2] ])
        print ("depthwise == net.shape: {}".format(net.shape))
        net = tf.pad(tensor=net, paddings=p, mode="CONSTANT")

        print ("depthwise == net.shape: {}".format(net.shape))
        
        _, in_channels, rows, cols = [i for i in net.get_shape()]
        out_channels = num_channels
        
        depthwise_channels_shape = [filter_size, filter_size, in_channels, channel_multiplier]
        pointwise_channels_shape = [1, 1, in_channels*channel_multiplier, out_channels]

        depthwise_channels = tf.Variable(tf.random.truncated_normal(depthwise_channels_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)
        pointwise_channels = tf.Variable(tf.random.truncated_normal(pointwise_channels_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)

        #strides_shape = [1, strides, strides, 1]
        strides_shape = [1, 1, 1, 1]
        net = tf.nn.separable_conv2d(
                        input=net,
                        depthwise_filter=depthwise_channels,
                        pointwise_filter=pointwise_channels,
                        strides=strides_shape,
                        data_format=data_format,
                        padding='VALID',
                        )
        
        print ("conv_depthwise.shape: {}".format(net.shape))

        if instanceNorm:
            net = _instance_norm_nchw(net)
        

    if relu:
        net = tf.nn.relu(net)

    return net

# upsample_by_2() - upsample the input to twice the size in h & w using only reshape and concat.
#                   supports both NCHW and NHWC data format.
#
# tensorflow resize methods only works for NHWC.  calling upsample_by_2() for NCHW data format
#
'''
# 2nd input adds epsilon to avoid 'sample output for multiple input' issue.
def upsample_by_2 (x, c, h, w): # for nhwc data format
    c_eps1 = tf.constant ([1e-9])
    c_eps2 = tf.constant ([1e-9])

    bb  = tf.reshape(x,[-1,c])
    cc  = tf.concat([bb,bb+c_eps1],1)
    cc1 = tf.reshape(cc,[-1,w*2*c])
    cc2 = tf.concat([cc1,cc1+c_eps2],1)
    
    out = tf.reshape(cc2,[-1,h*2,w*2,c])

    return out
'''
'''
tflite gpu error:
    ```
    RESIZE_BILINEAR: Expected 1 input tensor(s), but node has 0 runtime input(s).
    First 163 operations will run on the GPU, and the remaining 49 on the CPU.TfLiteGpuDelegate Prepare: Only identical batch dimension is supportedNode number 212 (TfLiteGpuDelegate) failed to prepare.
    ```
def upsample_by_2 (x, c, h, w): # for nhwc data format
    c_eps1 = tf.constant ([1e-9], shape=[1,1,1,1])
    c_eps2 = tf.constant ([1e-9], shape=[1,1,1,1])

    # concat #1
    bb  = tf.reshape(x,[-1,c])

    bb1 = tf.reshape(x,[1,-1,c,1])
    #c_eps1 = tf.constant (1e-9, shape=bb.shape)  # to avoid tflite gpu "Only identical batch dimension is supported"
    c_eps_a = tf.image.resize_bilinear(c_eps1, bb.shape)   # to avoid tflite gpu "Only identical batch dimension is supported"
    bb2 = bb1 + c_eps_a
    bb3 = tf.reshape(bb2,[-1,c])

    cc  = tf.concat([bb,bb3],1)     # tflite gpu_delegate does not take same output for 2 inputs

    # concat #2
    cc1 = tf.reshape(cc,[-1,w*2*c])

    #c_eps2 = tf.constant (1e-9, shape=cc1.shape)
    cc11 = tf.reshape(cc,[1,-1,w*2*c,1])
    c_eps_b = tf.image.resize_bilinear(c_eps2, cc1.shape)   # to avoid tflite gpu "Only identical batch dimension is supported"
    cc2 = cc11 + c_eps_b
    cc3 = tf.reshape(cc2,[-1,w*2*c])
    
    cc4 = tf.concat([cc1,cc3],1)   # tflite gpu_delegate does not take same output for 2 inputs
    
    out = tf.reshape(cc4,[-1,h*2,w*2,c])

    return out
'''

'''
tflite gpu error:
    ```
    error: "java.lang.IllegalArgumentException: Internal error: Failed to apply delegate: TfLiteGpuDelegate Prepare: Dimension can not be reduced to linear."
    ```

def upsample_by_2 (x, c, h, w): # for nhwc data format
    #c_eps1 = tf.constant ([1e-9], shape=[1,1,1,1])
    #c_eps2 = tf.constant ([1e-9], shape=[1,1,1,1])

    # concat #1
    bb  = tf.reshape(x,[-1,c])
    c_eps1 = tf.constant (1e-9, shape=bb.shape)  # to avoid tflite gpu "Only identical batch dimension is supported"
    bb1 = bb + c_eps1
    bb2  = tf.concat([bb,bb1],1)     # tflite gpu_delegate does not take same output for 2 inputs

    # concat #2
    cc = tf.reshape(bb2,[-1,w*2*c])
    c_eps2 = tf.constant (1e-9, shape=cc.shape)
    cc1 = cc + c_eps2
    cc2 = tf.concat([cc,cc1],1)   # tflite gpu_delegate does not take same output for 2 inputs
    
    out = tf.reshape(cc2,[-1,h*2,w*2,c])

    return out
'''

def upsample_by_2 (x, c, h, w): # for nhwc data format
    #c_eps1 = tf.constant ([1e-9], shape=[1,1,1,1])
    #c_eps2 = tf.constant ([1e-9], shape=[1,1,1,1])

    #c_one_1 = tf.constant ([1])
    #c_one_2 = tf.constant ([1])

    # concat #1
    bb  = tf.reshape(x,[(h*w),c])
    bb1 = tf.reshape(x,[(h*w),c])
    bb2  = tf.concat([bb,bb1],1)     # tflite gpu_delegate does not take same output for 2 inputs

    # concat #2
    cc  = tf.reshape(bb2,[h,w*2*c])
    cc1 = tf.reshape(bb2,[h,w*2*c])
    cc2 = tf.concat([cc,cc1],1)   # tflite gpu_delegate does not take same output for 2 inputs
    
    out = tf.reshape(cc2,[1,h*2,w*2,c])

    return out

def upsample_by_2_nchw (x, c, h, w): # for nchw data format
    bb  = tf.reshape(x,[-1,1])
    cc  = tf.concat([bb,bb],1)
    cc1 = tf.reshape(cc,[-1,w*2])
    cc2 = tf.concat([cc1,cc1],1)
    
    out = tf.reshape(cc2,[-1,c,h*2,w*2])

    return out

def _upsample2(net, out_channels, kernel_size, stride, data_format='NHWC', evaluate=False):
    if data_format == 'NHWC':
        c = net.shape[3]
        h = net.shape[1]
        w = net.shape[2]
        
        #print ('evaluate: %s'% evaluate)
        if evaluate :
            net = tf.image.resize (net,[h*2,w*2], method=tf.image.ResizeMethod.BILINEAR)
        else :
            net = tf.image.resize (net,[h*2,w*2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) # tflite nnapi does not support 'resize_nearest'
                                                                # nor 'resize bilinear', unlike what's listed here: https://developer.android.com/ndk/reference/group/neural-networks
        #net = tf.image.resize_bilinear (net,[w*2,h*2])
        #net = upsample_by_2(net, c, h, w)
    else:
        c = net.shape[1]
        h = net.shape[2]
        w = net.shape[3]
        
        net = upsample_by_2_nchw(net, c, h, w)
        
    net = _conv_layer(net, out_channels, kernel_size, stride, data_format=data_format, evaluate=evaluate)

    return net

def _upsample2_conv2d_depthwise(net, out_channels, kernel_size, stride, data_format='NHWC', evaluate=False):
    if data_format == 'NHWC':
        c = net.shape[3]
        h = net.shape[1]
        w = net.shape[2]
        
        if evaluate:
            #net = tf.image.resize_nearest_neighbor (net,[w*2,h*2]) # tflite nnapi does not support 'resize_nearest'
                                                                    # nor 'resize bilinear', unlike what's listed here: https://developer.android.com/ndk/reference/group/neural-networks
            net = tf.image.resize (net,[h*2,w*2], method=tf.image.ResizeMethod.BILINEAR)  # this is only for running inference on TFLite GPU Delegate
        else:
            net = upsample_by_2(net, c, h, w)  # use this for training
    else:
        c = net.shape[1]
        h = net.shape[2]
        w = net.shape[3]
        
        net = upsample_by_2_nchw(net, c, h, w)
        
    #net = _conv_layer(net, out_channels, kernel_size, stride, data_format=data_format)
    net = _conv_depthwise_layer(net, out_channels, kernel_size, stride, data_format=data_format, evaluate=evaluate)

    return net

def _conv_tranpose_layer(net, num_channels, filter_size, strides, data_format='NHWC'):
    weights_init = _conv_init_vars(net, num_channels, filter_size, transpose=True, data_format=data_format)

    if data_format == 'NHWC':
        batch_size, rows, cols, in_channels = [i for i in net.get_shape()]
        new_rows, new_cols = int(rows * strides), int(cols * strides)
        new_shape = [batch_size, new_rows, new_cols, num_channels]
        tf_shape = tf.stack(new_shape)
        strides_shape = [1,strides,strides,1]
    
        net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME',
                                        data_format=data_format)
        #net = _instance_norm(net)
        #net = _instance_norm_tflite_nnapi(net)
        net = _instance_norm_tflite_gpudelegate(net)
    else:
        batch_size, in_channels, rows, cols = [i for i in net.get_shape()]
        new_rows, new_cols = int(rows * strides), int(cols * strides)
        new_shape = [batch_size, num_channels, new_rows, new_cols]
        tf_shape = tf.stack(new_shape)
        strides_shape = [1,1,strides,strides]
    
        net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME',
                                        data_format=data_format)
        net = _instance_norm_nchw(net)
        # net = tf.layers.batch_normalization(net, axis=1, momentum=0.99, epsilon=0.001)

    return tf.nn.relu(net)

def _residual_block(net, num_channels = 128, filter_size=3, data_format='NHWC', evaluate=False):
    tmp = _conv_layer(net, num_channels, filter_size, 1, data_format=data_format, evaluate=evaluate)
    return net + _conv_layer(tmp, num_channels, filter_size, 1, relu=False, data_format=data_format, evaluate=evaluate)

# use depthwise conv2d
def _residual_block_conv2d_depthwise(net, num_channels = 128, filter_size=3, data_format='NHWC', evaluate=False):
    tmp = _conv_depthwise_layer(net, num_channels, filter_size, 1, data_format=data_format, evaluate=evaluate)
    return net + _conv_depthwise_layer(tmp, num_channels, filter_size, 1, relu=False, data_format=data_format, evaluate=evaluate)

# InstanceNorm for NHWC
def _instance_norm(net):
    batch, rows, cols, channels = [i for i in net.get_shape()]

    mu, sigma_sq = tf.nn.moments(x=net, axes=[1,2], keepdims=True)

    epsilon = 1e-9 # 1e-3 originally.  set to 1e-9 to avoid a conversion issue for Intel OpenVINO model optimizer
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)

    var_shape = [channels]
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    net = scale * normalized + shift

    return net


# square root estimator using "Babylonian method"
#   ref: https://en.wikipedia.org/wiki/Methods_of_computing_square_roots
def _sqrt (net):
    '''
    accuracy = 6
    sqrt_est = accuracy * 100
    num      = net*1000000 #make it a 6-digit figure
    '''
    accuracy      = 6 # <-- must be even number
    
    # all constants must be wrapped in tf.constant to avoid tflite conversion 'rank 0' error
    sqrt_est = tf.constant([accuracy * 100.])
    #sqrt_est1= tf.constant([accuracy * 100. + 1e-9])
    half     = tf.constant([0.5])
    one6     = tf.constant([10.**accuracy])
    one3     = tf.constant([10.**(accuracy/2)])

    num      = net * one6 # make it a 6-digit figure

    for i in range(accuracy):
        half = half# + tf.constant([1e-9])
        sqrt_est = half * (sqrt_est + num/sqrt_est)
        #sqrt_est = half * (sqrt_est + tf.div(num, sqrt_est))

    sqrt_est = sqrt_est / one3
    #sqrt_est = tf.div(sqrt_est, one3)

    return sqrt_est

# InstanceNorm for TFLite nnapi - NHWC
#   nnapi does not have 'reduced_mean', squred_difference' and 'sqrt' ops.
def _instance_norm_tflite_nnapi(net, train=True):
    batch, rows, cols, channels = [i for i in net.get_shape()]

    mu = tf.math.reduce_mean(input_tensor=net, axis=[1,2], keepdims=True)
    diff     = net-mu
    
    #sqr_diff = (diff*diff)  # nnapi does not support 'sqare' op
    #sqr_diff = tf.math.multiply (diff, diff)
    epsilon     = 1e-9
    c_eps       = tf.constant([epsilon]) # all constants must be wrapped in tf.constant to avoid tflite conversion 'rank 0' error

    diff1 = diff + c_eps
    sqr_diff = diff*diff1

    sigma_sq = tf.math.reduce_mean(input_tensor=sqr_diff, axis=[1,2], keepdims=True)
    print ("sigma_sq.shape:{}".format(sigma_sq.shape))

    c_eps       = tf.constant([epsilon]) # all constants must be wrapped in tf.constant to avoid tflite conversion 'rank 0' error
    normalized  = (net-mu)/_sqrt(sigma_sq + c_eps) # tflite nnapi does not support sqrt.  use sqrt estimation instead.
    #normalized = tf.div( (net-mu), _sqrt(sigma_sq + c_eps) )

    var_shape = [channels]
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    net = scale * normalized + shift

    return net

# InstanceNorm for TFLite GPUDelegate - NHWC
#   GPUDelegate adds new ops, "Implement DIV, POW and SQUARED_DIFF operations with two inputs."
#   reference https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/gpu/common/operations.cc
'''
# this version is good for tflite gpu_delegate 20190411, but slow due to many resize_bilinear ops used for working around issues.
# LG G4:
#        cpu: ~2520ms      gpu: ~635ms
#
def _instance_norm_tflite_gpudelegate(net, train=True):
    batch, rows, cols, channels = [i for i in net.get_shape()]

    epsilon = 1e-9
    c_eps   = tf.constant([epsilon]) # all constants must be wrapped in tf.constant to avoid tflite conversion 'rank 0' error
    c_eps2  = tf.constant([epsilon]) # all constants must be wrapped in tf.constant to avoid tflite conversion 'rank 0' error
    
    net1  = net# + c_eps
    net2  = net# + c_eps2

    mu = tf.nn.pool (net1, [rows,cols], 'AVG', 'VALID', strides=[2,2])
    
    #c_negone = tf.constant([-1.0])
    #mu1 = c_negone * mu      # !!! this would result in run-time error.  constant has to be in 2nd input !!!
    #mu1 = mu * c_negone

    #diff = net2 - mu   # !!! GPU 'Sub' slows down to about the same speed as CPU !!!
    #diff = net + mu1
    #diff = net2 + mu

    negone = [-1.0 for i in range (channels)]
    c_neg1 = tf.constant([negone])
    c_negone = tf.reshape(c_neg1, [1,1,1,channels]) # 'mul' in gpu_delegate needs both inputs to have same shape
    mu1 = mu * c_negone
    #mu1 = tf.reshape (mu1,[mu1.shape[3]]) # reduce to rank 1 to avoid run-time error "TfLiteGpuDelegate Prepare: Error while rewriting '$input_data_1[gid.z]$"
    mu1 = tf.image.resize_bilinear(mu1, [rows,cols])
    diff = net2 + mu1  # needs to use 'add' because 'sub' has a bug that slows down GPU *a lot*...

    #sqr_diff = tf.math.squared_difference(net, mu)
    #sqr_diff = diff*diff
    sqr_diff = tf.math.square(diff)
    #sqr_diff = diff

    #sigma_sq = avg_pool2d(sqr_diff)
    #sigma_sq = avg_pool2d(diff) #test to skip square
    sigma_sq = tf.nn.pool (sqr_diff, [rows,cols], 'AVG', 'VALID', strides=[2,2])
    #print ("sigma_sq.shape:{}".format(sigma_sq.shape))
    
    c_eps       = tf.constant([epsilon]) # all constants must be wrapped in tf.constant to avoid tflite conversion 'rank 0' error
    #c_half      = tf.constant([0.5])

    #normalized  = (net-mu)/(sigma_sq + c_eps)**(c_half)
    #c_one  = tf.constant([1.0])
    #inv    = c_one / (sigma_sq + c_eps)**(c_half)
    inv = tf.math.rsqrt (sigma_sq + c_eps)
    #normalized = (net-mu)*inv
    
    #inv = tf.image.resize_bilinear()
    #normalized = diff * inv # !!! this is not supported by tflite gpu_delegate as it does not do multi-channel broadcast with single HW dimension (i.e. 1x1x1x16) !!!
    
    #normalized = diff / tf.math.sqrt (sigma_sq + c_eps)
    #normalized = diff + sigma_sq
    #normalized = diff * sigma_sq  # !!! this is not supported by tflite gpu_delegate as it does not do multi-channel broadcast with single HW dimension (i.e. 1x1x1x16) !!!
    #normalized = diff * inv   # test out rsqrt performance
    
    #inv1 = tf.reshape(inv, [channels]) # <-- gpu_delegate still complains.  error: "TfLiteGpuDelegate Prepare: Only identical batch dimension is supported"
    inv1 = tf.image.resize_bilinear (inv, [rows,cols])  # gpu_delegate 'mul' only accepts same h & w shape for both inputs.
    normalized = diff * inv1   # test out rsqrt performance

    var_shape = [channels]
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    #net = scale * normalized + shift
    #net = normalized * scale + shift
    #net = net + mu + shift # fast gpu in tflite
    #net = net - mu
    #net1 = net1 + mu
    net = normalized * scale + shift
    #net = normalized

    return net
'''

# mcky - notes about tflite gpu delegate (for 2019/04/11 version)
# 1. 'sub' is very slow for some reason.  doing all changes here to avoid 'sub'.
#     - this includes 'sub' in other ops, such as 'sqaured_difference'.
# 2. 'mul' -
#     - needs both inputs to have same shape.
#     - broadcast tensor only supports variables inputs and needs to be 2nd input.
# 3. 'add' -
#     - input with rank 1 (i.e. input shape [16]) results in error "TfLiteGpuDelegate Prepare: Only identical batch dimension is supported"
#     - when broadcasting, results in error "TfLiteGpuDelegate Prepare: Error while rewriting '$input_data_1[gid.z]$'"
# 4. 'reshape' - 
#     - if 2 dims, both dim needs to be the same.  Or would have error "TfLiteGpuDelegate Prepare: Only identical batch dimension is supported"

def _instance_norm_tflite_gpudelegate(net, train=True):
    batch, rows, cols, channels = [i for i in net.get_shape()]

    epsilon = 1e-3
    c_eps   = tf.constant([epsilon]) # all constants must be wrapped in tf.constant to avoid tflite conversion 'rank 0' error
    c_eps2  = tf.constant([epsilon]) # all constants must be wrapped in tf.constant to avoid tflite conversion 'rank 0' error
    
    net1  = net# + c_eps
    net2  = net# + c_eps2

    #mu = tf.nn.pool (net1, [rows,cols], 'AVG', 'VALID', strides=[2,2])

    #mcky, use 2-stage pooling for parallelisation, target specifically for 512x512
    
    '''
    print ("net1.shape: {}".format(net1.shape))
    mu = tf.nn.pool (net1, [4,4], 'AVG', 'SAME', strides=[4,4])   # 1st stage
    print ("mu.shape: {}".format(mu.shape))
    mu = tf.nn.pool (mu, [4,4], 'AVG', 'SAME', strides=[4,4])     # 2nd stage
    print ("mu.shape: {}".format(mu.shape))
    mu = tf.nn.pool (mu, [4,4], 'AVG', 'SAME', strides=[4,4])     # 3rd stage
    print ("mu.shape: {}".format(mu.shape))

    if (cols == 512) :
        mu = tf.nn.pool (mu, [8,8], 'AVG', 'SAME', strides=[8,8])     # 4th stage
    elif(cols == 256) :
        mu = tf.nn.pool (mu, [4,4], 'AVG', 'SAME', strides=[4,4])     # 4th stage
    elif(cols == 128) :
        mu = tf.nn.pool (mu, [2,2], 'AVG', 'SAME', strides=[2,2])     # 4th stage

    print ("mu.shape: {}".format(mu.shape))
    '''
    #clc, support 512x512, 480p,720p and 1080p
    print ("net1.shape: {}".format(net1.shape))
    if (cols in {512, 256, 128}) :  # specific for 512x512
        mu = tf.nn.pool (input=net1, window_shape=[4,4], pooling_type='AVG', padding='SAME', strides=[4,4])   # 1st stage
    else: 
        mu = tf.nn.pool (input=net1, window_shape=[3,4], pooling_type='AVG', padding='SAME', strides=[3,4])   # 1st stage
    print ("mu.shape: {}".format(mu.shape))
    if (cols in {640, 320, 160, 512, 256, 128}) :  #specific for 640x480 & 512x512
        mu = tf.nn.pool (input=mu, window_shape=[4,4], pooling_type='AVG', padding='SAME', strides=[4,4])     # 2nd stage
    else :
        mu = tf.nn.pool (input=mu, window_shape=[3,4], pooling_type='AVG', padding='SAME', strides=[3,4])     # 2nd stage
    print ("mu.shape: {}".format(mu.shape))
    if (cols in {512, 256, 128}) :  # specific for 512x512
        mu = tf.nn.pool (input=mu, window_shape=[4,4], pooling_type='AVG', padding='SAME', strides=[4,4])   # 3rd stage
    else : 
        mu = tf.nn.pool (input=mu, window_shape=[5,5], pooling_type='AVG', padding='SAME', strides=[5,5])     # 3rd stage
    print ("mu.shape: {}".format(mu.shape))

    if (cols == 1920) :
        mu = tf.nn.pool (input=mu, window_shape=[24,24], pooling_type='AVG', padding='SAME', strides=[24,24])     # 4th stage
    elif(cols == 1280) :
        mu = tf.nn.pool (input=mu, window_shape=[16,16], pooling_type='AVG', padding='SAME', strides=[16,16])     # 4th stage
    elif(cols == 960) :
        mu = tf.nn.pool (input=mu, window_shape=[12,12], pooling_type='AVG', padding='SAME', strides=[12,12])     # 4th stage
    elif(cols in {640, 512}) :
        mu = tf.nn.pool (input=mu, window_shape=[8,8], pooling_type='AVG', padding='SAME', strides=[8,8])     # 4th stage
    elif(cols == 480) :
        mu = tf.nn.pool (input=mu, window_shape=[6,6], pooling_type='AVG', padding='SAME', strides=[6,6])     # 4th stage
    elif(cols in {320, 256}) :
        mu = tf.nn.pool (input=mu, window_shape=[4,4], pooling_type='AVG', padding='SAME', strides=[4,4])     # 4th stage
    elif(cols in {160, 128}) :
        mu = tf.nn.pool (input=mu, window_shape=[2,2], pooling_type='AVG', padding='SAME', strides=[2,2])     # 4th stage

    print ("mu.shape: {}".format(mu.shape))

    '''
    # only 512 uses avg_pool 3-stage.
    print ("net1.shape: {}".format(net1.shape))
    mu = tf.nn.pool (net1, [16,16], 'AVG', 'SAME', strides=[16,16])     # 1st stage
    
    # 3rd stage
    if (cols == 512) :
        mu = tf.nn.pool (mu, [16,16], 'AVG', 'SAME', strides=[16,16])     # 2nd stage
        print ("mu.shape: {}".format(mu.shape))
        mu = tf.nn.pool (mu, [2,2], 'AVG', 'SAME', strides=[2,2])         # 3rd stage 
    elif(cols == 256) :
        mu = tf.nn.pool (mu, [16,16], 'AVG', 'SAME', strides=[16,16])     # 2nd stage
    elif(cols == 128) :
        mu = tf.nn.pool (mu, [8,8], 'AVG', 'SAME', strides=[8,8])         # 2nd stage
    
    print ("mu.shape: {}".format(mu.shape))
    '''


    
    #mcky, use 3-stage pooling for parallelisation, target specifically for 1024x1024
    #                 1024 x 1024
    # stage 1 result - 128 x  128
    # stage 2 result -  16 x   16
    # stage 3 result -   1 x    1

    '''
    print ("net1.shape: {}".format(net1.shape))
    mu = tf.nn.pool (net1, [8,8], 'AVG', 'SAME', strides=[8,8])     # 1st stage
    print ("mu.shape: {}".format(mu.shape))

    mu = tf.nn.pool (mu,   [8,8], 'AVG', 'SAME', strides=[8,8])     # 2nd stage
    print ("mu.shape: {}".format(mu.shape))
    '''
    '''
    if (cols == 1024) :
        mu = tf.nn.pool (mu,   [16,16], 'AVG', 'SAME', strides=[16,16])     # 3rd stage
    elif (cols == 512) :
        mu = tf.nn.pool (mu,   [8,8], 'AVG', 'SAME', strides=[8,8])     # 3rd stage
    elif (cols == 256) :
        mu = tf.nn.pool (mu,   [4,4], 'AVG', 'SAME', strides=[4,4])     # 3rd stage
    '''
    '''
    if (cols == 768) :
        mu = tf.nn.pool (mu,   [12,12], 'AVG', 'SAME', strides=[12,12])     # 3rd stage
    elif (cols == 384) :
        mu = tf.nn.pool (mu,   [6,6], 'AVG', 'SAME', strides=[6,6])     # 3rd stage
    elif (cols == 192) :
        mu = tf.nn.pool (mu,   [3,3], 'AVG', 'SAME', strides=[3,3])     # 3rd stage
    '''
    '''
    if (cols == 512) :
        mu = tf.nn.pool (mu,   [8,8], 'AVG', 'SAME', strides=[8,8])     # 3rd stage
    elif (cols == 256) :
        mu = tf.nn.pool (mu,   [4,4], 'AVG', 'SAME', strides=[4,4])     # 3rd stage
    elif (cols == 128) :
        mu = tf.nn.pool (mu,   [2,2], 'AVG', 'SAME', strides=[2,2])     # 3rd stage
    '''

    #print ("mu3.shape: {}".format(mu.shape))
    
    negone = [-1.0 for i in range (channels)]
    c_neg1 = tf.constant([negone])
    c_negone = tf.reshape(c_neg1, [1,1,1,channels]) # 'mul' in gpu_delegate needs both inputs to have same shape
    mu1 = mu * c_negone
    
    #mu1 = tf.image.resize_bilinear(mu1, [rows,cols]) # gpu_delegate 'mul' only accepts same h & w shape for both inputs.
    diff = net2 + mu1  # needs to use 'add' because 'sub' has a bug that slows down GPU *a lot*...

    sqr_diff = tf.math.square(diff)

    #sigma_sq = tf.nn.pool (sqr_diff, [rows,cols], 'AVG', 'VALID', strides=[2,2])
    
    # 4-stage avg_pool
    '''
    sigma_sq = tf.nn.pool (sqr_diff, [4,4], 'AVG', 'SAME', strides=[4,4])   # 1st stage
    sigma_sq = tf.nn.pool (sigma_sq, [4,4], 'AVG', 'SAME', strides=[4,4])   # 2nd stage
    sigma_sq = tf.nn.pool (sigma_sq, [4,4], 'AVG', 'SAME', strides=[4,4])   # 3rd stage

    if (cols == 512) :
        sigma_sq = tf.nn.pool (sigma_sq, [8,8], 'AVG', 'SAME', strides=[8,8])     # 4th stage
    elif(cols == 256) :
        sigma_sq = tf.nn.pool (sigma_sq, [4,4], 'AVG', 'SAME', strides=[4,4])     # 4th stage
    elif(cols == 128) :
        sigma_sq = tf.nn.pool (sigma_sq, [2,2], 'AVG', 'SAME', strides=[2,2])     # 4th stage
    '''
    if (cols in {512, 256, 128}) :  # specific for 512x512
        sigma_sq = tf.nn.pool (input=sqr_diff, window_shape=[4,4], pooling_type='AVG', padding='SAME', strides=[4,4])   # 1st stage
    else:
        sigma_sq = tf.nn.pool (input=sqr_diff, window_shape=[3,4], pooling_type='AVG', padding='SAME', strides=[3,4])   # 1st stage
    if (cols in {640, 320, 160, 512, 256, 128}) :  #specific for 640x480
        sigma_sq = tf.nn.pool (input=sigma_sq, window_shape=[4,4], pooling_type='AVG', padding='SAME', strides=[4,4])   # 2nd stage
    else :
        sigma_sq = tf.nn.pool (input=sigma_sq, window_shape=[3,4], pooling_type='AVG', padding='SAME', strides=[3,4])   # 2nd stage
    if (cols in {512, 256, 128}) :  # specific for 512x512
        sigma_sq = tf.nn.pool (input=sigma_sq, window_shape=[4,4], pooling_type='AVG', padding='SAME', strides=[4,4])   # 3rd stage
    else :
        sigma_sq = tf.nn.pool (input=sigma_sq, window_shape=[5,5], pooling_type='AVG', padding='SAME', strides=[5,5])   # 3rd stage

    if (cols == 1920) :
        sigma_sq = tf.nn.pool (input=sigma_sq, window_shape=[24,24], pooling_type='AVG', padding='SAME', strides=[24,24])     # 4th stage
    elif(cols == 1280) :
        sigma_sq = tf.nn.pool (input=sigma_sq, window_shape=[16,16], pooling_type='AVG', padding='SAME', strides=[16,16])     # 4th stage
    elif(cols == 960) :
        sigma_sq = tf.nn.pool (input=sigma_sq, window_shape=[12,12], pooling_type='AVG', padding='SAME', strides=[12,12])     # 4th stage
    elif(cols in {640, 512}) :
        sigma_sq = tf.nn.pool (input=sigma_sq, window_shape=[8,8], pooling_type='AVG', padding='SAME', strides=[8,8])     # 4th stage
    elif(cols == 480) :
        sigma_sq = tf.nn.pool (input=sigma_sq, window_shape=[6,6], pooling_type='AVG', padding='SAME', strides=[6,6])     # 4th stage
    elif(cols in {320, 256}) :
        sigma_sq = tf.nn.pool (input=sigma_sq, window_shape=[4,4], pooling_type='AVG', padding='SAME', strides=[4,4])     # 4th stage
    elif(cols in {160, 128}) :
        sigma_sq = tf.nn.pool (input=sigma_sq, window_shape=[2,2], pooling_type='AVG', padding='SAME', strides=[2,2])     # 4th stage
    '''
    # only 512 uses avg_pool 3-stage.
    sigma_sq = tf.nn.pool (sqr_diff, [16,16], 'AVG', 'SAME', strides=[16,16])     # 1st stage

    if (cols == 512) :
        sigma_sq = tf.nn.pool (sigma_sq, [16,16], 'AVG', 'SAME', strides=[16,16])   # 2nd stage
        sigma_sq = tf.nn.pool (sigma_sq, [ 2, 2], 'AVG', 'SAME', strides=[ 2, 2])   # 3rd stage
    elif(cols == 256) :
        sigma_sq = tf.nn.pool (sigma_sq, [16,16], 'AVG', 'SAME', strides=[16,16])   # 2nd stage
    elif(cols == 128) :
        sigma_sq = tf.nn.pool (sigma_sq, [8,8], 'AVG', 'SAME', strides=[8,8])       # 2nd stage
    '''

    #mcky, use 3-stage pooling for parallelisation, target specifically for 1024x1024
    '''
    sigma_sq = tf.nn.pool (sqr_diff, [8,8],   'AVG', 'SAME', strides=[8,8])     # 1st stage
    sigma_sq = tf.nn.pool (sigma_sq, [8,8],   'AVG', 'SAME', strides=[8,8])     # 2nd stage
    '''
    '''
    if (cols == 1024) :
        sigma_sq = tf.nn.pool (sigma_sq,   [16,16], 'AVG', 'SAME', strides=[16,16])     # 3rd stage
    elif (cols == 512) :
        sigma_sq = tf.nn.pool (sigma_sq,   [8,8], 'AVG', 'SAME', strides=[8,8])     # 3rd stage
    elif (cols == 256) :
        sigma_sq = tf.nn.pool (sigma_sq,   [4,4], 'AVG', 'SAME', strides=[4,4])     # 3rd stage
    '''
    '''
    if (cols == 768) :
        sigma_sq = tf.nn.pool (sigma_sq,   [12,12], 'AVG', 'SAME', strides=[12,12])     # 3rd stage
    elif (cols == 384) :
        sigma_sq = tf.nn.pool (sigma_sq,   [6,6], 'AVG', 'SAME', strides=[6,6])     # 3rd stage
    elif (cols == 192) :
        sigma_sq = tf.nn.pool (sigma_sq,   [3,3], 'AVG', 'SAME', strides=[3,3])     # 3rd stage
    '''
    '''
    if (cols == 512) :
        sigma_sq = tf.nn.pool (sigma_sq,   [8,8], 'AVG', 'SAME', strides=[8,8])     # 3rd stage
    elif (cols == 256) :
        sigma_sq = tf.nn.pool (sigma_sq,   [4,4], 'AVG', 'SAME', strides=[4,4])     # 3rd stage
    elif (cols == 128) :
        sigma_sq = tf.nn.pool (sigma_sq,   [2,2], 'AVG', 'SAME', strides=[2,2])     # 3rd stage
    '''
    
    c_eps = tf.constant([epsilon]) # all constants must be wrapped in tf.constant to avoid tflite conversion 'rank 0' error
    inv   = tf.math.rsqrt (sigma_sq + c_eps)
    normalized = diff * inv
    #inv1  = tf.image.resize_bilinear (inv, [rows,cols])  # gpu_delegate 'mul' only accepts same h & w shape for both inputs.
    #normalized = diff * inv1   # test out rsqrt performance

    var_shape = [channels]
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    net = normalized * scale + shift

    return net

# InstanceNorm for NCHW
def _instance_norm_nchw(net, train=True):
    batch, channels, rows, cols = [i for i in net.get_shape()]
    mu, sigma_sq = tf.nn.moments(x=net, axes=[2,3], keepdims=True)
    # mu, sigma_sq = tf.nn.moments(x=net, axes=[0], keepdims=True)

    # epsilon = 1e-9 # 1e-3 originally.  set to 1e-9 to avoid a conversion issue for Intel OpenVINO model optimizer
    # normalized = (net-mu)/(sigma_sq + epsilon)**(.5)

    # change constants to tensors to avoid tflite 'rank 0'
    c_eps = tf.constant([1e-9])
    c_pow = tf.constant([.5])
    normalized = (net-mu)/(sigma_sq + c_eps)**(c_pow)

    var_shape = [1,channels,1,1]
    # var_shape = [batch]
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    net = scale * normalized + shift
    # net = tf.nn.batch_normalization(net, mu, sigma_sq, shift, scale, epsilon)

    return net

def _conv_init_vars(net, out_channels, filter_size, transpose=False, data_format='NHWC'):
    
    if data_format == 'NHWC':
        _, rows, cols, in_channels = [i for i in net.get_shape()]
    else:
        _, in_channels, rows, cols = [i for i in net.get_shape()]

    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.random.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)

    return weights_init
