import tensorflow as tf

class Layer:

    def __init__(self,initializer):
        self.weightsDict = {}
        self.biasDict = {}
        if (initializer == "Xavier"):
            self.tf_initializer = tf.contrib.layers.xavier_initializer()
        elif (initializer == "Normal"):
            self.tf_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        elif (initializer == "He"):
            self.tf_initializer = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)

    
    def conv2d(self,inputFeature,filterSize, inputSize, outputSize, name,strides = 1):
        filter_shape = [filterSize, filterSize, inputSize, outputSize]
        self.weightName = name + "weight"
        self.biasName = name + "bias"
        with tf.variable_scope("variable", reuse=tf.AUTO_REUSE):
            self.weightsDict[self.weightName] = tf.get_variable(self.weightName, shape=filter_shape,initializer=self.tf_initializer)                
            self.biasDict[self.biasName] = tf.get_variable(self.biasName, shape = outputSize, initializer=self.tf_initializer)                
        convOutput = tf.nn.conv2d(input = inputFeature, filter = self.weightsDict[self.weightName], strides=[1, strides, strides, 1], padding='SAME', name = name)
        finalOutput = tf.nn.bias_add(convOutput, self.biasDict[self.biasName])
        return finalOutput
     
    def avgpool2d(self,inputData):
        return tf.nn.avg_pool(value = inputData, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

    def downSamplingBlock(self,x,input_channels,output_channels,down_size,name):
        if down_size != None:
            x = self.avgpool2d(x)
        print (x.shape)
        self.conv1 =  self.conv2d(x,3,input_channels,output_channels,name+"conv1")
        print (self.conv1.shape)
        x1 = tf.nn.leaky_relu(self.conv1)
        x21 = tf.concat([x,x1],axis=3)
        print (x21.shape)
        self.conv21 = self.conv2d(x21,1, input_channels+output_channels,output_channels,name+"conv21")
        self.conv22 = self.conv2d(self.conv21,3,output_channels,output_channels,name+"conv22")
        print (self.conv22.shape)
        x22 = tf.nn.leaky_relu(self.conv22)
        x31 = tf.concat([x21,x22],axis=3)
        print (x31.shape)
        self.conv31 = self.conv2d(x31, 1, input_channels+2*output_channels,output_channels,name+"conv31")
        self.conv32 = self.conv2d(self.conv31,3,output_channels,output_channels,name+"conv32")
        print (self.conv32.shape)
        out = tf.nn.leaky_relu(self.conv32)
        return tf.layers.batch_normalization(out)

    def upSamplingBlock(self,currentInput,previousInput,skip_channels,input_channels,output_channels,image_width, image_height,name):
        print ("Upsampling")
        x = tf.image.resize_images(images=currentInput,size=[image_width,image_height], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,align_corners=False,preserve_aspect_ratio=False)
        print (x.shape)
        x_concat = tf.concat([x,previousInput],axis=3)
        print (x_concat.shape)
        self.conv11 = self.conv2d(x_concat, 1,skip_channels+input_channels,output_channels,name + "conv11")
        print (self.conv11.shape)
        self.conv12 = self.conv2d(self.conv11,3,output_channels,output_channels,name + "conv12")
        print (self.conv12.shape)
        x1 = tf.nn.leaky_relu(self.conv12)
        x21 = tf.concat([x,x1],axis=3)
        print (x21.shape)
        self.conv41 = self.conv2d(x21,1,skip_channels+input_channels,output_channels,name + "conv41")
        print (self.conv41.shape)
        self.conv42 = self.conv2d(self.conv41, 3, output_channels,output_channels,name + "conv42")
        print (self.conv42.shape)
        out = tf.nn.leaky_relu(self.conv42)
        return out

    def runBlock(self,inputData,in_channels=1,out_channels=2,channel_size=32):
        self.x1 = self.downSamplingBlock(inputData,input_channels=in_channels,output_channels=channel_size, down_size=None,name = "DownBlock1")
        self.x2 = self.downSamplingBlock(self.x1,input_channels=channel_size,output_channels=channel_size, down_size=(2,2),name = "DownBlock2")
        self.x3 = self.downSamplingBlock(self.x2,input_channels=channel_size,output_channels=channel_size, down_size=(2,2),name = "DownBlock3")
        self.x4 = self.downSamplingBlock(self.x3, input_channels=channel_size,output_channels=channel_size, down_size=(2,2),name = "DownBlock4")
        self.x5 = self.downSamplingBlock(self.x4,input_channels=channel_size,output_channels=channel_size, down_size=(2,2),name = "DownBlock5")
        self.x6 = self.upSamplingBlock(self.x5, self.x4, skip_channels=channel_size,input_channels=channel_size, output_channels=channel_size,image_width = 30,image_height = 40,name = "UpBlock1")
        self.x7 = self.upSamplingBlock(self.x6, self.x3,skip_channels=channel_size,input_channels=channel_size, output_channels=channel_size,image_width = 60,image_height = 80,name = "UpBlock2")
        self.x8 = self.upSamplingBlock(self.x7, self.x2,skip_channels=channel_size,input_channels=channel_size, output_channels=channel_size,image_width = 120,image_height = 160, name = "UpBlock3")
        self.x9 = self.upSamplingBlock(self.x8, self.x1,skip_channels=channel_size,input_channels=channel_size, output_channels=channel_size,image_width = 240,image_height = 320, name = "UpBlock4")
        self.out_conv1 = self.conv2d(self.x9,1,channel_size,out_channels,name = "Inference/Output")
        return self.out_conv1
        

            

        







            

	