# 重新建立模型
import numpy as np
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers,optimizers,datasets,Sequential


class SIFTnet(keras.Model):
    def __init__(self):
        super(SIFTnet, self).__init__()
        
        self.layer1 = Sequential([layers.experimental.preprocessing.Rescaling(1. / 1),])
            
        self.layer2 = Sequential([layers.Conv2D(64, (3, 3), activation='relu',padding = 'same'),])
        self.layer3 = Sequential([layers.MaxPooling2D(2, 2),])

        self.layer4 = Sequential([layers.Conv2D(128, (3, 3), activation='relu',padding = 'same'),])
        self.layer5 = Sequential([layers.MaxPooling2D(2, 2),])

        self.layer6 = Sequential([layers.Conv2D(256, (3, 3), activation='relu',padding = 'same'),])
        self.layer7 = Sequential([layers.MaxPooling2D(2, 2),])

        self.layer8 = Sequential([layers.Conv2D(512, (3, 3), activation='relu',padding = 'same'),])
        self.layer9 = Sequential([layers.MaxPooling2D(2, 2),])

        self.layer10 = Sequential([layers.Flatten(),
                          layers.Dropout(0.5),
                          layers.Dense(256, activation='relu'),])
        

        
        
    def call(self, inputs, training = None):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)

        return x
        
        
# def main():
#     model = SIFTnet(32,0.1,4)
#     model.build(input_shape=(None, 256, 256, 3))
#     model.summary()
    
    
# if __name__ == '__main__':
#     main()

