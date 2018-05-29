import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, device=gpu2, floatX=float32"
import keras
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class DCGAN(object):
    ''' The ConvGAN class takes the input shapes (width, height, channels) along with number of classes (num_classes), 
	dimension of latent noise and the binary input for whether show the model configures or not.
	    It should be noted that the presented DCGAN is conditioned on the MNIST labels using embedding layer.

	Author: Ata Akbari Asanjan
	Last Modified: 5/25/2018 6:08 PM
    '''
    def __init__(self, width=28, height=28, channels=1, num_classes=10, latent_dim=100, summary=False):
        self.width = width
        self.height = height
        self.channels = channels
        self.image_shape = (width, height, channels)
	self.latent_dim = latent_dim
	self.num_classes = num_classes
	self.summary = summary
        
        self.optimizerG = keras.optimizers.Adam(beta_1=0.5)
	self.optimizerD = keras.optimizers.Adam(beta_1=0.5)
        
        self.G = self.generator()
        self.G.compile(loss='binary_crossentropy', optimizer=self.optimizerG)
        
        self.D = self.discriminator()
        self.D.compile(loss='binary_crossentropy', optimizer=self.optimizerD)
        
        self.stacked_G_D = self.stack_G_D()
        self.stacked_G_D.compile(loss='binary_crossentropy', optimizer=self.optimizerG)
        
    def generator(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2DTranspose(256, kernel_size=(7, 7), padding='Valid', input_shape=((1, 1, self.latent_dim))))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LeakyReLU(0.2))

        model.add(keras.layers.Conv2DTranspose(128, strides=(2, 2), kernel_size=(5, 5), padding='same'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LeakyReLU(0.2))

        model.add(keras.layers.Conv2DTranspose(64, strides=(2, 2), kernel_size=(5, 5), padding='same'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LeakyReLU(0.2))
                
        model.add(keras.layers.Conv2DTranspose(1, kernel_size=(5, 5), padding='same', activation='tanh'))

        if self.summary:
	    print("Generator Summary:")
            model.summary()

	noise = keras.layers.Input(shape=(self.latent_dim,))
        label = keras.layers.Input(shape=(1,), dtype='int32')
        label_embedding = keras.layers.Flatten()(keras.layers.Embedding(self.num_classes, self.latent_dim)(label))

        model_input = keras.layers.multiply([noise, label_embedding])
	model_input = keras.layers.Reshape((1, 1, self.latent_dim))(model_input)
        img = model(model_input)

        return keras.models.Model([noise, label], img)
    
    def discriminator(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(128, kernel_size=(5, 5), padding='same', strides=(2, 2), input_shape=self.image_shape))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LeakyReLU(0.2))

        model.add(keras.layers.Conv2D(256, kernel_size=(5, 5), padding='same', strides=(2, 2)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LeakyReLU(0.2))

        model.add(keras.layers.Conv2D(1, kernel_size=(7, 7), padding='valid', activation='sigmoid'))
	model.add(keras.layers.Reshape((1,)))

        if self.summary:
	    print("Discriminator Summary:")
            model.summary()

	img = keras.layers.Input(shape=self.image_shape)
        label = keras.layers.Input(shape=(1,), dtype='int32')

        label_embedding = keras.layers.Flatten()(keras.layers.Embedding(self.num_classes, np.prod(self.image_shape))(label))
        flat_img = keras.layers.Flatten()(img)

        model_input_flat = keras.layers.multiply([flat_img, label_embedding])
	model_input = keras.layers.Reshape(self.image_shape)(model_input_flat)

        validity = model(model_input)

        return keras.models.Model([img, label], validity)

    def stack_G_D(self):
	noise = keras.layers.Input(shape=(self.latent_dim,))
        label = keras.layers.Input(shape=(1,), dtype='int32')
        img = self.G([noise, label])

        self.D.trainable = False

	valid = self.D([img, label])

        return keras.models.Model([noise, label], valid)
    
    def train(self, trainX, trainY=None, nb_epochs=1, batch_size=1, verbose=True, save_intervals=False, plot_intervals=False):
	if trainY is not None:
		print("The Convolutional GAN is conditioned on the label data.")        

        for cnt in range(nb_epochs):
            
            # train discriminator
            random_index = np.random.randint(0, len(trainX) - batch_size/2)
            valid_images = trainX[random_index : random_index + batch_size/2]
            valid_images = valid_images.reshape(batch_size/2, self.width, self.height, self.channels)
	    valid_labels = trainY[random_index : random_index + batch_size/2]
            gen_noise = np.random.normal(0, 1, (batch_size/2, 100))

            synthetic_images = self.G.predict([gen_noise, valid_labels])
            
            x_combined_batch = np.concatenate((valid_images, synthetic_images))
            y_combined_batch = np.concatenate((np.ones((batch_size/2, 1)), np.zeros((batch_size/2, 1))))
	    label_combined_batch = np.concatenate((valid_labels, valid_labels))
            
            d_loss = self.D.train_on_batch([x_combined_batch, label_combined_batch], y_combined_batch)
            
            # train generator
            noise = np.random.normal(0, 1, (batch_size, 100))
	    sample_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)
            y_mislabeled = np.ones((batch_size, 1))
            
            g_loss = self.stacked_G_D.train_on_batch([noise, sample_labels], y_mislabeled)
            if verbose:
                print("epoch: {}, [Discriminator_loss: {:.4}], Generator_loss: {:.4}]".format(cnt+1, 
                                                                                              d_loss, g_loss))

            if not isinstance(save_intervals, (bool)) and (cnt % save_intervals == 0):
                self.stacked_G_D.save("~/Stacked_generator_discriminator_{}.h5".format(cnt))
        
            if not isinstance(plot_intervals, (bool)) and (cnt % plot_intervals == 0):
                self.plot_images(step=cnt)
    
        if isinstance(save_intervals, (bool)) and save_intervals:
            self.stacked_G_D.save("~/Stacked_generator_discriminator_Conditional_DCGAN_03.h5")  
        if isinstance(plot_intervals, (bool)) and plot_intervals:
            self.plot_images(step=cnt)
            
#    def predict(self,testX, testY=None, batch_size=1, save_intervals=False, plot_intervals=False):
	

    def plot_images(self, step=0):
        noise=np.random.normal(0, 1, (16, 100))
	sample_labels = np.array([[2]*4, [4]*4, [5]*4, [8]*4]).reshape(-1, 1)
        filename="~/Cond_GAN_03/mnist_{}_conditional_DCGAN_03.png".format(step)
        images = self.G.predict([noise, sample_labels])
        # Denormalize data to [0., 255.]
        images *= 127.5
        images += 127.5
        plt.figure(figsize=(10, 10))    
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = np.squeeze(images[i])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')


if __name__ == '__main__':
    (trainX, trainY), (_, _) = keras.datasets.mnist.load_data()
    print("MNIST dataset shape is {}.".format(trainX.shape))
    # Rescale -1 to 1
    trainX = (trainX.astype(np.float32) - 127.5) / 127.5
    trainX = np.expand_dims(trainX, axis=3)
    gan = DCGAN(summary=True, width=28, height=28, channels=1)
    gan.train(trainX, trainY, nb_epochs=4000, batch_size=100, plot_intervals=20, save_intervals=400)
