from tensorflow import keras
from tensorflow.keras import layers

from tensorflow_docs.vis import embed
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import imageio

import argparse

import data
from model import ConditionalGAN
from model import discriminator
from model import generator



if __name__=="__main__":
    
    # Input type: python train.py --epochs 20 --batchsize 64 --latentdimension 128 --lrdiscriminator 0.0003 --lrgenerator 0.0003--data data_mnist

    # TODO: Move to argparse
    batch_size = 64
    latent_dim = 128 
    epochs = 20

    num_channels = 1 # Infer from data?
    num_classes = 10 # Infer from dataset?
    image_size = 28 # Infer from dataset?


    # Load data
    dataset, image_size, num_channels, num_classes = data.mnist(batch_size)

    # Load model
    generator_in_channels = latent_dim + num_classes
    discriminator_in_channels = num_channels + num_classes
    print(generator_in_channels, discriminator_in_channels)
    
    # Load the generator and disriminator networks, make the model
    disc = discriminator(discriminator_in_channels)
    gen = generator(generator_in_channels)
    cond_gan = ConditionalGAN(discriminator=disc, generator=gen, latent_dim=latent_dim)


    # Train
    cond_gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),)
    
    cond_gan.fit(dataset, epochs=epochs)