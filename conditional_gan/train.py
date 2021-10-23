from tensorflow import keras
from tensorflow.keras import layers

from tensorflow_docs.vis import embed
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import imageio

import argparse
from pathlib import Path
import os

import data
from model import ConditionalGAN
from model import discriminator
from model import generator



if __name__=="__main__":
    
    # Input type: python train.py --epochs 20 --batchsize 64 --latentdimension 128 --lrdiscriminator 0.0003 --lrgenerator 0.0003--data data_mnist
    parser = argparse.ArgumentParser(description="Train a GAN model. Select a dataset, set parameters and wait!")
    parser.add_argument("--epochs", default=None, help="Number of epochs")
    parser.add_argument("--batchsize", default=64, help="Batch size")
    parser.add_argument("--latentdimension", default=None, help="Dimension of latent vector")
    parser.add_argument("--data", choices=["MNIST", "CIFAR10"], default=None, help="Select dataset")
    parser.add_argument("--lrdiscriminator", default=0.0003, help="Discriminator learning rate")
    parser.add_argument("--lrgenerator", default=0.0003, help="Generator learning rate")


    # TODO: Add remaining arguments
    args = parser.parse_args()
    epochs = int(args.epochs)
    batch_size = int(args.batchsize)
    latent_dim = int(args.latentdimension)

    num_channels = 1 # Infer from dataset?
    num_classes = 10 # Infer from dataset?
    image_size = 28 # Infer from dataset?

    # Create new folder for experiment
    folder_ext = max([int(name[3:]) for name in os.listdir("./experiments") if name[:3] == "exp"]) + 1
    Path("experiments/exp"+str(folder_ext)).mkdir(parents=True, exist_ok=True)

    # =================================================================
    # Load data 
    #
    dataset, image_size, num_channels, num_classes = data.mnist(batch_size)

    # Load model
    generator_in_channels = latent_dim + num_classes
    discriminator_in_channels = num_channels + num_classes
    print(generator_in_channels, discriminator_in_channels)
    
    # Load the generator and disriminator networks, instantiate the model
    disc = discriminator(discriminator_in_channels)
    gen = generator(generator_in_channels)
    cond_gan = ConditionalGAN(discriminator=disc, generator=gen, 
                                latent_dim=latent_dim)


    # =================================================================
    # Train
    cond_gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),)
    

    # Checkpoints TODO: Fix
    checkpoint_filepath = "./experiments/exp" + str(folder_ext)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor="discriminator_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        options=None,
    )

    # Run training loop
    cond_gan.fit(dataset, epochs=epochs)