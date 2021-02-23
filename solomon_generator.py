import tensorflow as tf
import os
import LTSM_text_gen as txtgen

print("\n\n\n")

# data import
data_path = './data/Proverbs.csv'
data = open(data_path, "r")
data = [line for line in data]
# create predictors and label
checkpoint_path = "training_solomon/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")  # Name of the checkpoint files
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,  # Automatically creates folder as needed
    save_weights_only=True)

solomon_parrot = txtgen.TextGen(data)

# solomon_parrot.train_model(epochs=300, checkpoint_callback=checkpoint_callback)
solomon_parrot.model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

num_chirped_words = 40
seed_texts = ["my son forget not", "my daughter forget not", "honor thy father by", "honor thy mother by", "he was every woman's man and every man's", "one two three", "love thy neighbor and", "if man lies with", "if woman lies with"]
solomon_parrot.generate_text(seed_texts, num_chirped_words)


