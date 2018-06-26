# From http://machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html

from random import shuffle
import glob
import cv2

shuffle_data = True
cat_dog_train_path = "../Datasets/dogs_and_cats/*.jpg"

# Read addresses and labels from folder
addrs = glob.glob(cat_dog_train_path)
labels = [0 if "cat" in addr else 1 for addr in addrs]

# Shuffle data
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)

# Divide the data into 60% training, 20% validation, and 20% test
train_addrs = addrs[0:int(0.6 * len(addrs))]
train_labels = labels[0:int(0.6 * len(labels))]

val_addrs = addrs[int(0.6 * len(addrs)):int(0.8 * len(addrs))]
val_labels = labels[int(0.6 * len(labels)):int(0.8 * len(labels))]

test_addrs = addrs[int(0.8 * len(addrs)):]
test_labels = labels[int(0.8 * len(labels)):]

# Load images into TFRecords file


def load_image(addr):
    """
    Read an image and resize it to (224 * 224)
    """
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img
