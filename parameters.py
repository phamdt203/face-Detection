import os

if os.path.exists("./cropped")==False:
    os.makedirs("./cropped")

ALPHA = 0.5
THRESHOLD = 0.5
IMAGE_SIZE= 96
LAYERS_TO_FREEZE= 60
NUM_EPOCHS= 100
STEPS_PER_EPOCH= 1
BATCH_SIZE= 32
LEARNING_RATE = 0.001
MARGIN = 0.2
EMBEDDING_SIZE = 128    