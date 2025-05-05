We first need to install lpips

We can do this by running the first code block:
# Install this first
!pip install lpips

After that we need to mount our Google Colab to drive.
Follow these directions and then run the corresponding cell in the Google Colab:
# THIS IS IMPORTANT IN ORDER TO RUN SOME OF THE CODE CELLS BELOW
# 1. Access this link: https://drive.google.com/drive/folders/1Glkvd8CqQwPk38LJiVh6cUG4-8RrL-h2?usp=sharing. This is the link for our shared Google Drive
# 2. Click on the title of the drive: Intro ML Final Project -> Organize -> Add Shortcut -> My Drive (this is really important)
# 3. Once the shortcut is added to "My Drive", refresh the notebook and run this cell again. You should be able to see the folder contents.

After that we need to access our data through kaggle.com
Follow these directions and then run the corresponding cell in the Google Colab:
# NOTE: YOU MUST DO THIS ON GOOGLE COLAB IN ORDER TO ACCESS OUR DATASET
# 1. Create an account on kaggle.com
# 2. Go to settings -> API -> Create new token -> Download token
# 3. You should've downloaded a file called kaggle.json which this cell will then add you to upload

Then you should be fine to run the rest of the Google Colab Notebook

NOTE:
We have one cell block that will take really really long to run (Approximately 80 minutes)
It is this block:

# This is our first CNN that we implemented. It is a classic CNN that uses the VGGNet-11 encoder and also incorporates a learning rate decay.
# This CNN takes an extremely long time to run especially when compared to the other ones.
#
# (For my group, DO NOT run this cell again as we don't want to waste time)

!python cnn_1/train_classification_cnn.py \
    --dataset custom \
    --data_dir /content/ \
    --n_batch 64 \
    --encoder_type 'vggnet11' \
    --n_epoch 20 \
    --learning_rate 0.1 \
    --learning_rate_decay 0.95 \
    --learning_rate_period 5 \
    --checkpoint_path /content/checkpoints \
    --device 'cuda'
