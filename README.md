# Persian Image Captioning (PIC)

## Overview


## Contents

- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Inference](#inference)
- [Improvment](#improvment)


## Dataset

This project's dataset is a subset of the [Coco-Flickr Farsi](https://www.kaggle.com/datasets/navidkanaani/coco-flickr-farsi) dataset, totaling 19 GB. To train the model effectively, the dataset has been filtered by clipping captions' length between 10 to 25 tokens. The resulting equalized dataset comprises 40,000 images. Histograms and plots illustrating the distribution of caption lengths in the dataset are as follows:

![coco_flicker_org_ds](https://github.com/rasoulasadiyan/Persian-Image-Captioning-PIC/assets/100882487/3dc4739e-f3ca-4ef8-9782-a058487a65d1)
*Caption: Distribution of original captions lenght*

![equalized_dataset](https://github.com/rasoulasadiyan/Persian-Image-Captioning-PIC/assets/100882487/e1968001-138e-4d00-9d96-94cd3ad9f00e)
*Caption: Distribution of filtered dataset caption lens*


## Model Architecture
PIC model is designed with a three-part architecture, utilizing Convolutional Neural Networks (CNNs), Encoders and Decoders (Transformers):

1. **CNN:**
   The EfficientNetB0 model is employed as the initial layer to extract meaningful features from input images. The pre-trained weights from ImageNet are used, and the feature extractor is frozen during training.

2. **Encoder:**
   The extracted image features are passed through a Transformer-based encoder. This encoder enhances the representation of the inputs, incorporating self-attention mechanisms for better context understanding.

3. **Decoder:**
   This model takes the encoder output and text data (sequences) as inputs. It is trained to generate captions by utilizing self-attention and cross-attention mechanisms. The decoder incorporates positional embeddings for sequence information and employs dropout layers for regularization.
Models**: Utilize the latest advancements in deep learning for image captioning.

## Inference
model outputs:

![predict01](https://github.com/rasoulasadiyan/Persian-Image-Captioning-PIC/assets/100882487/d56872f8-cf5b-40c8-b512-bec91af74944)

![predict02](https://github.com/rasoulasadiyan/Persian-Image-Captioning-PIC/assets/100882487/56e89a0c-9857-409d-90bb-1ec3e762ca40)


### Improvment
