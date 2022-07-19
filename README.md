# NN-Weather-Analysis
Many industries have the need to identify current and past weather conditions. The data helps them plan, organize, and/or optimize their operations. Using CNNs we are offering the potential to automate this by providing a digital eye. If an image recognition model could be built to identify conditions by looking at images of the weather, it could be deployed to automatically trigger smart devices. The dataset used in this project consists of 5,551 training images and 1,300 testing images. It has 11 classes representing different weather conditions, these classes are: dew, fogsmog, frost, glaze, hail, lightning, rain, rainbow, rime, sandstorm, snow.

***

## Approaches Used in all trials
**a) Early Stopping** </br>
&emsp; &emsp; A basic problem that arises in training a neural network is to decide how many epochs a model should be trained. </br>
&emsp; Too many epochs may lead to overfitting of the model and too few epochs may lead to underfitting of the model. </br>
&emsp; In this technique, we can specify an arbitrarily large number of training epochs and stop training once the model </br>
&emsp; performance stops improving on a **hold out validation** dataset.

**b) Model Check Point** </br>
&emsp; &emsp; The EarlyStopping callback will stop training once triggered, but the model at the end of training may not be the model with </br>
&emsp; the best performance on the validation dataset. An additional callback is required that will save the best model observed </br>
&emsp; during training for later use. This is known as the **ModelCheckpoint** callback. The ModelCheckpoint callback is flexible in </br>
&emsp; the way it can be used, but in our case, we will use it only to save the best model observed during training </br>
&emsp; as defined by a chosen performance measure on the validation dataset. 

***

### References
- [TensorFlow](https://www.tensorflow.org/guide/low_level_intro)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html)
- [CNN](https://medium.com/technologymadeeasy/the-best-explanation-of-convolutional-neural-networks-on-the-internet-fbb8b1ad5df8)
- [Vgg16](https://neurohive.io/en/popular-networks/vgg16/)
- [DenseNet](https://www.pluralsight.com/guides/introduction-to-densenet-with-tensorflow)
- [Image Data Generator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)
- [Early Stopping](https://keras.io/api/callbacks/early_stopping/)

### Copyrights
- KAN Org.
- University of Ain Shams, Egypt
