# **Traffic Sign Recognition Project** 

---

[//]: # (Image References)

[image1]: ./img/writeup_img1.png "Images of different classes"
[image2]: ./img/writeup_img2.png "Images of the same class"
[image3]: ./img/writeup_img3.png "Image distribution per class in original training dataset"
[image4]: ./img/writeup_img4.png "Image filters"
[image5]: ./img/writeup_img5.png "Image distribution in extended training datatset"
[image6]: ./img/writeup_img6.png "Loss and Accuracy"
[image7]: ./img/writeup_img7.png "Traffic Sign Images in wild 1"
[image8]: ./img/writeup_img8.png "Traffic Sign Images in wild 2"
[image9]: ./img/writeup_img9.png "Cropped traffic sign images"
[image10]: ./img/writeup_img10.png "Scaled traffic sign images"
[image11]: ./img/writeup_img11.png "Soft Max probabilities"
[image12]: ./img/writeup_img12.png "Soft Max for sign 1"
[image13]: ./img/writeup_img13.png "Soft Max for sign 2"
[image14]: ./img/writeup_img14.png "Soft Max for sign 3"
[image15]: ./img/writeup_img15.png "Soft Max for sign 4"
[image16]: ./img/writeup_img16.png "Soft Max for sign 5"
[image17]: ./img/writeup_img17.png "Soft Max for sign 6"
[image18]: ./img/writeup_img18.png "Soft Max for sign 7"
[image19]: ./img/writeup_img19.png "Soft Max for sign 8"
[image20]: ./img/writeup_img20.png "Soft Max for sign 9"
[image21]: ./img/writeup_img21.png "Soft Max for sign 10"

Note, that the [python notebook](https://github.com/selyunin/carnd_t1_p2/blob/master/Traffic_Sign_Classifier.ipynb) 
already contains a solid description of the project.

---

### Data Set Summary & Exploration

#### 1. A basic summary of the data set.

The original dataset is provided as pickle files, which include train, validation and test data.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Let us plot one random image from each training class

![ ][image1]

Let us plot a number of images from the same training class to see the variations in data

![ ][image2]

Let us plot a number of images from the same training class to see the variations in data

![ ][image3]

The data is highly unbalanced, we will extend the training set to equalize the number of images per class

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. 

I used functions available in `skimage.exposure` and `skimage.transform`
to equalize image histogram, apply logarithmic and sigmoidal transformation, 
and slight rotation of the image. This resulted in 12 filters, which I can apply on the 
image to extend the original dataset.

![ ][image4]

After applying image filters on underrepresented images (see pseudo-code and implementation in the python notebook), 
the distribution of classes looks as follows:

![ ][image5]

As a last step, I normalized the image data to make a data with 0 mean the scaled images are 
in domain [-0.5, 0.5].


#### 2. CNN model architecture.

I extended the original LeNet architecture with additional convolutional layer, and added dropout between the layers.
Moreover, since there are much more image classes then handwritten digits, I added more features to the convolutional layers.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x24 				|
| Dropout				| Training: keep prob : 0.5, evaluation : 1.0	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x64  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 					|
| Dropout				| Training: keep prob : 0.5, evaluation : 1.0	|
| Convolution 2x2	    | 1x1 stride, valid padding, outputs 4x4x128  	|
| RELU					|												|
| Dropout				| Training: keep prob : 0.5, evaluation : 1.0	|
| Fully connected 1		| Input: 2048, out: 820							|
| RELU					|												|
| Fully connected 2		| Input: 820, out: 256 							|
| RELU					|												|
| Fully connected 3		| Input: 256, out: 43							|
| RELU					|												|
|:---------------------:|:---------------------------------------------:| 
 


#### 3. Training the model

To train the model I used Adam Optimizer, available in Tensorflow.
The batch size is set to 256 images, we train for 50 epochs, and use learning rate of 0.8e-4.
Training and validation loss and accuracy look as follows:

![ ][image6]

#### 4. Finding a solution to get the validation set accuracy > 0.93. 

Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I started with the LeNet architecture.

My final model results are:
* training set accuracy of 0.995
* validation set accuracy of 0.956 
* test set accuracy of 0.952

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model s accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Getting new images

I used Google street view to get the traffic sign images from streets of Hamburg.

![ ][image7]

![ ][image8]

Then I cropped the images to a square, obtaining:

![ ][image9]

To be applicable with the trained CNN, the images are re-scaled to 32x32:

![ ][image10]

The last step was to normalize the images, i.e. convert them to the domain [-0.5, 0.5].


#### 2.  Model predictions on new traffic signs 

Let us visualize softmax probabilites when predicting the traffic signs.

![ ][image11]

Here are the results of the prediction:
1.  Slippery road
2.  Go straight or right
3.  Pedestrians
4.  Wild animals crossing
5.  Keep right
6.  General caution
7.  No entry
8.  Speed limit (20km/h) -- not correct, should be 30!
9.  Priority road
10. Yield


The model was able to correctly guess 9 of the 10 traffic signs, which gives an accuracy of 90%. 
This is comparable with test and validation set accuracy. Moreover, the model is unsure about the sign
8, although all the probabilities indicate that this is a speed restriction sign.

#### 3. Visualizing the softmax probabilities for each prediction of the new image. 


![ ][image12]

![ ][image13]

![ ][image14]

![ ][image15]

![ ][image16]

![ ][image17]

![ ][image18]

![ ][image19]

![ ][image20]

![ ][image21]
