#**Traffic Sign Recognition** 

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

Note, that the [python notebook](https://github.com/selyunin/carnd_t1_p2/blob/master/Traffic_Sign_Classifier.ipynb) 
already contains  solid description of the project.

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
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


