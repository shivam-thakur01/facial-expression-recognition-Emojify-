Introduction of our project

Emojis are small images that are commonly included in social media text
messages. The combination of visual and textual content in the same
message builds up a modern way of communication. Emojis or avatars are
ways to indicate nonverbal cues. These cues have become an essential
part of online chatting, product review, brand emotion, and many more. It
also led to increasing data science research dedicated to emoji-driven
storytelling. With advancements in computer vision and deep learning, it is
now possible to detect human emotions from images. In this deep learning
project, we will classify human facial expressions to filter and map
corresponding emojis or avatars. This project is not intended to solve a
real-world problem, instead it allows us to see things more colorful in the
chatting world. Emojify is a software which deals with the creation of
Emoji’s or Avatars.
So, in this project I tried to use facial expressions recognition to implement
the emojify facial expressions using Convolutional Neural Network (CNN)
and Deep Learning

Tech Stack Used
● Deep Learning
● Convolutional Neural Network (CNN)
● OpenCV
● Tenserflow

DATASET & ITS FEATURES

In this project, the dataset used to train the models is FER-2013. The FER-2013
dataset consists of 35887 images, of which 28709 labelled images belong to the
training set and the remaining 7178 images belong to the test set. The images
in FER-2013 dataset is labeled as one of the seven universal emotions: Happy,
Sad, Angry, Surprise, Disgust, Fear and Neutral. Among these emotion
classifications, the most images belong to ‘happy’ emotions account to 7215
images of the 28709 training images. The number of images that belong to each
emotion is given by returning the length of each directory using the OS module
in Python. The images in FER-2013 dataset are in grayscale and is of
dimensions 48x48 pixels. This dataset was created by gathering results from
Google Image search of each emotions. The number of images of each emotion
is given in table 1. The number of images of each emotion type is returned by
the functions of ‘OS’ module in python. To explore the dataset further, and to
understand what kind of images lie in the dataset, we plot few example images
from the dataset using the ‘utils’ module in python. The resultant plot of
example images obtained is given in figure 1. From various research papers,
we have studied that the average attainable accuracy of a training model
developed using FER-2013 dataset is 66.7% and our aim is also to design a
CNN model with similar or a better accuracy.

CONCLUSION

Using the FER-2013 dataset, a test accuracy of 56% and train accuracy is
62.939% is attained with this designed CNN model. The achieved results are
satisfactory as the average accuracies on the FER-2013 dataset is 65% +/- 5%
and therefore, this CNN model is nearly accurate. For an improvement in this
project and its outcomes, it is recommended to add new parameters wherever
useful in the CNN model and remove unwanted and not-so useful parameters.
Adjusting the learning rate and adapting with the location might help in
improving the model. Accommodating the system to adapt to a low graded
illumination setup and nullify noises in the image can also add onto the efforts
to develop the CNN model. Increasing the layers in the CNN model might not
deviate from the achieved accuracy, but the number of epochs can be set to
higher number, to attain a higher accurate output. Though, increasing the
number of epochs to a certain limit, will increase the accuracy, but increasing
the number of epochs to a higher value will result in over- fitting.


FUTURE WORK
1. At this level, we have included only 5 emojis to mask our facial expressions,
but our future work would include far more emojis so that we can map emojis
as per the expressions accurately.
2. To improve the accuracy of our model. In order to match it to the average
accuracy of FER dataset.
3. To integrate this model with the website and provide option to the user to click
or upload the photo as well as video to generate the emoji based on the facial
expressions.