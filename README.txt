This is my second machine learning algorithm and my first deep learning imeple-
mentation, so expect it to be confusing. It has low clean code and it's not o-
bject oriented, as I used it mainly to put in practice what I have learned; as
a secondary objective, it can serve for people who are new to DNNs and want a
basic implementation of them. In resume, the trick is to predict a category with
a CNN, append the result to a sequence and predict this sequence in a real-time
game.

Some of the folders are deprecated, so don't care about them, focus on the py-
thon executables and cited folders. I'll list the steps to get it working in or-
der of file execution:



Note: since I've labeled data and saved the models, you can execute 5_In_Time_Pre-
dictions directly, or also go directly to 4_Image_Classification_CNN and 6_Pro-
ductionTab_RNN to train your own models. If you want more data for the RNNs, I've
left some CSVs in the temp folder, so move them to the Sequences_Data folder.
However, this might be harmful for the predictions, as the original data I've
collected included builds from Terran vs Zerg, which means the sequences differ
totally; as now I'm using only ZvZ late games.
If I had 500000 of sequence data, it possibly could work for against all
races, but it would still struggle to predict the build changing point; this hap-
pens because the model doesn't know if the player will want to play in defense or
aggressive style, neither the opponent's units composition; it can only predict
based on previously built units and structures sequence.



1_Screenshot.py
This is used to take screenshots of your screen each 12 seconds, which was one
of the lowest times required to have a completed building/unit on the replay pro-
duction bar. If you're getting the images from youtube, put the video at full-
screen.
For instance, I'm using: https://www.youtube.com/watch?v=cKT9IETOkt4&t=3498s&ab_channel=ESLArchives



Now, move the screen images stored at ./Games_Img/Game9 or whichever Game folder
you choosed to screenshots and move them to Img_to_Slice.



2_Img_Dataset_Extraction
File 2 will get over all screenshots and slice the production bar for us. You'll
need to move the sliced images at Sliced_ProductionBar_Imgs to Labeled_Img and
label them for your own, you'll also need to copy them and put them into Num_Data
and organize them into their tens and units on num_tens and num_units folders.



3_Time_Img_Dataset_Extraction
This one gets the screenshots too, then it extracts the match minutes. You'll
also have to label this one, from Sliced_Time_Img to Num_Data, then minutes_tens
and minutes_units folders.



4_Image_Classification_CNN
Originally I wanted to use a complex CNN, but my teacher said it was overkilling,
as all of the given images in a certain label look the same, so one one neuronal
activation could already be enough to predict the label; in contrast to analy-
sing, for example, a species of a plant from different angles.
Note that you've to specify the image path you want to process and the model you
want to save, i.e: is it the minutes tens or the production bar units?



5_In_Time_Predictions
Open this and let a match replay run at the background. It will automatically ge-
nerate a CSV containing a historic of the production bar units, amount of each u-
nit and match time. Remember that for this you'll need to have all the CNN's
trained, saved and moved into trained_nns. After each different match, you should
change the CSV name in the very last line of code.
You'll also notice that the RNN model will do it's predictions after a sequence
of 8 units have been reached. It's important to not have deleted the rnn I saved,
otherwise your code will get an error.



6_ProductionTab_RNN
This is the final step to train the RNN for your own, after finally collecting
all the data. Note that I left some comments on the Preprocessing and RNN Model
parts, in case you want to try a custom loss function I tried as I wanted to pre-
dict time and unit amount too, by also adding them to the loss and making it more
efficient. Unfortunately, you can only use the custom loss on the Jupyterlab it-
self; as Tensorflow can't save custom losses, you can't bring it to the In_Time_
Predictions.
In case you just wanted to try different parammeters for the model with the stan-
dard loss, remember to move the saved rnn model into trained_nns folder.



7° step: go back to 5_In_Time_Predictions and execute it with your own trained
models.




Note: for .ipynb files I used the JupyterLab and for .py I used cmd.

It took me hours of GCloud Boost Machine Learning courses and more hours of read-
ing Tensorflow's documentation/Stackoverflow, but I think it was worth it.

Finally, a huge thanks to my teacher who helped me to understand the custom loss
theory, the one hot and lstm functionalities.