# paw_tracking

Project to track the position of a mouse paw from webcam video using convolutional neural networks, implemented in the Theano python package. 

Contents:
  * paw_pos is a directory containing frames which include a reaching paw
  * paw_neg is a directory containing frames which do include a reaching paw
  * paw_net.py lays out the network architecture and implements training and testing
  * load_dataset.py contains a function which loads the data
  * paws.json contains the location of the paws in each frame, created using the Sloth annotation software.
  
To train and test the network, simply run

    python paw_net.py

