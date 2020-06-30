# Face-Recognition-using-FaceNET-Deep-learning-Opencv

Face Recognition with OpenCV,Python & DeepLearning(Facenet)

Objective:
 To build our face recognition system, we’ll first perform face detection, 
Extract face embeddings from each face using deep learning, train a face recognition model on the embeddings, 
And then finally recognize faces in both images and video streams with OpenCV.

Steps:
Detect Faces
Compute 128-d face embeddings to quantify a face
Train a Support Vector Machine (SVM) on top of the embeddings
Recognize faces in images and video streams

How Face Recognition works








In order to build our OpenCV face recognition pipeline, we’ll be applying deep learning in two key steps:
To apply face detection, which detects the presence and location of a face in an image, but does not identify it
To extract the 128-d feature vectors (called “embeddings”) that quantify each face in an image


Methodology :
 First, we input an image or video frame to our face recognition pipeline. Given the input image, we apply face detection to detect the location of a face in the image.
Using a pre-trained SSD face detection framework.

We can compute facial landmarks, enabling us to preprocess and alings the faces. Face alignment, as the name suggests, is the process of (1) identifying the geometric structure of the faces and (2) attempting to obtain a canonical alignment of the face based on translation, rotation, and scale.

While optional, face alignment has been demonstrated to increase face recognition accuracy in some pipelines

After we’ve (optionally) applied face alignment and cropping, we pass the input face through our deep neural network:






The FaceNet deep learning model computes a 128-d embedding that quantifies the face itself. 






Process of network compute the face embedding 

To train a face recognition model with deep learning, each input batch of data includes three images:
The anchor
The positive image
The negative image
The anchor is our current face and has identity A.
The second image is our positive image — this image also contains a face of person A.
The negative image, on the other hand, does not have the same identity, and could belong to person B, C, or even Y!
The point is that the anchor and positive image both belong to the same person/face while the negative image does not contain the same face.
The neural network computes the 128-d embeddings for each face and then tweaks the weights of the network (via the triplet loss function) such that:
   
The 128-d embeddings of the anchor and positive image lie closer together
While at the same time, pushing the embeddings for the negative image father away

7. In this manner, the network is able to learn to quantify faces and return highly robust and discriminating embeddings suitable for face recognition.

For own model:
Even though the deep learning model we’re using today has (very likely) never seen the faces we’re about to pass through it, the model will still be able to compute embeddings for each face — ideally, these face embeddings will be sufficiently different such that we can train a “standard” machine learning classifier (SVM, SGD classifier, Random Forest, etc.) on top of the face embeddings, and therefore obtain our OpenCV face recognition pipeline




Below steps for which i will use in  coding :
Step #1 which is responsible for using a deep learning feature extractor to generate a 128-D vector describing a face. All faces in our dataset will be passed through the neural network to generate embeddings.
Step #2. We’ll detect faces, extract embeddings, and fit our SVM model to the embeddings data.
Step #3 and we’ll recognize faces in images. We’ll detect faces, extract embeddings, and query our SVM model to determine who is in an image. We’ll draw boxes around faces and annotate each box with a name.

                                             


 
