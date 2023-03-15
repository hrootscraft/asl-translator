# American Sign Language Translation

### Overview
In this project, we aim to recognize American Sign Language (ASL) using a novel approach that combines MediaPipe and Long Short-Term Memory (LSTM) neural networks. Unlike the state-of-the-art methods that employ pretrained models with Convolutional Neural Networks (CNN) followed by LSTM layers (and now Transformers), our approach achieves higher accuracy and faster training using only 90 sequences of data. The MediaPipe+LSTM approach offers simplicity and speed, making it suitable for real-time detection. Additionally, the use of MediaPipe eliminates the need for a data generator, allowing for on-the-fly training. Overall, our approach provides an efficient and effective means for ASL detection using minimal data.

<br>
<hr>

Initially we have a Python script that uses the MediaPipe library to detect and extract features from different body parts, such as pose, face, left hand, and right hand. The features are then used to recognize and classify different actions such as "hello," "thanks," and "ok" through machine learning techniques. The code collects the actions through a loop where a camera captures frames and makes detections using the MediaPipe model. The collected frames are saved as .npy files in a specific folder structure that corresponds to each action, sequence, and frame number.

The script contains several functions, including 
* mediapipe_detection that uses the MediaPipe library to detect and process the input image, 
* draw_styled_landmarks that draws connections between the landmarks detected by mediapipe_detection, 
* extract_keypoints that extracts the detected keypoints for each body part, 
* and the main loop that captures frames and makes detections.

The script also applies a collection/wait logic that pauses for two seconds at the start of each sequence to allow the user to reposition themselves and start the action from the beginning.

The code defines and trains a deep learning model for sign language detection using `LSTM layers`. The model architecture is built using `Keras Tuner`, a library for `hyperparameter tuning`, which searches for the optimal set of hyperparameters to maximize the validation accuracy of the model. The best set of hyperparameters is then used to train the final model.

The model architecture consists of an LSTM layer with a variable number of units and activation functions, followed by one to three additional LSTM layers with variable units and activations. Each LSTM layer is followed by a `dropout layer` to prevent overfitting. The output of the LSTM layers is then flattened and passed through one to three dense layers with variable units, activations, and `regularizers`. Each dense layer is followed by another dropout layer. The final output layer uses `softmax activation` to output the probabilities of the three classes.

The hyperparameters are optimized using `Bayesian optimization`, a probabilistic model-based optimization method that seeks to find the `global maximum` of the objective function (in this case, validation accuracy) by building a probabilistic model of the objective function and using it to select the next set of hyperparameters to evaluate.

The code then trains the final model using the best set of hyperparameters and evaluates its performance on a test set. The best epoch is determined based on the validation accuracy, and the model is retrained using the optimal number of epochs. The final model is then evaluated and tested in realtime.
