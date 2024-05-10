**Project Name:** 

# ExplainableAI_GridSearch_CatsVSDogs

**Author Name:**

# Manai Mohamed Mortadha

**Description:**

A project focusing on binary classification using Explainable Artificial Intelligence (XAI) methods, specifically SHAP (SHapley Additive exPlanations), and Grid Search for hyperparameter tuning. The project utilizes EfficientNetV2-B0 architecture on the Cat VS Dog dataset.

**Key Features:**

- Binary classification using EfficientNetV2-B0
- Explainable Artificial Intelligence (XAI) using SHAP
- Grid Search for hyperparameter tuning
- Handling imbalanced dataset using weight adjustment
- Model evaluation metrics: accuracy, precision, recall, F1 score
- Inference time analysis

**Purpose:**

To create a transparent and efficient binary classification model while utilizing XAI methods like SHAP and hyperparameter tuning with Grid Search.

# SHAP_GridSearch_CatsVSDogs
Binary classification, SHAP (Explainable Artificial Intelligence), and Grid Search (for tuning hyperparameters) using EfficientNetV2-B0 on Cat VS Dog dataset.

<a href="https://imgbb.com/"><img src="https://i.ibb.co/Q9fRKzk/Screenshot-3-8-2024-10-18-59-PM.png" alt="Screenshot-3-8-2024-10-18-59-PM" border="0" /></a>

We implemented EfficientNetV2 to perform a binary classification task.
We chose Adam Optimizer because of its dynamic learning rate in default. So, in this case, we don't need to tune the learning rate, but it doesn't necessarily mean that our dynamic learning rate performance is the best. For instance, in some cases, it will converge slower. 

Handling imbalanced dataset:
For handling the imbalanced dataset problem, many approaches have been proposed. The most typical approach is to reduce our sample of classes to the minimum samples. (For instance, if class A has 200 samples and class B has 500 samples, we reduce samples of class B from 500 to 200) However, in this case, we lose a lot of data. The best solution has been proposed in this book: 'Deep Learning for Computer Vision with Python' by 'Adrian Rosebrock'. Briefly, we need to consider the weight for different classes based on the number of samples in them to avoid training our model based on the class that has the most samples. The more samples a class has, the less weight it must have.
At last, based on confusion matrix, we will see that our method for handling imbalance dataset was practical.

To conclude, the model seems to perform well overall, demonstrating high accuracy, precision, recall, and F1 score. Also, the average inference time is approximately 0.14s per image on Google Colab GPU. So, this model can classify 7 images (224,224,3) per second on this GPU.

SHAP (SHapley Additive exPlanations) is a method utilized in explainable artificial intelligence to find out the effectiveness of every feature to the predictions of a machine learning model. One use case in which SHAP can effectively boost model transparency and accountability in image data is in medical images. For instance, for detecting lung tumors from chest X-ray images using a deep learning method, the model performs well in terms of accuracy, but there is a need to understand which regions of the image are more effective in our model's decision. By applying the SHAP method to our model, we can find the specific pixels (features) of the X-ray images that have the most impact on the model's predictions. This information can help medical professionals understand how the model is making its decisions and provide transparency into its reasoning process. It can also aid in identifying potential biases that the model might be relying on. Thus, the SHAP method can improve the accountability of the model significantly by allowing experts to validate the model's predictions. In this case, SHAP can build trust in the model's performance and ensure patient safety.
In this dataset (CatVSDog), SHAP can prove to us that our model's detection is not based on non-important features, such as background. Consequently, if we use our model on different datasets with different backgrounds, our accuracy won't decrease.

Grid Search for Hyperparameter tuning:
In terms of accuracy: For hyperparameter tuning, we can use different methods, such as Grid Search, Random Search, Biyasian Optimization, etc. Here we utilized the Grid Search method.
In terms of speed:
Training time: Obviously, the higher batch size or epochs we have, the less training time becomes. Also, the lower the learning rate we have, the less training time becomes and our model will converge slower but it can achieve better accuracy.
Inference time: None of these three hyperparameters are effective on inference time. To reduce the inference time, we can scale down the resolution, depth of our model, or filter size. Also, pruning and Quantization are practical to reduce the inference time. Most of these methods are effective in reducing inference time and training time.
# XAI_GridSearch_CatsVSDogs_EffNetV2
