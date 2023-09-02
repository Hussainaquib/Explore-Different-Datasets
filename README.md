
# Explore Different Datasets with Different Classifiers

This Streamlit app enables users to explore different classifiers applied to various datasets, offering real-time interactivity to select datasets, classifiers, and parameters, while visualizing data in a 2D space using PCA


## Required libraries
- streamlit
- pandas
- matplotlib
- scikit-learn
- numpy
## Features

- Select the any dataset from given list
- Select any classifier from given list
- set the parameter
- Shows the graph of dataset , Shape of dataset ,number of class in dataset , accuracy of model


## Datasets Used
- Iris
- Breast Cancer
- Wine
- Diabetes
- Digits
## Classifier Used
- KNN
- SVM
- Random Forest
## How to use ?
To use this web application project, follow these steps:
- Select a dataset from the dropdown menu.
- Select a classifier from the dropdown menu.
- Set the parameter 
and then following things is display on screen
- Dataset Name
- Shape of dataset
- number of classes
- Classifier Name
- Accuracy
- Graph plot of Dataset
## Running Tests

To run tests, run the following command on cmd

```bash
  streamlit run project_name.py
```


## Demo

link for demo

https://explore-different-datasets-gv5g2rf7zpmhqplqnsdeaz.streamlit.app/
# Lessons Learned

## What did you learn while building this project?

Throughout the process of creating the "Explore Different Datasets with Different Classifiers" project, I gained valuable insights into the five fundamental stages of machine learning:

- Dataset Selection: I began by selecting appropriate datasets for analysis. In this project, I leveraged the built-in datasets readily available in scikit-learn. These datasets are well-prepared and require minimal preprocessing.

- Data Preprocessing: Due to the use of scikit-learn's built-in datasets, extensive data preprocessing was unnecessary. These datasets are already cleaned and well-structured, allowing me to focus more on model selection and evaluation.

- Algorithm Selection: The next step involved choosing the most suitable machine learning algorithm for the task at hand. This decision was crucial in achieving accurate and meaningful results.

- Model Training: To train the selected machine learning models, I employed the .fit() function. This phase involved using the chosen algorithm to learn patterns and relationships within the data.

- Model Evaluation: After training the models, I evaluated their performance using various metrics. To assess the model's accuracy and effectiveness, I utilized the .predict(X_test) function to make predictions on a test dataset.

Additionally, to enhance the project's visual appeal and facilitate a better understanding of the datasets, I incorporated graphical elements. I achieved this by utilizing the Matplotlib library to create insightful graphs and plots. Furthermore, to add a touch of sophistication to these visuals, I employed the .colorbar() function to incorporate color schemes, making the graphs more informative and visually engaging. Through these steps, I successfully developed an interactive platform for exploring machine learning classifiers on diverse datasets.
 
## What challenges did you face and how did you overcome them?

Throughout the deployment phase of my project, "Explore Different Datasets with Different Classifiers," I encountered a series of challenges that tested my problem-solving skills and determination. Here's a breakdown of the challenges I faced and the steps I took to overcome them:

- 1st Problem (Platform Selection) : Initially, I was uncertain about which platform to choose for deployment. After careful consideration of factors like cost and ease of use, I decided to go with Streamlit Sharing due to its free availability.
- 2nd Problem (with pycharm) : Problem i faced that is i write my python code on pycharm project is done and easy to run by using streamlit run main.py command on cmd but for deployment on streamlit you first your upload your .py file on github but i could not able to upload my .py file with attached required file and then i decide that i use jupyter notebook for this project
- 3rd Problem (Jupyter Notebook Performance) :  I encountered performance issues when running the Streamlit app's code within Jupyter Notebook. This slowdown was hindering the deployment process. To address this, I created a main.py file manually on my system to separate the code from Jupyter Notebook.
- 4th Problem (Missing Requirement.txt) : While preparing for deployment, I realized that my GitHub repository lacked a proper requirement.txt file, which is essential for specifying package dependencies. To rectify this, I added the necessary packages to the requirement.txt file based on research and guidance from online resources. 

These challenges highlighted the importance of thorough preparation and attention to detail when deploying a project. By persistently seeking solutions and leveraging online resources, I ultimately overcame these obstacles and successfully deployed my project on Streamlit. This experience taught me valuable lessons about the intricacies of the deployment process and reinforced the significance of proper project organization and documentation.
## References

How to Deploy Your Streamlit App on Streamlit sharing:
https://medium.com/swlh/how-to-deploy-your-streamlit-app-on-streamlit-sharing-4705958ee944#:~:text=To%20deploy%20your%20app%20on,on%20Sign%20in%20with%20Github.

Why Streamlit App’s code take more time to run in Jupyter Notebook:
https://medium.com/p/da327300e918
## Documentation

[Link](https://medium.com/p/4afa01eb92ba)


## Author

- [Saquib Hussain](https://github.com/Hussainaquib)


## Feedback

If you have any feedback, please reach out to me at saquib9451@gmail.com

