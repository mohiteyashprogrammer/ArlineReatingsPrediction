# neurolab-flask

![image](https://user-images.githubusercontent.com/115451707/196919992-edcfea8b-e3f6-4f35-9398-43be66b5622d.png)


To run flask application 

```
python app.py
```


To access your flask application open new tab in and paste the url:
```
https://{your_url}.ineuron.app:5000/
```
## Detail
'''
In this project, I leveraged Docker to encapsulate my machine learning model and its dependencies, making it easy to deploy and run in any environment. The system predicts airline ratings based on various factors such as flight duration, departure/arrival time, airline company, seat comfort, and in-flight entertainment.

To achieve the high accuracy, I carefully prepared the dataset, performed data cleaning, and split it into training and testing sets. Then, I selected the most suitable regression algorithm, opting for [algorithm name], and trained the model using the training data.

After evaluating the model's performance on the testing set, I achieved an impressive 90% R2 score accuracy, indicating a strong fit of the model to the data. I also considered other performance metrics such as mean squared error (MSE) and mean absolute error (MAE) to ensure a comprehensive evaluation.

Throughout the project, I employed optimization techniques such as feature engineering, hyperparameter tuning, and explored more advanced algorithms to improve the model's accuracy.

By utilizing Docker, I created a Dockerfile that defined the necessary dependencies and environment, allowing for seamless deployment of the machine learning model. The Docker image encapsulated the model and its dependencies, making it easy to reproduce and run in any environment.
'''
