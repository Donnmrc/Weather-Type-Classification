# 🌦️ Weather Type Classification Web Application 🌦️

Welcome to the Weather Type Classification Web Application repository! This project is a Flask-based ML web application designed to classify different weather types such as Sunny, Rainy, Cloudy, and Snowy using multiple machine learning models. Whether you're a data science enthusiast, a machine learning practitioner, or just curious about weather classification, this web app provides an interactive platform to explore and understand how models can predict weather patterns.

## Overview ℹ️

In today's world of interconnected technologies, the ability to accurately predict weather conditions is crucial for various industries such as agriculture, transportation, and emergency services. This web application leverages the power of machine learning to classify different weather types based on historical data. By utilizing models like decision trees, deep learning, k-nearest neighbors (KNN), logistic regression, random forest, and support vector machines (SVM), the application provides users with insights into how these models can be trained and deployed for weather classification tasks.

## Features 🛠️

- Interactive UI for inputting weather data
- Classification of weather types: Sunny, Rainy, Cloudy, Snowy
- Multiple machine learning models for comparison
- Predictive modeling for weather forecast
- Seamless integration with Flask for web deployment
- Easy-to-understand visualization of model performance

## Repository Details 📁

- **Repository Name:** Weather-Type-Classification-WebApp
- **Short Description:** A Flask-based ML web app for classifying weather types (Sunny, Rainy, Cloudy, Snowy) using multiple models.
- **Topics:** ai, classification, data-science, decision-trees, deep-learning, flask, iot, knn, logistic-regression, machine-learning, ml-models, predictive-modeling, python, random-forest, support-vector-machines, weather, weather-classification, weather-forecast
- **Release Link:** [Download and Execute](https://github.com/IsaCouture/Weather-Type-Classification-WebApp/releases)

For more details and to access the latest version of the web application, please visit the [Releases](https://github.com/IsaCouture/Weather-Type-Classification-WebApp/releases) section of this repository.

## How to Use 🚀

1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Flask application with `python app.py`.
4. Access the web application through your browser by copying the URL/link code given after doing Step 3.
5. Input the relevant weather data and explore the different weather classifications provided by the machine learning models.

## Installation and Setup (Specific) Guide 🖥️⚙️

Follow these step-by-step instructions to properly download and run the project on your local machine.
1️⃣ Download the Repository

Option 1: Download ZIP (Recommended for Beginners)

Go to the repository page

Click the green Code button

Select Download ZIP

Locate the downloaded .zip file in your Downloads folder

Option 2: Clone via Git

git clone https://github.com/IsaCouture/Weather-Type-Classification-WebApp.git
2️⃣ Extract the ZIP File

Right-click the downloaded .zip file

Click Extract All

Choose your preferred location

Open the extracted folder

3️⃣ Open the Project in VS Code

Open Visual Studio Code

Click File → Open Folder

Select the project folder

Click Select Folder

4️⃣ Open a Terminal in VS Code

Click Terminal → New Terminal

Make sure the terminal path is inside your project folder

Example:

Weather-Type-Classification>
5️⃣ (Recommended) Create a Virtual Environment

This avoids dependency conflicts.

python -m venv venv

Activate it:

Windows:

venv\Scripts\activate

Mac/Linux:

source venv/bin/activate
6️⃣ Install Required Dependencies

Run the following:

pip install -r requirements.txt

This will install the required libraries such as:

Flask

PyTorch (torch)

pandas

scikit-learn

joblib

7️⃣ (IMPORTANT) Train the Model First

Before running the app, you need to generate the model and preprocessing files.

python train_dl_model.py

This will create:

weather_model.pth

scaler.pkl

label_encoder.pkl

Make sure these files exist before proceeding.

8️⃣ Run the Flask Application
python app.py

If successful, you should see:

Running on http://127.0.0.1:5000/
9️⃣ Open the Web Application

Open your browser

Go to:

http://127.0.0.1:5000/

Input weather data and view predictions

Troubleshooting 🔧

Python not recognized

Make sure Python is installed and added to PATH

pip not working

python -m pip install -r requirements.txt

Missing files error (VERY COMMON)

Make sure you ran:

python train_dl_model.py

Port 5000 already in use

Close other running Flask apps

## Technologies Used 💻

- Python for machine learning model development
- Flask for web application framework
- HTML/CSS for front-end design
- Various machine learning libraries for model training and prediction

## Contributions Welcome 🤝

If you're passionate about data science, machine learning, or web development, feel free to contribute to this project by submitting pull requests or opening issues. Your expertise and insights can help enhance the functionality and usability of this Weather Type Classification Web App.

## About the Author 🌟

This project is maintained by a team of dedicated developers and data scientists who are committed to making machine learning more accessible and relevant in real-world applications, our goal is to empower individuals with the tools and knowledge to leverage technology for solving everyday challenges.

---

Stay tuned for future updates and improvements to the Weather Type Classification Web Application. Start exploring the world of weather predictions and classification today! 🌦️⛅⚡❄️


