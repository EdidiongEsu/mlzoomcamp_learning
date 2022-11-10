# Midterm project : Predicting churn in marketing

Source Data : https://www.kaggle.com/datasets/undersc0re/predict-the-churn-risk-rate

## Description of Data
-----------------------
Churn rate is a marketing metric that describes the number of customers who leave a business over a specific time period. . Every user is assigned a prediction value that estimates their state of churn at any given time. This value is based on:

User demographic information
Browsing behavior
Historical purchase data among other information
It factors in our unique and proprietary predictions of how long a user will remain a customer. This score is updated every day for all users who have a minimum of one conversion. 

## Features
----------------

| Column name                  | Description                                                              |
| ---------------------------- | ------------------------------------------------------------------------ |
| customer_id                  | Represents the unique identification number of a customer                |
| Name                         | Represents the name of a customer                                        |
| age                          | Represents the age of a customer                                         |
| security_no                  | Represents a unique security number that is used to identify a person    |
| region_category              | Represents the region that a customer belongs to                         |
| membership_category          | Represents the category of the membership that a customer is using       |
| joining_date                 | Represents the date when a customer became a member                      |
| joined_through referral      | Represents whether a customer joined using any referral code or ID       |
| referral_id                  | Represents a referral ID                                                 |
| preferred_offer types        | Represents the type of offer that a customer prefers                     |
| medium_of operation          | Represents the medium of operation that a customer uses for transactions |
| internet_option              | Represents the type of internet service a customer uses                  |
| last_visit time              | Represents the last time a customer visited the website                  |
| days_since last login        | Represents the no. of days since a customer last logged into the website |
| avg_time spent               | Represents the average time spent by a customer on the website           |
| avg_transaction value        | Represents the average transaction value of a customer                   |
| avg_frequency login days     | Represents the no. of times a customer has logged in to the website      |
| points_in wallet             | Represents the points awarded to a customer on each transaction          |
| used_special discount        | Represents whether a customer uses special discounts offered             |
| offer_application preference | Represents whether a customer prefers offers                             |
| past_complaint               | Represents whether a customer has raised any complaints                  |
| complaint_status             | Represents whether the complaints raised by a customer was resolved      |
| feedback                     | Represents the feedback provided by a customer                           |
| churn_risk score             | Represents the churn risk score that 0 or 1                              |

## Aim
Train multiple models to predict the churn of customers and pick the best model (taking into consideration the evaluation metrics). Deploy the machine learning service to the web.

## Approach
Machine learning service was bundled with bentoml and deployed to heroku.

<p align="center">
  <img width="460" height="300" src="https://github.com/EdidiongEsu/mlzoomcamp_learning/blob/main/projects/midterm_project/images/15.%20heroku_bento.png">
</p>

## Result
The service was successfully deployed and can be viewed through this link: https://decisionclassifier.herokuapp.com/.
                        ![](images/12.%20heroku_app.png)

You can test the service by clicking on post and iputting the data:

``` python
{
"age": 12,
"gender": "m",
"region_category": "missing",
"membership_category": "silver_membership",
"joined_through_referral": "no",
"preferred_offer_types": "credit/debit_card_offers",
"medium_of_operation": "smartphone",
"internet_option": "fiber_optic",
"last_visit_time": "13:38:19",
"days_since_last_login": 24.0,
"avg_time_spent": 121.82,
"avg_transaction_value": 30385.88,
"avg_frequency_login_days": 25.0,
"points_in_wallet": 698.62,
"used_special_discount": "no",
"offer_application_preference": "yes",
"past_complaint": "yes",
"complaint_status": "solved",
"feedback": "too_many_ads"
}

```
It will look this:
        ![](images/14.%20result1.png)

Click on execute to show the result.

## What the main files contain
  -----------------------
- `bentofile.yaml`:This contains the requirements for this project. 
- `service.py`: This contains the code used to create and serve bentoml. Where service is created.
- `Data Folder`: Contains the data used in the project.
- `midterm_project_EDA.ipynb`: This contains the experimental data analysis and model comparisons.
- `train.py`: Contains code required to engineer data and train model. In the script, there is also the code to create bentoml.
``` python
tag = bentoml.sklearn.save_model('decision_tree', dt, 
                                custom_objects = {
                                    "dictVectorizer":dv
                                }, signatures={
                                    "predict_proba":{
                                        "batchable": True,
                                        "batch_dim": 0
                                    }})
                                             
```
To replicate the whole project, check the next section.

- `locustfile.py`: Contains code for testing high performance of the created ml deployment.
  
## Steps to create (reproduce) project
---------------------------------------
### 1. Install all requirements
Create a folder and Install all the requirements in the requirements.txt file including bentoml
``` python
scikit-learn
pandas
pydantic
bentoml
```
### 2. Run train script
Run train.py script to train, create model and create bentoml tag. This can be done using `python train.py` in your terminal which might take around 2/3 minutes to run. Once it is done, the model tag and the path will be created and its information will be outputted in your terminal.
                                ![](images/1a.%20run_trainfile.png)

### 3. How to run bentoml service
Use the code to serve bento locally:
`bentoml serve service.py:svc` or `bentoml serve service.py:svc --reload` which automatically reloads when the service.py file is saved with additional data/code. 
Once this is done, this will show on your terminal:
      ![](images/1ab.%20bento%20serve%20code.png)
Open your browser to the address as shown in the terminal picture above. Click on try it out and input a sample JSON dictionary to test the serve (check EDA file).
        ![](images/1b.%20UI_execute1.png)
        ![](images/2.%20UI_execute2.png)
        ![](images/2b.%20UI_execute3.png)
                                                                                                          
### 4. Build your bento
Execute `bentoml build` and a bento will be built instantly. Your terminal will look like:
    ![](images/2c.%20bentoml%20build.png)
A new tag is created seperate from the tag created when the train.py file was ran. This is because bento creates a new tahg every instance and seperates model and bento tags.

### 5. Containerize and deploy the prediction service
Use `bentoml containerize decision_tree:latest` to containerize and deploy the bento production service.
    ![](images/3.%20terminal_containerize.png)

In the terminal, the instructions are shown on how to run serve.

### 6. Docker run serve
RUn the instruction shown after. For example:
`docker run -it --rm -p 3000:3000 decision_tree:5z7map3agcyfdshc serve --production` where `5z7map3agcyfdshc` is the name of the tag.
Once it is completed, go to the listed webad=ress (the one you opened earlier). You would notice that the tag is now beside the decison tree name.
        ![](image/../images/4.%20local_deployed.png)

You have successfully deployed your model service!

### 7. High Performance serving: Locust 
Locust helps to simulate influx of users. The user and spawn rate can be adjusted to see how the service would respond. This step is optional but ehlps understand bento as a process better.

Run
`bentoml serve --production` and `locust -H http://localhost:3000` simultaneously in differently terminals. The former comes first. Input how may Users and the spawn rate (users started per second. An example is 10 users at a spawn rate of 2. 
        ![](images/5.%20Locust%20spawn.png)
        ![](images/6.%20Locust%20spawn2.png)
Charts can also be viewed on Locust:
        ![](images/7.%20Locust%20chart.png)


### 9. Deploy to heroku
The web service can be deployed to the web through heroku. If you are not familiar with heroku, you can learn more about them, here: 
https://www.heroku.com/. Sign up and create an account on the platform to be able to access heroku cli which will make the deployment easier.

### 10. Download Heroku CLI
Download heroku cli for your OS through this link:
https://devcenter.heroku.com/articles/heroku-cli#install-the-heroku-cli
        ![](images/7b.%20heroku%20web.png)

and then open command line to type `heroku login -i`. Input your credentials. For windows, command line prompt interacts well with heroku.

### 11. Login to heroku container
Login to the heroku container by typing `heroku container:login` in your terminal. AFter login, you will receive a message like below:
                ![](images/8.%20heroku_container.png)

### 12. Create a new app
Create a new app so that the service can be deployed to it. If you have already created an app, you can skip to the next step. Create app by using `heroku create app_name` for example `heroku create decisionclassifier`.
        ![](images/9.%20heroku_create_app.png)

### 13. Change the working directory
Change directory to where the environment in the bentoml file is. To view the path type `bentoml list -o json` in terminal.

```python
$ bentoml list -o json
[
  {
    "tag": "decision_tree:iugbmmc55opxvshc",
    "path": 
"~\\bentoml\\bentos\\decision_tree\\iugbmmc55opxvshc",
    "size": "704.66 KiB",
    "creation_time": "2022-11-06 16:55:35"
  }
```
After changing your working environment, move to the Docker Folder to access the docker file. Use `cd env\docker` to change to that directory.

### 14. Tag your model with Docker
Execute `docker tag f547144181b6 registry.heroku.com/decisionclassifier/web` where `f547144181b6` is the image ID of the docker image. Run  `docker images` to get the list of your recent docker images and their IDS. The first one will be the most recent image and the third column should contain the image ID.
            ![](images/9b.%20docker_images.png)

### 15. Docker push to web
Run `docker push registry.heroku.com/decisionclassifier/web` tp push images to the service
        ![](images/10.%20docker_push.png)

### 16. Release images into Heroku
Run `heroku container:release web --app decisionclassifier` to releaase images to the web : Heroku.
        ![](images/11.%20heroku_push.png)

## 17. Deploy ML service.
To interact with the application, Visit the web address https://{app_name}.herokuapp.com/.
        ![](images/12.%20heroku_app.png)
        ![](images/13.%20heroku_app2.png)