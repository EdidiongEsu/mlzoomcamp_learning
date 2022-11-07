# Midterm project : Predicting Job change of data scientists

Source Data : https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists?taskId=3015

## Description of Data
Churn rate is a marketing metric that describes the number of customers who leave a business over a specific time period. . Every user is assigned a prediction value that estimates their state of churn at any given time. This value is based on:

User demographic information
Browsing behavior
Historical purchase data among other information
It factors in our unique and proprietary predictions of how long a user will remain a customer. This score is updated every day for all users who have a minimum of one conversion. 

## Features
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


## How to run bentoml service
To serve bentoml locally:

Use the code to run bento locally:
`bentoml serve service.py:svc`

## How to deploy
Use
`bentoml build`


## deploying the prediction service 
`bentoml containerize decision_tree:latest`

### run serve shows next
`docker run -it --rm -p 3000:3000 decision_tree:iugbmmc55opxvshc serve --production`

## High Performance serving: Locust
Run
`bentoml serve --production` and `locust -H http://localhost:3000` simultaneously in differently terminals. The former comes first.

## Deploy to heroku
Download heroku cli through this link:

https://devcenter.heroku.com/articles/heroku-cli#install-the-heroku-cli

and then open command line to type `heroku login -i`. Input your credentials. 
Login to the heroku container by typing
`heroku container:login` in your terminal.

create app by using `heroku create app_name` for example `heroku create decisionclassifier`

change directory to where the environment in the bentoml file is. To view the path type `bentoml list -o json` in terminal.

```python
$ bentoml list -o json[
  {
    "tag": "decision_tree:iugbmmc55opxvshc",
    "path": 
"~\\bentoml\\bentos\\decision_tree\\iugbmmc55opxvshc",
    "size": "704.66 KiB",
    "creation_time": "2022-11-06 16:55:35"
  }
```
