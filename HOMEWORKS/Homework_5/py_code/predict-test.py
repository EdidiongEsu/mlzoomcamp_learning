import requests

# n/b:host is open to the world

url = 'http://localhost:9696/predict'

client_id = 'xyz-123'
client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}

response = requests.post(url, json=client).json()
print(response)

if response['churn'] == True:
    print("sending promo email to %s" % client_id)
else:
    print("not sending promo email to %s" % client_id)
