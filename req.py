import requests

url = "https://myagri.tech/apitimer.php?id=N0060&dkey=e3rokC0N1E&tmedit&hour=15.3&feeder=3"
response = requests.get(url)

print("Status Code:", response.status_code)
print("Response Text:", response.text)