import requests

base_url = "http://127.0.0.1:8000"

def base():
    response = requests.get(f"{base_url}")
    print(response.text)
response2 = requests.get('http://127.0.0.1:8000/p/fghj')
print(response2.text)

