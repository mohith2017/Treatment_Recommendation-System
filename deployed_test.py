import requests
import json
from azureml.core import Webservice
from azureml.core import Workspace

# URL for the web service
scoring_uri = 'http://97d4b24f-8f41-46e7-b875-32d30bfb8deb.westus2.azurecontainer.io/score'
ws=Workspace.get('Mohith_workspace', auth=None, subscription_id='c229fecc-7c30-489c-b79f-1de37cd2de58', resource_group='Mohith_group')
print(ws)

services = Webservice.list(ws)
print(services)
print(services[0].scoring_uri)
print(services[0].swagger_uri)

service=services[0]
print(service)

# Set the content type
headers = {'Content-Type': 'application/json'}


if service.auth_enabled:
    headers['Authorization'] = 'Bearer '+service.get_keys()[0]
elif service.token_auth_enabled:
    headers['Authorization'] = 'Bearer '+service.get_token()[0]
# If the service is authenticated, set the key or token
key = service.get_keys()[0]

# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Two sets of data to score, so we get two results back
data = {"data":
        [
            [
                0
            ],
            [
                1]
        ]
        }
# Convert to JSON string
input_data = json.dumps(data)


print(input_data)
# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.text)

# import requests
# import json
# service = 'testmodel1'
#
# headers = {'Content-Type': 'application/json'}
#
# if service.auth_enabled:
#     headers['Authorization'] = 'Bearer '+service.get_keys()[0]
# elif service.token_auth_enabled:
#     headers['Authorization'] = 'Bearer '+service.get_token()[0]
#
# print(headers)
#
# test_sample = json.dumps({'data': [
#     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
# ]})
#
# response = requests.post(
#     service.scoring_uri, data=test_sample, headers=headers)
# print(response.status_code)
# print(response.elapsed)
# print(response.json())
