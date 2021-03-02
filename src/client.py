import json
import requests

USEFUL = [5, 7, 8, 9, 10]
BASE = [
    0.0,
    0.0,
    -10,
    0.0,
    0.0,
    -1.83669770e-15,
    0.0,
    2.29435303e-05,
    -2.04721003e-06,
    -1.59792834e-08,
    9.98214034e-10,
]

API_ENDPOINT = "http://10.4.21.156"
MAX_DEG = 11
SECRET_KEY = "0suppMDvWimbxGKVY7BzOIjh65t1I55r64Mj6N0NDMgabOE28E"


def urljoin(root, path=""):
    if path:
        root = "/".join([root.rstrip("/"), path.rstrip("/")])
    return root


def sendRequest(id, vector, path):
    api = urljoin(API_ENDPOINT, path)
    vector = json.dumps(vector)
    response = requests.post(api, data={"id": id, "vector": vector}).text
    if "reported" in response:
        print(response)
        exit()

    return response


def getErrors(vector, id=SECRET_KEY):
    for i in vector:
        assert 0 <= abs(i) <= 10
    assert len(vector) == MAX_DEG

    newVector = list(BASE)

    for newIndex, index in enumerate(USEFUL):
        newVector[newIndex] = index

    return json.loads(sendRequest(id, vector, "geterrors"))
