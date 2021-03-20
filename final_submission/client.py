import json, numpy as np
import requests

USEFUL = [5, 7, 8, 9, 10]
SUBMISSION = [0, 0, 0, 0]
BASE = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -8.708335186857455e-16,
    0.0,
    1.419890139874561e-05,
    -1.0044704173757682e-06,
    -5.385216397608063e-09,
    3.7988126119265983e-10,
]


API_ENDPOINT = "http://10.4.21.156"
MAX_DEG = 11
# SECRET_KEY = "0suppMDvWimbxGKVY7BzOIjh65t1I55r64Mj6N0NDMgabOE28E"
SECRET_KEY = "rrmQj2DT1EwULu26UxrqMCvj5NuJL3BTE1Mi3qtGDU6gD7X50V"


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


def getErrors(vector, once=True, id=SECRET_KEY):
    # for i in vector:
    #     assert 0 <= abs(i) <= 10

    # newVector = list(BASE)

    # for index, newIndex in enumerate(USEFUL):
    #    newVector[newIndex] = vector[index]

    # assert len(newVector) == MAX_DEG

    if once:
        err = np.array(json.loads(sendRequest(id, vector, "geterrors")))
        print("Train and Test Err: ", err)
        return
    return json.loads(sendRequest(id, vector, "geterrors"))


def testErrors(vector):
    for i in vector:
        assert 0 <= abs(i) <= 10

    newVector = list(BASE)

    for index, newIndex in enumerate(USEFUL):
        newVector[newIndex] = vector[index]

    assert len(newVector) == MAX_DEG

    return [0, 0]


def submit(vector, id=SECRET_KEY):
    # for i in vector:
    # assert 0 <= abs(i) <= 10

    # newVector = list(BASE)

    # for index, newIndex in enumerate(USEFUL):
    # newVector[newIndex] = vector[index]

    # assert len(newVector) == MAX_DEG

    print((sendRequest(id, vector, "submit")) + ", rank below")


if __name__ == "__main__":
    vec = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1e-8,
        4e-9,
        -5e-11,
    ]
    print(vec)
    getErrors(vec)
    submit(vec)

    # count = 0
    # for i in data:
    #     print("----------- " + str(count) + " -------------")
    #     getErrors(i)
    #     submit(i)
    #     print("Vector: ", i)
    #     input()
    #     count += 1
