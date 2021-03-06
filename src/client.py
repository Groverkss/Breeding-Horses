import json, numpy as np
import requests
import sys

USEFUL = [7, 8, 9, 10]
# BASE = [
#     0.0,
#     0.0,
#     -10,
#     0.0,
#     0.0,
#     -1.83669770e-15,
#     0.0,
#     2.29435303e-05,
#     -2.04721003e-06,
#     -1.59792834e-08,
#     9.98214034e-10,
# ]

BASE = [
    0,
    0,
    0,
    0,
    0,
    -0.40000000e-15,
    0,
    0,
    0,
    0,
    0,
]

# SUBMISSION = [
#     -1.93765161e-15,
#     2.29226307e-05,
#     -2.04984573e-06,
#     -1.59780183e-08,
#     9.96132235e-10,
# ]

# SUBMISSION = [
#     -1.93727666e-15,
#     2.29207672e-05,
#     -2.27026207e-06,  # Saturated
#     -1.59742729e-08,
#     7.80416598e-10,  # Saturated
# ]

# SUBMISSION = [
#     -1.93870183e-15,
#     2.09031834e-05,
#     -2.26965921e-06,
#     -1.59687922e-08,
#     7.80643074e-10,
# ]

# SUBMISSION = [
#     -1.93881007e-15,
#     2.09078302e-05,
#     -2.27063900e-06,
#     -1.59706887e-08,
#     7.80278911e-10,
# ]

# SUBMISSION = [
#     -1.93908168e-15,
#     2.09240884e-05,
#     -2.26855349e-06,
#     -1.59632136e-08,
#     7.81269165e-10,
# ]


SUBMISSION = [2.41555638e-05, -1.90014724e-06, -1.71141768e-08, 9.01109219e-10]

API_ENDPOINT = "http://10.4.21.156"
MAX_DEG = 11
SECRET_KEY = "0suppMDvWimbxGKVY7BzOIjh65t1I55r64Mj6N0NDMgabOE28E"
# SECRET_KEY = "rrmQj2DT1EwULu26UxrqMCvj5NuJL3BTE1Mi3qtGDU6gD7X50V"


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


def getErrors(vector, once=False, id=SECRET_KEY):
    for i in vector:
        assert 0 <= abs(i) <= 10

    newVector = list(BASE)

    for index, newIndex in enumerate(USEFUL):
        newVector[newIndex] = vector[index]

    assert len(newVector) == MAX_DEG

    if once:
        err = np.array(json.loads(sendRequest(id, newVector, "geterrors")))
        print(err)
        print(np.array([np.sum(err)]))
        sys.exit(0)
    return json.loads(sendRequest(id, newVector, "geterrors"))


def testErrors(vector, once=False, id=SECRET_KEY):
    for i in vector:
        assert 0 <= abs(i) <= 10

    newVector = list(BASE)

    for index, newIndex in enumerate(USEFUL):
        newVector[newIndex] = vector[index]

    assert len(newVector) == MAX_DEG

    return [0, 0]


def submit(vector=SUBMISSION, id=SECRET_KEY):
    for i in vector:
        assert 0 <= abs(i) <= 10

    newVector = list(BASE)

    for index, newIndex in enumerate(USEFUL):
        newVector[newIndex] = vector[index]

    assert len(newVector) == MAX_DEG

    print(np.array(json.loads(sendRequest(id, newVector, "submit"))))


if __name__ == "__main__":
    submit()
