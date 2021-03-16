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

    data = [
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -5.74726809e-16,
            0.0,
            1.52226779e-05,
            -1.04623995e-06,
            -5.47552982e-09,
            3.8344835e-10,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -5.77e-16,
            0.0,
            1.52294786e-05,
            -1.03985242e-06,
            -5.4574558e-09,
            3.78988721e-10,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -5.74726809e-16,
            0.0,
            1.52226779e-05,
            -1.04623995e-06,
            -5.47552982e-09,
            3.8344835e-10,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -5.74726809e-16,
            0.0,
            1.51827267e-05,
            -1.04623995e-06,
            -5.47549689e-09,
            3.83653816e-10,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -5.77e-16,
            0.0,
            1.52294786e-05,
            -1.03985242e-06,
            -5.4574558e-09,
            3.78988721e-10,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -5.74726809e-16,
            0.0,
            1.52226779e-05,
            -1.04623995e-06,
            -5.47552982e-09,
            3.8344835e-10,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -5.77e-16,
            0.0,
            1.52294786e-05,
            -1.03985242e-06,
            -5.4574558e-09,
            3.78988721e-10,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -5.77e-16,
            0.0,
            1.52294786e-05,
            -1.03985242e-06,
            -5.38474558e-09,
            3.78988721e-10,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -5.74726809e-16,
            0.0,
            1.52226779e-05,
            -1.04623995e-06,
            -5.47552982e-09,
            3.8344835e-10,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -5.77e-16,
            0.0,
            1.52294786e-05,
            -1.03985242e-06,
            -5.4574558e-09,
            3.78988721e-10,
        ],
    ]

    # getErrors(vec)
    # submit(vec)

    count = 0
    for i in data:
        print("----------- " + str(count) + " -------------")
        getErrors(i)
        submit(i)
        print("Vector: ", i)
        input()
        count += 1
