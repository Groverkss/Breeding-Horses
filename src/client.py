import json, numpy as np
import requests

USEFUL = [5, 7, 8, 9, 10]
SUBMISSION = [0, 0, 0, 0]
BASE = [
    0.0,
    -1.5453559231046635e-12,
    -4.342298667696394e-13,
    1.6287026302828268e-10,
    -1.349365591502193e-10,
    -8.708335186857455e-16,
    8.821994522452975e-16,
    1.419890139874561e-05,
    -1.0044704173757682e-06,
    -5.385216397608063e-09,
    3.7988126119265983e-10,
]

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

    vector = [
        0,
        0,
        0,
        0,
        0,
        -0.9e-15,
        0,
        2.605e-05,  # down abs - done
        -1.88e-06,  # down abs - done
        -1.106e-08,  # down abs
        7.648e-10,  # up abs
    ]

    vector = [
        0,
        0,
        0,
        0,
        0,
        -0.45e-15,
        0,
        2.51e-05,  # down abs - done
        -1.88e-06,  # down abs - done
        -1.106e-08,  # down abs
        7.648e-10,  # up abs
    ]

    data = [
        [
            0.0,
            -1.5453559231046635e-12,
            -4.342298667696394e-13,
            1.6287026302828268e-10,
            -1.349365591502193e-10,
            -8.708335186857455e-16,
            8.821994522452975e-16,
            1.419890139874561e-05,
            -1.0044704173757682e-06,
            -5.385216397608063e-09,
            3.7988126119265983e-10,
        ],
        [
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
        ],
    ]

    vector = [
        0.00000000e00,
        -1.54649393e-12,
        -4.36115012e-13,
        1.63062632e-10,
        -1.34997781e-10,
        -8.67386191e-16,
        8.82877935e-16,
        1.42284786e-05,
        -1.00685242e-06,
        -5.38474558e-09,
        3.78988721e-10,
    ]
    vec = [
        0,
        0,
        0,
        0,
        0,
        -5.77e-16,
        0,
        1.52294786e-05,
        -1.03985242e-06,
        -5.4574558e-09,
        3.78988721e-10,
    ]
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

# BASE = [
#     0, 0, 0, 0, 0, -0.40000000e-15, 0,
#     0,
#     0,
#     0,
#     0,
# ]

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
# SUBMISSION = [-1.10741253e-08, 7.72494357e-10]
