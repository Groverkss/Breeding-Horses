vec = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    -1.86135069e-15,
    2.44330101e-05,
    -1.97038159e-06,
    -1.55953038e-08,
]


bold = [0] * 11

print("\\begin{bmatrix}")
for ind, i in enumerate(vec):
    vals = str(i).split("e")
    if len(vals) == 1:
        if bold[ind] == 1:
            print("\\mathbf{{{}}}\\\\".format(vals[0]))
        else:
            print("{}\\\\".format(vals[0]))
    else:
        if bold[ind] == 1:
            print(
                "\\mathbf{{{} \\times 10^{{{}}} }}\\\\".format(vals[0], vals[1])
            )
        else:
            print("{} \\times 10^{{{}}}\\\\".format(vals[0], vals[1]))
print("\\end{bmatrix}")
