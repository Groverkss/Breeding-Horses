vec = [
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    -5.74980395e-16,
    0.00000000e00,
    1.50074509e-05,
    -1.03755096e-06,
    -5.46000000e-09,
    3.85718176e-10,
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
