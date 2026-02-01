from matplotlib import pyplot as plt


reportFile = open("report.txt", "r")
N = []
isproportional = []
for line in reportFile.readlines():
    reportLine = line.split()
    N.append(int(reportLine[0]))
    isproportional.append(int(reportLine[1]))

# plt.hist(isproportional,bins = 2)
# plt.show()
plt.scatter(N, isproportional)
plt.ion()
plt.savefig("report.png")