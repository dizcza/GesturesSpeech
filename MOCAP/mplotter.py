# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np

zipped = [(1e-06, 0.17508874776345451, 0.0647448), (0.0001, 0.1750887441645072, 0.0647448), (0.01, 0.17508838448171357, 0.0647446), (0.1, 0.1750851147626187, 0.0647432), (1.0, 0.17505242793231682, 0.0647286), (10.0, 0.1747263457118941, 0.0645836), (100.0, 0.17160113631910878, 0.0632738), (1000.0, 0.15821415968124972, 0.0593719)]
betas, bvars, stds = np.array(zip(*zipped))

# plt.errorbar(np.log(betas), bvars, stds, marker='^', ms=8)
plt.plot(np.log(betas), bvars, linestyle='-', marker='^', ms=8)
plt.xlabel("log(beta)")
rstd_mean = np.average(stds / bvars) * 100.
plt.ylabel("Db")
plt.legend(["between variance Db\nstd=%.1f%%" % rstd_mean], loc=3, numpoints=1)
plt.title("Choosing the best beta")
plt.grid()

plt.savefig("choosing_beta.png")
# plt.show()
