import numpy as np
import matplotlib.pyplot as plt

'''example 1'''

data = np.random.normal(loc=0, scale=1, size=2000)

mean = np.mean(data)
var = np.var(data)

print(mean)
print(var)

plt.hist(data, bins=30, density=True, alpha=0.6, color='b')
plt.title('Histogram of Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

'''example 2'''
disease = 0.01
disease_positive_test = 0.99
non_disease_positive_test = 0.01

positive_test_probability_of_disease = (disease_positive_test * disease) / (disease_positive_test * disease + (1 - disease) * non_disease_positive_test)

print(positive_test_probability_of_disease)
