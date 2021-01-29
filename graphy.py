import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import random

#https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
def mean_confidence_interval(data, confidence=0.68):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


tf0 = np.load("td3_flagrun_0.npy").reshape((1, 333))
tf1 = np.load("td3_flagrun_1.npy").reshape((1, 333))
tf2 = np.load("td3_flagrun_2.npy").reshape((1, 333))

td3 = np.concatenate((np.concatenate((tf0, tf1), axis=0), tf2), axis=0)

avg_td3 = np.mean(td3, axis=0)
plus = avg_td3 + np.std(td3, axis=0)
minus = avg_td3 - np.std(td3, axis=0)

sf0 = np.load("sd3_flagrun_0.npy").reshape((1, 333))
sf1 = np.load("sd3_flagrun_1.npy").reshape((1, 333))
sf2 = np.load("sd3_flagrun_2.npy").reshape((1, 70))

sd31 = np.concatenate((np.concatenate((sf0[0][:70].reshape((1, 70)), sf1[0][:70].reshape((1, 70))), axis=0), sf2), axis=0)

sd32 = np.concatenate((sf0[0][70:].reshape((1, 263)), sf1[0][70:].reshape((1, 263))), axis=0)

avg_sd31 = np.mean(sd31, axis=0)
plus_s1 = avg_sd31 + np.std(sd31, axis=0)
minus_s1 = avg_sd31 - np.std(sd31, axis=0)

avg_sd32 = np.mean(sd32, axis=0)
plus_s2 = avg_sd32 + np.std(sd32, axis=0)
minus_s2 = avg_sd32 - np.std(sd32, axis=0)

avg_sd3 = np.concatenate((avg_sd31, avg_sd32))
plus_s = np.concatenate((plus_s1, plus_s2))
minus_s = np.concatenate((minus_s1, minus_s2))

x = np.load("frames.npy") / 1e6
fig, ax = plt.subplots()

ax.plot(x, avg_td3, label='TD3', color='red')
ax.fill_between(x, (minus), (plus), color='pink')


ax.plot(x, avg_sd3, label='SD3', color='blue')
ax.fill_between(x, (minus_s), (plus_s), color='cornflowerblue')

plt.legend(loc='lower right')
plt.title("Humanoid Flagrun")
plt.ylabel('Average Reward')
plt.xlabel('Million Steps')
plt.show()


tf2 = np.load("td3_pendulum_0.npy").reshape((1, 160))
tf1 = np.load("td3_pendulum_1.npy").reshape((1, 333))
tf0 = np.load("td3_pendulum_2.npy").reshape((1, 333))

td31 = np.concatenate((np.concatenate((tf0[0][:160].reshape((1, 160)), tf1[0][:160].reshape((1, 160))), axis=0), tf2), axis=0)

td32 = np.concatenate((tf0[0][160:].reshape((1, 173)), tf1[0][160:].reshape((1, 173))), axis=0)

avg_td31 = np.mean(td31, axis=0)
plus1 = avg_td31 + np.std(td31, axis=0)
minus1 = avg_td31 - np.std(td31, axis=0)

avg_td32 = np.mean(td32, axis=0)
plus2 = avg_td32 + np.std(td32, axis=0)
minus2 = avg_td32 - np.std(td32, axis=0)

avg_td3 = np.concatenate((avg_td31, avg_td32))
plus = np.concatenate((plus1, plus2))
minus = np.concatenate((minus1, minus2))

sf2 = np.load("sd3_pendulum_0.npy").reshape((1, 100))
sf1 = np.load("sd3_pendulum_1.npy").reshape((1, 333))
sf0 = np.load("sd3_pendulum_2.npy").reshape((1, 333))

sd31 = np.concatenate((np.concatenate((sf0[0][:100].reshape((1, 100)), sf1[0][:100].reshape((1, 100))), axis=0), sf2), axis=0)

sd32 = np.concatenate((sf0[0][100:].reshape((1, 233)), sf1[0][100:].reshape((1, 233))), axis=0)

avg_sd31 = np.mean(sd31, axis=0)
plus_s1 = avg_sd31 + np.std(sd31, axis=0)
minus_s1 = avg_sd31 - np.std(sd31, axis=0)

avg_sd32 = np.mean(sd32, axis=0)
plus_s2 = avg_sd32 + np.std(sd32, axis=0)
minus_s2 = avg_sd32 - np.std(sd32, axis=0)

avg_sd3 = np.concatenate((avg_sd31, avg_sd32))
plus_s = np.concatenate((plus_s1, plus_s2))
minus_s = np.concatenate((minus_s1, minus_s2))

fig, ax = plt.subplots()

ax.plot(x, avg_td3, label='TD3', color='red')
ax.fill_between(x, (minus), (plus), color='pink')


ax.plot(x, avg_sd3, label='SD3', color='blue')
ax.fill_between(x, (minus_s), (plus_s), color='cornflowerblue')

plt.legend(loc='lower right')
plt.title("Pendulum")
plt.ylabel('Average Reward')
plt.xlabel('Million Steps')
#plt.ylim(-200, -120)
plt.show()


tf0 = np.load("td3_inpen_0.npy").reshape((1, 333))
tf1 = np.load("td3_inpen_1.npy").reshape((1, 333))
tf2 = np.load("td3_inpen_2.npy").reshape((1, 333))

td3 = np.concatenate((np.concatenate((tf0, tf1), axis=0), tf2), axis=0)


avg_td3 = np.mean(td3, axis=0)
plus = avg_td3 + np.std(td3, axis=0)
minus = avg_td3 - np.std(td3, axis=0)

sf0 = np.load("sd3_inpen_0.npy").reshape((1, 333))
sf1 = np.load("sd3_inpen_1.npy").reshape((1, 333))
sf2 = np.load("sd3_inpen_1.npy").reshape((1, 333))

sd3 = np.concatenate((np.concatenate((sf0, sf1), axis=0), sf2), axis=0)

avg_sd3 = np.mean(sd3, axis=0)
plus_s = avg_sd3 + np.std(sd3, axis=0)
minus_s = avg_sd3 - np.std(sd3, axis=0)

fig, ax = plt.subplots()

ax.plot(x, avg_td3, label='TD3', color='red')
ax.fill_between(x, (minus), (plus), color='pink')


ax.plot(x, avg_sd3, label='SD3', color='blue')
ax.fill_between(x, (minus_s), (plus_s),  color='cornflowerblue')

plt.legend(loc='lower right')
plt.title("Double Inverted Pendulum")
plt.ylabel('Average Reward')
plt.xlabel('Million Steps')
plt.show()
