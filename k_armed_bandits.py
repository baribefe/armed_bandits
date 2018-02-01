import numpy as np
import matplotlib.pyplot as plt
import random as rand

def k_armed_softmax(k,N,M,temperature):

    bandits = np.random.normal(5.0,1.0,size=(N,k))
    avg_reward = []
    indexes = range(k)

    for i in range(N):

        rewards = []
        q_k = np.zeros(k) 
        counts = np.zeros(k)
        bandit = bandits[i]

        for j in range(M):

            q_values = np.exp(q_k/temperature)/(np.exp(q_k/temperature).sum())
            arm_to_sample = np.random.choice(indexes,p=q_values) 
            s_prime = bandit[arm_to_sample]
            reward = np.random.normal(s_prime,1.0)
            rewards.append(reward)
            counts[arm_to_sample] += 1
            q_k[arm_to_sample] = q_k[arm_to_sample] + (1.0/counts[arm_to_sample])*(reward-q_k[arm_to_sample])
        avg_reward.append(rewards)
        print q_k
    return np.mean(avg_reward,axis=0),q_k

def k_armed_epsilon_greedy2(k,epsilon,N,M):

    bandits = np.random.normal(5.0,1.0,size=(N,k))
    avg_reward = []
    alpha=0.01
    for i in range(N):

        rewards = []
        q_k = np.zeros(k)
        counts = np.zeros(k)
        bandit = bandits[i]

        for j in range(M):
            if rand.random() < epsilon:
                arm_to_sample = rand.randint(0, k-1)
            else:
                arm_to_sample = np.argmax(q_k)
            s_prime = bandit[arm_to_sample]
            reward = np.random.normal(s_prime,1.0)
            rewards.append(reward)
            counts[arm_to_sample] += 1
            q_k[arm_to_sample] = q_k[arm_to_sample] + (1.0/counts[arm_to_sample])*(reward-q_k[arm_to_sample])
        avg_reward.append(rewards)
    return np.mean(avg_reward,axis=0), q_k


r,Q = k_armed_softmax(100,2000,50000,0.3)
print "Mean Reward: ", np.mean(r)

plt.plot(r)
plt.xlim([-1000,50000])
plt.xlabel("Number of Pulls")
plt.ylabel('Average Reward')
plt.title(r'Softmax with $\mathrm{\tau}$ = 0.3 for 100 Bandits', family='sans-serif',size='18',stretch='ultra-condensed',color='r')
plt.savefig('softmax_0.3.pdf')
plt.savefig('softmax_0.3.png')

