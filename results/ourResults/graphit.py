import argparse
import matplotlib.pyplot as plt
import numpy as np
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--start',default=6)
	parser.add_argument('--end',default=11)
	args = parser.parse_args()

	count=int(args.start)
	end=int(args.end)
	figNum=0
	envList=['AntMuJoCoEnv-v0' ,'LunarLanderContinuous-v2','Walker2DMuJoCoEnv-v0' ,'HopperPyBulletEnv-v0' ,'HalfCheetahMuJoCoEnv-v0','HumanoidPyBulletEnv-v0']
	fig, axs = plt.subplots(2, 3)

	for env in envList:
		count=int(args.start)
		end=int(args.end)
		dataS={}
		while count < end:
			f = open(env+'scores'+str(count)+'SD3'+'.txt', 'r')
			for line in f:
				info=line.split()
				if not int(info[0]) in dataS:
					dataS[int(info[0])]=[float(info[1])]
				else:
					dataS[int(info[0])].append(float(info[1]))
			count+=1
		xvalS=[]
		avgS=[]
		devS=[]
		topS=[]
		botS=[]
		for key in dataS:
			avg=sum(dataS[key])/len(dataS[key])
			acc=0
			for pnt in dataS[key]:
				acc+=(pnt-avg)**2
			acc/=len(dataS[key])
			acc=np.sqrt(acc)
			avgS.append(avg)
			devS.append(acc)
			xvalS.append(key)
			topS.append(avg+acc)
			botS.append(avg-acc)
		count=int(args.start)
		end=int(args.end)
		dataT={}
		while count < end:
			f = open(env+'scores'+str(count)+'TD3'+'.txt', 'r')
			for line in f:
				info=line.split()
				if not int(info[0]) in dataT:
					dataT[int(info[0])]=[float(info[1])]
				else:
					dataT[int(info[0])].append(float(info[1]))
			count+=1
		xvalT=[]
		avgT=[]
		devT=[]
		topT=[]
		botT=[]
		for key in dataT:
			avg=sum(dataT[key])/len(dataT[key])
			acc=0
			for pnt in dataT[key]:
				acc+=(pnt-avg)**2
			acc/=len(dataT[key])
			acc=np.sqrt(acc)
			avgT.append(avg)
			devT.append(acc)
			xvalT.append(key)
			topT.append(avg+acc)
			botT.append(avg-acc)
		ind1=figNum//3
		ind2=figNum%3
		print(ind1,ind2)
		axs[ind1,ind2].set_title(env)

		axs[ind1,ind2].fill_between(xvalT,avgT,topT,color='pink')
		axs[ind1,ind2].fill_between(xvalT,botT,avgT,color='pink')

		axs[ind1,ind2].fill_between(xvalS,avgS,topS,color='cornflowerblue')
		axs[ind1,ind2].fill_between(xvalS,botS,avgS,color='cornflowerblue')

		axs[ind1,ind2].plot(xvalT,avgT,color='r',label='TD3')
		axs[ind1,ind2].plot(xvalS,avgS,color='b',label='SD3')
		figNum+=1

	for ax in axs.flat:
		ax.set(xlabel='Steps', ylabel='Return')
	handles, labels = ax.get_legend_handles_labels()
	fig.legend(handles, labels, loc='upper center')
	plt.show()
