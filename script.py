import os

file_names = ['arxiv.txt','Celegans.txt','facebook.txt','NS.txt','PB.txt','Power.txt','Router.txt','USAir.txt','Yeast.txt']
for i in range(len(file_names)):
    file_names[i] = os.path.join('data', file_names[i])

file_command = ','.join(file_names)
os.system('python main.py --input {0}'.format(file_command))