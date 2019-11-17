import os


file_names = ['Celegans.txt','facebook.txt','NS.txt','PB.txt','Power.txt','Router.txt','USAir.txt','Yeast.txt']
for i in range(len(file_names)):
    file_names[i] = os.path.join('data', file_names[i])

file_command = ','.join(file_names)
# os.system('python main.py --input {0} --goal auc'.format(file_command))
os.system('python main.py --input {0} --goal acc --at 0'.format(file_command))
ats = [0, 50, 100, 200]
for at in ats:
    os.system('python main.py --input {0} --goal acc --at {1}'.format(file_command, at))
