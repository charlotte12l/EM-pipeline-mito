# Automatically generate slurm scripts for running inference on RC cluster 
import os, sys

Do = '/n/pfister_lab2/Lab/xingyu/Human/slurms/'
def get_pref(mem=150000, do_gpu=False):
    pref = '#!/bin/bash\n'
    pref+= '#SBATCH -N 1 # number of nodes\n'
    pref+= '#SBATCH -p cox\n'
    pref+= '#SBATCH -n 2 # number of cores\n'
    pref+= '#SBATCH --mem '+str(mem)+' # memory pool for all cores\n'
    if do_gpu:
        pref+= '#SBATCH --gres=gpu:2 # memory pool for all cores\n'
    pref+= '#SBATCH -t 3-00:00:00 # time (D-HH:MM)\n'
    pref+= '#SBATCH -o /n/pfister_lab2/Lab/xingyu/Human/Human_Outputs/logs/deploy_%j.log\n\n'
    pref+='module load Anaconda3/5.0.1-fasrc01\n'
    pref+='module load cuda/9.0-fasrc02 cudnn/7.4.1.5_cuda9.0-fasrc01\n'
    pref+= 'source activate py3_torch\n\n'
    return pref

cmd=[]
mem=50000
do_gpu= True

fn='deploy' # output file name
suf = '\n'
num = 5
#cn = 'deploy.py'
cn = 'deploy.py'
#cmd+=['python -u /n/pfister_lab2/Lab/xingyu/JWR/pytorch_connectomics/torch_connectomics/run/'+cn+' %d '+str(num)+suf]
cmd+=['python -u /n/pfister_lab2/Lab/xingyu/Human/pytorch_connectomics/torch_connectomics/run/'+cn+' %d '+str(num)+suf]

pref=get_pref(mem, do_gpu)
print('1')
if not os.path.exists(Do):
    os.makedirs(Do)
print('2')
for i in range(num):
    a=open(Do +'%02d.sh'%(i),'w')
    a.write(pref)
    for cc in cmd:
        if '%' in cc:
            a.write(cc%i)
        else:
            a.write(cc)
    a.close()
