import subprocess

use_prox = False
iter_repeat = 200
EPOCHS = 5
lr=1e-2

# Remotely run the training code with arguments
# (remotely, progress is printed and new weights are saved to a file)
print("remotely training!")
python_path = "/home/rverkuil/integration/integration/bin/python"
script_path = "/home/rverkuil/integration/drake-pytorch/python/remote_train.py"
sub_cmd = python_path+" "+script_path+" {} {} {} {}".format(int(use_prox), iter_repeat, EPOCHS, lr)
#p = subprocess.Popen(['ssh','RLG',sub_cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
p = subprocess.Popen(['stdbuf', '-o0']+['ssh','RLG',sub_cmd], bufsize=0, stdout=subprocess.PIPE)
#while p.poll() is None:
#    l = p.stdout.readline() # This blocks until it receives a newline.
#    print(l)
## When the subprocess terminates there might be unconsumed output 
## that still needs to be processed.
#print(p.stdout.read())

import sys
while True:
    #out = p.stderr.read(1)
    out = p.stdout.read(1)
    if out == '' and p.poll() != None:
        break
    if out != '':
        sys.stdout.write(out)
        sys.stdout.flush()

#for line in iter(p.stdout.readline, b''):
#    print(">>> " + line.rstrip())
