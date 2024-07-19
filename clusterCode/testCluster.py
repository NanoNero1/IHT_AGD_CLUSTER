print(__file__)

import os
cwd = os.getcwd()
print(cwd)

with open('readme.txt', 'w') as f:
    f.write('Create a new text file!')