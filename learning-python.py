# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#let's begin 
print("hello word")
print ('hello "python"')
print ("hello \"python\"")
'''
I can write comments here 
'''

# define values
name = "gizem"
age= 33

print(name)
print(type(name))
print(name, age, sep=",")#seperator
print(name, age, sep="-", end="***") # default end using is : "/n" 
print(name)

#input ()
name = input("name: ")
print(name)

#if else 
user="admin"
password="demo"

user2=inpurt("user name: ")


if user==user2:
    password2=inpurt("password: ")
    if password==password2:
        age=input("age: ")
        if age>=18:
            print("you can enter the system")
        else:
            print("you can not enter the system")
    else:
        print("you can not enter the system")
else:
    print("you can not enter the system")

    #easy way 
    
user="admin"
password="demo"

user2=inpurt("user name: ")
password2=inpurt("password: ")
age=input("age: ")

if user==user2 and password==password2 and age>=18:
    print ("you can enter the system")
else:
    print ("you can not enter the system")
    
    