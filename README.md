# nalu
Implementation of Neural Arithmetic Logical Units for Linear Transformations and Multiplications
NALU has been implemented to run the following tasks:

1st Task  
Suppose there is a list of 6 numbers given by [6,5,4,3,2,1].  
This gives the solutions as (6-5) + (4-3) + (2-1) which comes to 1 + 1 + 1 = 3.  

2nd Task  
Suppose there is a list of 4 numbers given as [1,2,3,4].  
This gives solution as 1*2*3*4 which gives a result as 24.  

Data folder has three files - data6.csv, data8.csv, data10.csv are files that contain the training and testing set.  
P.S. - data6.csv, data8.csv and data10.csv are big files
NALU_LT.py does the first task  
NALU_MUL.py does the second task  
Make sure that data6, data8, data10 are in the same folder from where you are running the programs else you can edit and insert
the path in the code at the appropriate line.

To run the codes you need to have the following:-
1. TensorFlow - Code uses TensorFlow version 1.x but if you have TensorFlow 2.x then I have used compat.v1 in the code, so that
should solve the problem.
2. NumPy
3. Python 3.x

###_______________________________________________________________________________________________________________###

