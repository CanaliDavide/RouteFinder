Place the input file in the folder and name it "input.dat"  

The imput file must have the following syntax:

param (name) :=	(value) ;
param (name) :=	(value) ;
param (name) :=	(value) ;
param (name) :=	(value) ;
param (name) :=	(value) ;

param: Cx Cy Dc usable	:=
1\t9\t22\t4\t1\t
*for each node*
\t;

where name should be: n,range,Vc,Fc,capacity
between each value and after the last one in every row of the matrix there's should be a tab

Then double-click on the "Find_Solution.bat" file and wait untill the computation is finished (5 min average)

In the same folder there will be 3 new files:
	output.txt : containts the values of the solution as specified in the email
	output_formatted.txt : containts the values of the solution but formatted so that they are more human readable
	route.png : is a image that show all the routes and the chosen markets


** Minimart50 and 100 solutions are respectively in the Solution_Minimart50 and 100 folders **