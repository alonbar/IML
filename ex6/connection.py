import math
people=['Augustus','Violet','Mike','Joe']
links=[ ('Mike', 'Joe'),
 ('Violet', 'Augustus')]



def crosscount(v):
 # Convert the number list into a dictionary of person:(x,y)
 loc=dict([(people[i],(v[i*2],v[i*2+1])) for i in range(0,len(people))])
 total=0
 # Loop through every pair of links
 for i in range(len(links)):
     for j in range(i+1,len(links)):
         # Get the locations
         (x1,y1),(x2,y2)=loc[links[i][0]],loc[links[i][1]]
         (x3,y3),(x4,y4)=loc[links[j][0]],loc[links[j][1]]
         den=(y4-y3)*(x2-x1)-(x4-x3)*(y2-y1)
         # den==0 if the lines are parallel
         if den==0:
             continue
         # Otherwise ua and ub are the fraction of the
         # line where they cross
         ua=((x4-x3)*(y1-y3)-(y4-y3)*(x1-x3))/den
         ub=((x2-x1)*(y1-y3)-(y2-y1)*(x1-x3))/den
         # If the fraction is between 0 and 1 for both lines
         # then they cross each other
         if ua>0 and ua<1 and ub>0 and ub<1:
            total+=1
 return total

if __name__ == "__main__":
    v = [1,0,0,1,0,0,1,1]
    toatl = crosscount(v)
    print (toatl)
