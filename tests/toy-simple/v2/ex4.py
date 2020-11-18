i=0
# Deletion of a single statement --> leads to ambiguity in log placement 
seq = []
def fun():
        if i == 0:
	        return seq
fun()
