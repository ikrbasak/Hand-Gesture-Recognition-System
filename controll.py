import pyautogui as p

# define the function
def controll(data):
	x, y = p.position()
	n=p.onScreen(x,y)
	if (n == True):
		if (data == 0):
			# Add task 1
		elif (data == 1):
			# Add task 2
		elif (data == 2):
			# Add task 3
		elif (data == 3):
			# Add task 4
		elif (data == 4):
			# Add task 5
		elif (data == 5):
			# Add task 6
		else:
			# Add task 7