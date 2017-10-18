import os

cwd = os.getcwd()
imgwd = os.path.join(cwd, "IDPhotos")
with open('./IDPhotos/IDPhotos.txt', 'w') as txt:
	for jpg in os.listdir(imgwd):
		if jpg.split('.')[-1] not in ['jpg', 'PNG', 'png', 'JPEG']:
			continue
		person = jpg.split('.')[0]
		path = os.path.join(imgwd, jpg)
		line = person + ' ' + path + '\n'
		print('writing {0}, location is {1}'.format(person, path))
		txt.write(line)

