print("hello")
import pymongo
import datetime
import random
from pymongo import MongoClient
from pprint import pprint
import datetime
from bson import ObjectId
import bson
from collections import Counter
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
from plotly.graph_objects import bar
import plotly.graph_objects as go
import plotly.figure_factory as ff
from matplotlib import pyplot as plt
import scipy
from plotly.offline import plot
from scipy.stats import gaussian_kde
from statsmodels.graphics.gofplots import qqplot
import plotly.express as px
import ast
import pandas as pd
import plotly.io as pio
from IPython.display import Image
import os
textfiles_path_laptop = "C:/Users/Anthony/OneDrive/Desktop/Assignment to find user who matched in two or more sessions/Second run with abs(diff)/"
textfiles_path ="C:/Users/antho/OneDrive/Desktop/Assignment to find user who matched in two or more sessions/Second run with abs(diff)/"

# Database info
URI = "mongodb+srv://gametreeuser:7eYe7WpganG3zfWy@gametreedb-v44.ul4vx.mongodb.net/gametreedb?authSource=admin"

client = MongoClient(URI)

db = client.get_database()

# Collections:

# notifs  events  news  messages  reviews  users  wantToPlay  invites  removedmessages
# comments  xppoints  matchinglogs  testanswers  tokens  mautics  newssources  reports
# newsclicks  admins  posts  connections  games  marketingemails  dailyreports  readmessages
# sessionfeedbacks  useractions  customgames  notifications  subscriptions  comment_v2
# objectlabs-system  unreadmessages  questions  objectlabs-system.admin.collections  usercontacts
#________________________________________________________________________________________________________________

	###############################################
	# HELPER FUNCTIONS THAT TAKE USERLIST OBJECTS #
	###############################################
	
def getAge(userList):
	#input a list of userId's and returns the frequency of user's ages and a list of everyone's  raw ages
	# Ex:
	#		age = [A_1,A_2,...,A_100] where A_i is a frequency
	#		temp = [a_1,a_2,...,a_k] where k is the length of the userList and a_i is one user's age

	temp = [db.users.find_one({'_id':ObjectId(str(userid))}).get('profile').get('age') for userid in userList]
	age_dict ={i:temp.count(i) for i in temp}
	age = [age_dict[i] if age_dict.get(i) else 0 for i in range(0,100)]
	print("found",sum(age),"out of",len(userList))

	return age,temp


def checkCompletedTests(userList):
	# input a list of userId's, checks whether each user has completed each tests independently
	# and returns a tuple of three lists containing all the users who completed either tests
	# respectively.
	u_dna = [] #users who maxed dna
	u_mbti = [] #users who maxed mbti
	u_value = [] #users who maxed values

	for index , i in enumerate(userList):

		user = db.users.find_one({'_id': ObjectId(str(i))})

		if(user.get('dnaProgress') == 27):
			u_dna.append(i)

		if(user.get('personalityProgress')==21):
			u_mbti.append(i)

		if(user.get('valueProgress')==25):
			u_value.append(i)

		print("checking:",str(index)+'/'+str(len(userList)))

	return(u_dna,u_mbti,u_value)


def getMbtiGrid(userList):
	# input a list of userId's, and returns a (N x 21) matrix/grid
	# where N is the number of users, and 21 corresponds to how many
	# questions there are in the mbti test.
	# NOTE: THIS FUNCTION WILL ONLY RETURN THE USERS WHO HAVE MAXED
	#       OUT THEIR MBTI SCORE
	mbti = []

	j=0


	for i in userList:
		

		u1 = []



		for answer in db.testanswers.find({'$and' :[

											  {'userId': str(i)},
											  { 'testType' : 'MBTI Test' } 
											  ] } ):
			val = answer.get('values')
			u1.append(val[len(val)-1])

		if(len(u1)==21):
			mbti.append(u1)

		print("Mbti:",str(j)+"/"+str(len(userList)))
		j += 1

	return mbti



def getDnaGrid(userList):
	# input a list of userId's, and returns a (N x 27) matrix/grid
	# where N is the number of users, and 27 corresponds to how many
	# questions there are in the gamerDNA test.
	# NOTE: THIS FUNCTION WILL ONLY RETURN THE USERS WHO HAVE MAXED
	#       OUT THEIR GAMERDNA SCORE	
	dna = []
	j = 0
	for i in userList:
	 
		

		u1 = []



		for answer in db.testanswers.find({'$and' :[

											  {'userId': str(i)},
											  { 'testType' : 'DNA Test' } 
											  ] } ):
			val = answer.get('values')
			u1.append(val[len(val)-1])

		if(len(u1)==27):
			dna.append(u1)

		print("Dna:",str(j)+"/"+str(len(userList)))
		j += 1

	return dna


def getValuesGrid(userList):
	# input a list of userId's, and returns a (N x 25) matrix/grid
	# where N is the number of users in the list, and 25 corresponds to how many
	# questions there are in the values test.
	# NOTE: THIS FUNCTION WILL ONLY RETURN THE USERS WHO HAVE MAXED
	#       OUT THEIR VALUES SCORE. IF A USER DOES NOT HAVE THEIR SCORE
	#       MAXED OUT, THEY WILL NOT BE RETURNED IN THE MATRIX.	
	values = []

	index = 0

	for line in userList:

		values.append(
					(							
					db.users.find_one({'_id':ObjectId(str(line))}).get('personalityProfile').get('valueTest')[1:]				
					)
				)
		print("values:",str(index)+"/"+str(len(userList)))
		index +=1
	return values

def getGameGenres(userIds):
	# input a list of userId's and returns a list with the frequency of
	# users game genres in this order:
	# [action,adventure,casual,fighting,music,rpg,sports,strategy]
	action,adventure,casual,fighting,music,rpg,sports,strategy = 0,0,0,0,0,0,0,0
	count = 0
	for user in userIds:
		genres = db.users.find_one({'_id':ObjectId(user)}).get('genres')

		if(genres):

			for genre in genres:

				if(genre == 'Action'):action+=1
				if(genre == 'Adventure'):adventure+=1
				if(genre == 'Casual'):casual+=1
				if(genre == 'Fighting'):fighting+=1
				if(genre == 'Music & Party'):music+=1
				if(genre == 'RPG'):rpg+=1
				if(genre == 'Sports'):sports+=1
				if(genre == 'Strategy'):strategy+=1

			print(count)
			count+=1

	return [action,adventure,casual,fighting,music,rpg,sports,strategy]




def getPlatforms(userIds):
	# input a list of userId's and returns a list with the frequency of
	# users game platforms in this order:
	# [computer,mobile,switch,xbox,playstation,wii,tabletop]
	computer,mobile,switch,xbox,playstation,wii,tabletop = 0,0,0,0,0,0,0

	for user in userIds:
		platforms = db.users.find_one({'_id':ObjectId(user)}).get('platforms')

		if(platforms):

			for platform in platforms:

				if(platform == 'computer'):computer+=1
				if(platform == 'mobile'):mobile+=1
				if(platform == 'switch'):switch+=1
				if(platform == 'xbox'):xbox+=1
				if(platform == 'playstation'):playstation+=1
				if(platform == 'wii'):wii+=1
				if(platform == 'tabletop'):tabletop+=1


	return [computer,mobile,switch,xbox,playstation,wii,tabletop]

def getLocation(userIds):
	# input a list of userId's and returns a dictionary where keys are
	# distinct countries and values are the frequency in which they
	# occur. 
	# EX. {Peru:27, Venezuela:10, United States: 89} 

	location = [location.get('country') if(location := db.users.find_one({'_id':ObjectId(user)}).get('location')) else None for user in userIds ]
	temp_dict ={i:location.count(i) for i in location}
	return temp_dict



	##############################################
	# HELPER FUNCTIONS THAT TAKE DATASET OBJECTS #
	##############################################


#TODO: Make it general and not just 10 data points
def getPercentages(Grid,q_index,):

	zero,ten,twenty,thirty,forty,fifty,sixty,seventy,eighty,ninety,hundred=0,0,0,0,0,0,0,0,0,0,0

	for value in Grid:

		if(-5 <= value[q_index] <= 5): 
			zero+=1

		elif(5 < value[q_index] <= 15):
			ten+=1

		elif(15 < value[q_index] <= 25):
			twenty+=1

		elif(25 < value[q_index] <= 35):
			thirty+=1
			
		elif(35 < value[q_index] <= 45):
			forty+=1

		elif(45 < value[q_index] <= 55):
			fifty+=1

		elif(55 < value[q_index] <= 65):
			sixty+=1
		
		elif(65 < value[q_index] <= 75):
			seventy+=1
			
		elif(75 < value[q_index] <= 85):
			eighty+=1

		elif(85 < value[q_index] <= 95):
			ninety+=1

		else:
			hundred+=1

	y_axis = [zero,ten,twenty,thirty,forty,fifty,sixty,seventy,eighty,ninety,hundred]
	y_axis = [n/len(Grid) for n in y_axis]
	x_axis = [0,10,20,30,40,50,60,70,80,90,100]

	return x_axis,y_axis


def write_grids(first_dataset,second_dataset):

	# with open(first_dataset[0],'r',encoding = 'utf8') as data_1, open(second_dataset[0],'r',encoding = 'utf8') as data_2:
	# 	temp = data_1.read().split('\n')
	# 	sample_1 = random.sample(temp,n)
	# 	temp = data_2.read().split('\n')
	# 	sample_2 = random.sample(temp,n)

	# with open(first_dataset[1]+'sample.txt','w',encoding = 'utf8') as data_1, open(second_dataset[1]+'sample.txt','w',encoding = 'utf8') as data_2:
	# 	for (user_1,user_2) in zip(sample_1,sample_2):
	# 		print(user_1,file = data_1)
	# 		print(user_2,file = data_2)

	S1_dna,S1_mbti,S1_value = checkCompletedTests(first_dataset[0])
	S2_dna,S2_mbti,S2_value = checkCompletedTests(second_dataset[0])

	S1_Grids = [getDnaGrid(S1_dna), getMbtiGrid(S1_mbti), getValuesGrid(S1_value)]
	S2_Grids = [getDnaGrid(S2_dna), getMbtiGrid(S2_mbti), getValuesGrid(S2_value)]

	with open(first_dataset[1]+"_Values_data.txt",'w',encoding = 'utf8') as values, open(first_dataset[1]+"_Dna_data.txt",'w',encoding = 'utf8') as dna, open(first_dataset[1]+"_Mbti_data.txt",'w',encoding = 'utf8') as mbti:
		for row in S1_Grids[0]:
			print(row,file = dna)
		for row in S1_Grids[1]:
			print(row,file = mbti)
		for row in S1_Grids[2]:
			print(row,file = values)

	with open(second_dataset[1]+"_Values_data.txt",'w',encoding = 'utf8') as values, open(second_dataset[1]+"_Dna_data.txt",'w',encoding = 'utf8') as dna, open(second_dataset[1]+"_Mbti_data.txt",'w',encoding = 'utf8') as mbti:
		for row in S2_Grids[0]:
			print(row,file = dna)
		for row in S2_Grids[1]:
			print(row,file = mbti)
		for row in S2_Grids[2]:
			print(row,file = values)

#____________________________________________________________________________________

def after2(dataset,test_type,q_index):
	grid_data = [] # will be a list whose elements consists of 2 Dimensional matrices

	for data_tuple in dataset:

		with open(data_tuple[1]+"_Values_data.txt",'r',encoding = 'utf8') as values, open(data_tuple[1]+"_Dna_data.txt",'r',encoding = 'utf8') as dna, open(data_tuple[1]+"_Mbti_data.txt",'r',encoding = 'utf8') as mbti:
			if(test_type == 'values'):
				grid_data.append(values.read().split('\n'))
			if(test_type == 'dna'):
				grid_data.append(dna.read().split('\n'))
			if(test_type == 'mbti'):
				grid_data.append(mbti.read().split('\n'))

		#__________________________________________________________________________
	for m,data in enumerate(grid_data):
		for index,row in enumerate(grid_data[m]):

			if(row):
				grid_data[m][index] = row[row.index('[') + 1: row.index(']')].split(', ')

			else:
				grid_data[m].pop(index)

		for i,row in enumerate(grid_data[m]):

			for j,element in enumerate(row):

				if(element == 'None'):
					grid_data[m][i][j] = int(50)

				else:
					grid_data[m][i][j] = int(grid_data[m][i][j])
	#___________________________________________________________________________
	with open(textfiles_path+"table_values_labels.txt",'r',encoding = 'utf8') as values, open(textfiles_path+"table_gamerDNA_labels.txt",'r',encoding = 'utf8') as dna, open(textfiles_path+"table_mbti_labels.txt",'r',encoding = 'utf8') as mbti:
		if(test_type == 'values'):
			labels = values.read().split('\n')
		if(test_type == 'dna'):
			labels = dna.read().split('\n')
		if(test_type == 'mbti'):
			labels = mbti.read().split('\n')

	axis_list = [getPercentages(grid_data[x],q_index) for x in range(0,len(dataset))]

	fig = go.Figure()

	for n in range(0,len(dataset)):
		fig.add_trace(go.Scatter(
	    	x=axis_list[n][0], y=axis_list[n][1], name = dataset[n][1]
		))

	fig.update_layout(title = labels[q_index])

	fig.write_image(width = 1920, height = 1080, file = "images/"+test_type+"/"+test_type+"_"+str(q_index)+".png")

def downloadImages(dataset):

	if not os.path.exists("images"):
	    os.mkdir("images")
	    if not os.path.exists("images/values"):
	    	os.mkdir("images/values")
	    if not os.path.exists("images/dna"):
	    	os.mkdir("images/dna")
	    if not os.path.exists("images/mbti"):
	    	os.mkdir("images/mbti")


	for i in range(0,27):
		if(i < 21):
			after2(dataset,'mbti',i)
		if(i < 25):
			after2(dataset,'values',i)
		if(i < 27):
			after2(dataset,'dna',i)

def plotLocations(dataset):

	print(len(dataset[0][0]))
	data_dicts = []
	for data in dataset:

		data_dicts.append(getLocation(data[0][:len(data[0])]))

	x_labels = []

	for data in data_dicts:

		x_labels += [k for k,v in data.items() if(k)]

	x_labels = sorted(list(set(x_labels)))
	data_axis = []
	for data in data_dicts:
		data_axis.append([data[x] if(data.get(x)) else 0 for x in x_labels ])


	fig_2 = go.Figure()

	for index,data in enumerate(data_axis):

		fig_2.add_trace(go.Bar(
	    x=x_labels, y=[x/len(dataset[0]) for x in data], name = dataset[index][1]
		))


	fig_2.update_layout(title = 'Countries')
	fig_2.show()


def plotAges(dataset):
	x_labels = np.array(range(0,100))

	fig_2 = go.Figure()

	for data in dataset:
		fig_2.add_trace(go.Bar(
		    x=x_labels, y=[x/len(data[0]) for x in getAge(data[0])], name = data[1]
		))

	fig_2.update_layout(title = 'Age')
	fig_2.show()

def plotPlatforms(dataset):

	x_labels = ['computer','mobile','switch','xbox','playstation','wii','tabletop']

	fig = go.Figure()

	for data in dataset:
		fig.add_trace(go.Bar(
		    x=x_labels, y=[x/len(data[0]) for x in getPlatforms(data[0])], name = data[1]
		))
	fig.update_layout(title = 'Game Platforms')
	fig.show()


def plotGameGenres(dataset):

	x_labels = ['action','adventure','casual','fighting','music','rpg','sports','strategy']

	fig = go.Figure()

	for data in dataset:
		fig.add_trace(go.Bar(
	    x=x_labels, y=[x/len(data[0]) for x in getGameGenres(data[0])], name = data[1]

		))
	fig.update_layout(title = 'Game Genres')
	fig.show()


	

	##################################################################
	# INFO FUNCTIONS THAT GIVE INFORMATION ABOUT A SPECIFIC QUESTION #
	##################################################################
# input a filename that leads to a .txt file whose data consists of a
# two dimensional array in string format.
# Outputs a pandas dataframe.
# Ex: [[1,2,3,4,5],[4,5,6,7,8],[7,8,9,10,11]]
# NOTE: can have different rows but all the columns must be the same!
def textGrid_toCsv(filename):
	with open(filename,'r',encoding = 'utf8') as textfile:
		grid_data = textfile.read().split('\n')

	for i,row in enumerate(grid_data):

		if(row):
			grid_data[i] = row[row.index('[') + 1: row.index(']')].split(', ')

		else:
			grid_data.pop(i)

	for j,row in enumerate(grid_data):
		for k, element in enumerate(grid_data[j]):

			if(element == 'None'):
				grid_data[j][k] = None

			else:
				grid_data[j][k] = int(grid_data[j][k])

	columns = [('Q'+str(i)) for i in range(1,len(grid_data[0])+1)]
	return pd.DataFrame(grid_data,columns = columns)

def plotGameRatingDistribution(gameId):
	# input a gameId to plot the distribution of a game
	raters_count = db.users.count_documents({'gameRatings':{'$exists':True, '$not': {'$size':0}}})
	raters = db.users.find({'gameRatings':{'$exists':True, '$not': {'$size':0}}})
	game_name = db.games.find_one({'_id':ObjectId(gameId)}).get('weburl')

	temp = []
	count = 0
	for user in raters:
		rating = user.get('gameRatings')
		for dic in rating:
			if(dic.get('gameId') == gameId):
				temp.append(dic.get('value'))

		print('rater:'+str(count)+"/"+str(raters_count))
		count+=1


	temp_dict ={i:temp.count(i) for i in temp}
	print(temp_dict)

	x_axis = range(-101,100)
	rate = []
	for i in x_axis:

		if(temp_dict.get(i)):
			rate.append(temp_dict[i])
		else:
			rate.append(0)

	fig = go.Figure()

	fig.add_trace(go.Bar(
	    x=np.array(x_axis), y=rate, name = game_name
	))
	
	fig.update_layout(title = 'ratings for'+ game_name)
	fig.show()


def gameRatingMoreThanN(gameId,N):
	#	inputs:
	#		gameId - the gameId of a game
	#		N      - An integer less than or equal to 100
	#	returns a list of userId's who rated the game higher than or equal to N
	raters_count = db.users.count_documents({'gameRatings':{'$exists':True, '$not': {'$size':0}}})
	raters = db.users.find({'gameRatings':{'$exists':True, '$not': {'$size':0}}})
	game_name = db.games.find_one({'_id':ObjectId(gameId)}).get('weburl')

	temp = []
	count = 0
	for user in raters:
		rating = user.get('gameRatings')
		for dic in rating:
			if(dic.get('gameId') == gameId):
				if(dic.get('value')>=N):
					temp.append(user.get('_id'))

		print('rater:'+str(count)+"/"+str(raters_count))
		count+=1

	return temp

def usersPlayingGame(gameId, rtrn_list = False):
	#	inputs:
	#		gameId
	#		rtrn_list (boolean) - Set this to true to return a list
	#							  of users who play the game, otherwise
	#							  will only print # of users playing it
	gamers = db.users.count_documents({'gameIds':{'$in': [gameId]}})
	print(gamers,"playing",db.games.find_one({'_id':ObjectId(gameId)}).get('weburl'))
	if(rtrn_list):
		game_name = db.games.find_one({'_id':ObjectId(gameId)}).get('weburl')
		temp = []
		count = 0
		for player in db.users.find({'gameIds':{'$in': [gameId]}}):
			temp.append(str(player.get('_id')))
			print(count,"/",gamers)
			count+=1
		return temp

		

def myGames():
	# prints the gameId and name of my gametree games (anthonym650)
	for game in (db.users.find_one({'username':'anthonym650'}).get('gameIds')):
		x = (db.games.find_one({'_id':ObjectId(game)}))
		print(x.get('weburl'),":",game)


	############################################################
	# REJECTED CODE THAT WAS UPDATED BUT MIGHT STILL BE USEFUL #
	############################################################

	# def after(first_dataset,second_dataset,test_type,k):

	# with open(first_dataset[1]+"_Values_data.txt",'r',encoding = 'utf8') as values, open(first_dataset[1]+"_Dna_data.txt",'r',encoding = 'utf8') as dna, open(first_dataset[1]+"_Mbti_data.txt",'r',encoding = 'utf8') as mbti:
	# 	if(test_type == 'values'):
	# 		S1_data = values.read().split('\n')
	# 	if(test_type == 'dna'):
	# 		S1_data = dna.read().split('\n')
	# 	if(test_type == 'mbti'):
	# 		S1_data = mbti.read().split('\n')
	# with open(second_dataset[1]+"_Values_data.txt",'r',encoding = 'utf8') as values, open(second_dataset[1]+"_Dna_data.txt",'r',encoding = 'utf8') as dna, open(second_dataset[1]+"_Mbti_data.txt",'r',encoding = 'utf8') as mbti:
	# 	if(test_type == 'values'):
	# 		S2_data = values.read().split('\n')
	# 	if(test_type == 'dna'):
	# 		S2_data = dna.read().split('\n')
	# 	if(test_type == 'mbti'):
	# 		S2_data = mbti.read().split('\n')
	# #__________________________________________________________________________

	# for index,row in enumerate(S1_data):
	# 	if(row):
	# 		S1_data[index] = row[row.index('[') + 1: row.index(']')].split(', ')
	# 	else:
	# 		S1_data.pop(index)

	# for i,row in enumerate(S1_data):

	# 	for j,element in enumerate(row):

	# 		if(element == 'None'):
	# 			S1_data[i][j] = int(50)

	# 		else:
	# 			S1_data[i][j] = int(S1_data[i][j])
	# #___________________________________________________________________________

	# for index,row in enumerate(S2_data):
	# 	if(row):
	# 		S2_data[index] = row[row.index('[') + 1: row.index(']')].split(', ')
	# 	else:
	# 		S2_data.pop(index)

	# for i,row in enumerate(S2_data):

	# 	for j,element in enumerate(row):

	# 		if(element == 'None'):
	# 			S2_data[i][j] = int(50)

	# 		else:
	# 			S2_data[i][j] = int(S2_data[i][j])
	# #___________________________________________________________________________
	# with open(textfiles_path+"table_values_labels.txt",'r',encoding = 'utf8') as values, open(textfiles_path+"table_gamerDNA_labels.txt",'r',encoding = 'utf8') as dna, open(textfiles_path+"table_mbti_labels.txt",'r',encoding = 'utf8') as mbti:
	# 	if(test_type == 'values'):
	# 		labels = values.read().split('\n')
	# 	if(test_type == 'dna'):
	# 		labels = dna.read().split('\n')
	# 	if(test_type == 'mbti'):
	# 		labels = mbti.read().split('\n')

	# x_axis,S1_axis = getPercentages(S1_data,k)
	# x_axis,S2_axis = getPercentages(S2_data,k)

	# fig = go.Figure()

	# fig.add_trace(go.Scatter(
	#     x=x_axis, y=S1_axis, name = first_dataset[1]
	# ))
	# fig.add_trace(go.Scatter(
	#     x=x_axis, y=S2_axis, name = second_dataset[1]
	# ))
	# fig.update_layout(title = labels[k])
	# fig.show()
	# # fig.write_image(width = 1920, height = 1080, file = "images/"+test_type+"/"+test_type+"_"+str(k)+".png")
