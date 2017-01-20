import pickle
import json

if __name__ == "__main__":
	beer_list = pickle.load(open("beer_list.p", "rb"))
	print "beer_list loaded"

	with open('beer_list.json','w') as bl:
		json.dump(beer_list, bl)
	print "beer_list json dumped"