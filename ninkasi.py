from flask import Flask, render_template, request, url_for
from model.CB.content_based_recommendation import *
from model.CF.CF_predict import *
import json
import pickle
import numpy as np
import zipfile
import os.path

app = Flask(__name__)

import socket

# conn = socket.create_connection(("d6efcf30.carbon.hostedgraphite.com", 2003))
# conn.send("a9ffb36d-310b-4e39-9c42-d30f000b36da.test.testing 1.2\n")
# conn.close()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/recom", methods=['GET', 'POST'])
def recom():
    # content-based model loading
	with open("./model/CB/beer_list.json", "rb") as bf:
		beer_list1 = json.load(bf)
	with open("./model/CB/index.p", "rb") as idxf:
		index = pickle.load(idxf)
	with open("./model/CB/beer_keywords.json", "rb") as bk:
		beer_keywords = json.load(bk)
	with open("./model/CB/dict_for_CB_table.p", "rb") as dfct:
		dict_for_CB_table = pickle.load(dfct)

	# collab model loading
	with open("./model/CF/beer_dict.pickle", "rb") as bd:
		beer_dict = pickle.load(bd)
	# beer_list2 = sorted(beer_dict.values())
	beer_list2 = beer_list1
	# beer_list2 = [x.encode('utf-8', 'ignore').decode('ascii', 'ignore') for x in beer_list2 if '\x8a\x97\xc8' not in x] # we need to fix this later
	# if not os.path.exists("./model/CF/ratings_svdpp.npy"):
	# 	zip_ref = zipfile.ZipFile("./model/CF/ratings_svdpp.zip", 'r')
	# 	zip_ref.extractall("./model/CF")
	# 	zip_ref.close()
	ratings_mat = np.load("./model/CF/ratings_svdpp_v2.npy")

	ratings_mat, global_avg = CF_mat_preprocess(ratings_mat)

	cb_rec = None
	table_list1 = None
	key_words = None
	warning1 = None
	cur_inp0 = None

	cf_rec = None
	warning2 = None
	table_list2 = None
	cur_inp1 = None
	cur_len = None
	# show_extra = None

	if request.method == "POST":
		if "beer_inp0" in request.form:
			beer_inp0 = request.form["beer_inp0"]
			cur_inp0 = beer_inp0
			if not beer_inp0:
				warning1 = 'Please enter a beer name'
			elif beer_inp0 not in beer_list1:
				warning1 = 'Beer name is not valid, please try again'
			else:
				cb_rec = get_similar_beers(beer_inp0, beer_list1, index, ntop=10)
				table_list1 = map(lambda x: dict_for_CB_table.get(x), cb_rec)  # this table is for getting rating info
				key_words = beer_keywords[beer_inp0]
				key_words = map(lambda x: x.decode('utf-8', 'ignore').encode('ascii', 'ignore'), key_words)
				key_words = ', '.join(key_words)

		elif "beer_inp1" in request.form:
			inp_tup = []
			for i in range(1, 11):
				beer_inp_key = "beer_inp" + str(i)
				rating_inp_key = "rating_inp" + str(i)
				if request.form[beer_inp_key] and request.form[rating_inp_key]:
					inp_tup.append((request.form[beer_inp_key], request.form[rating_inp_key]))
			if inp_tup == '' or len(inp_tup) == 1:
				warning2 = 'Please rate at least two beers'
			elif True not in [x[0] in beer_list2 for x in inp_tup]:
				warning2 = 'Beer names are not valid, please try again'
			else:
				inp_tup = [x for x in inp_tup if x[0] in beer_list2]
				cur_inp1 = inp_tup
				# cur_len = len(cur_inp1)
				# print cur_inp1, cur_len
				d = {}
				for i in cur_inp1:
					d[i[0]] = i[1]
				cur_inp1 = [(x, d[x]) for x in order_list([x[0] for x in cur_inp1])]
				cur_len = len(cur_inp1)
				# print cur_inp1, cur_len
				user_data = CF_user_preprocess(inp_tup, ratings_mat, beer_dict)
				cf_rec = CF_rec(user_data, ratings_mat, global_avg, beer_dict)
				table_list2 = map(lambda x: dict_for_CB_table.get(x), cf_rec)

	return render_template("recom.html", beer_list1=beer_list1, cb_rec=cb_rec, table_list1=table_list1, key_words=key_words, warning1=warning1, cur_inp0=cur_inp0,
										 beer_list2=beer_list2, cf_rec=cf_rec, table_list2=table_list2, warning2=warning2, cur_inp1=cur_inp1, cur_len=cur_len)



if __name__ == "__main__":
    app.run(debug=True)