#-*- coding:utf8 -*-#
"""
---------------------------------------
*功能：
*保存：
---------------------------------------

"""
import os
import csv
import time
from collections import defaultdict

####################### date ####################################################
# map date into num
# date
print ""
print "===start generate date rank=================================="
date_to_rank = {}
rank_to_date = {}
import datetime
dt = datetime.datetime(2015, 03, 01, 00, 00, 01)
end = datetime.datetime(2015, 10, 30, 23, 59, 59)
step = datetime.timedelta(days=1)
day_rank = 0
while dt < end:
    day_date = dt.strftime('%Y%m%d')
    rank_to_date[day_rank] = day_date
    date_to_rank[day_date] = day_rank
    dt += step
    day_rank += 1
print "date num ", len(rank_to_date)
print "rank to date :", rank_to_date
print "===end generate date rank=================================="
####################### date ####################################################


####################### songs ####################################################
# load songs date
# song artist
song_id_set = set()
songs_id_to_songinfo = defaultdict(tuple)
songs_rank_to_iddate = [] #song rank to song_id and publish_date
songs_id_to_rank = {}

artist_id_set = set()
artists_id_to_artistinfo = defaultdict(tuple)
artists_rank_to_id = []
artists_id_to_rank = {}
artists_id_to_songs_id =  defaultdict(list) #artist_id to list of song_id
artists_rank_to_songs_num = {}

artist_gender_set = set()

language_type_set = set()


print ""
print "===start load songs=================================="
t0 = time.time()

song_file_path = "./data/p2_mars_tianchi_songs.csv"
f = open(song_file_path, 'r')
rows = csv.reader(f)

for row in rows:
    song_id = row[0]
    song_id_set.add(song_id)

    artist_id = row[1]
    artist_id_set.add(artist_id)

    publish_time = int(row[2])

    init_play_num = int(row[3])

    language_type = int(row[4])
    language_type_set.add(language_type)

    artist_gender = int(row[5])
    artist_gender_set.add(artist_gender)

    artists_id_to_songs_id[artist_id].append(song_id)
    artists_id_to_artistinfo[artist_id] = (artist_gender)

    songs_rank_to_iddate.append((song_id, publish_time))
    songs_id_to_songinfo[song_id] = (artist_id, publish_time, init_play_num, language_type, artist_gender)

    #print song_id, artist_id, publish_time, init_play_num,\
    #        language_type, artist_gender

# rank songs by date
songs_rank_to_iddate.sort(key = lambda item : item[1])
for rank, item in enumerate(songs_rank_to_iddate):
    songs_id_to_rank[item[0]] = rank

artists_rank_to_id = list(artist_id_set)
for rank, item in enumerate(artists_rank_to_id):
    artists_id_to_rank[item] = rank
artists_rank_to_id = list(artist_id_set)

for k, v in artists_id_to_songs_id.items():
    artists_rank_to_songs_num[artists_id_to_rank[k]] = len(v)

print "songs num ", len(song_id_set)
print "songs_id_to_songinfo num ", len(songs_id_to_songinfo)
print "artist num ", len(artist_id_set)
print "language type num ", len(language_type_set)
print "artist gender num ", len(artist_gender_set)
print "k th artist songs num ", artists_rank_to_songs_num

t1 = time.time()
print "It takes %f s to load songs" %(t1-t0)
print "===end load songs==================================="
####################### songs ####################################################



####################### actions ####################################################
# load songs actions
# song user actions
user_id_set = set()
users_rank_to_id = []
users_id_to_rank = {}

song_hasact_id_set = set()

action_type_set = set()

print ""
print "===start user statistics=================================="
tu0 = time.time()

ua_file_path1 = "./data/p2_mars_tianchi_user_actions.csv"
f1 = open(ua_file_path1, 'r')
rows1 = csv.reader(f1)
for idx, row in enumerate(rows1):
    user_id = row[0]
    user_id_set.add(user_id)

    song_id = row[1]
    song_hasact_id_set.add(song_id)

    action_type = int(row[3])
    action_type_set.add(action_type)

users_rank_to_id = list(user_id_set)
for rank, item in enumerate(users_rank_to_id):
    users_id_to_rank[item] = rank

print "user num", len(user_id_set)
print "song num that has action", len(song_hasact_id_set)
print "action type num", len(action_type_set)

tu1 = time.time()
print "It takes %f s to do user statistics" %(tu1-tu0)
print "===end user statistics==================================="
####################### actions ####################################################

####################### actions statistics####################################################
artists_play = defaultdict(list)
artists_play_inday = defaultdict(list)
print ""
print "===start action statistics=================================="
ta0 = time.time()

ua_file_path = "./data/p2_mars_tianchi_user_actions.csv"
f = open(ua_file_path, 'r')
rows = csv.reader(f)
for idx, row in enumerate(rows):
    user_id = row[0]
    user_rank = users_id_to_rank[user_id]

    song_id = row[1]
    song_rank = songs_id_to_rank[song_id]
    artist_rank = artists_id_to_rank[songs_id_to_songinfo[song_id][0]]

    action_time_hour = int(row[2])

    action_type = int(row[3])

    action_time_date = date_to_rank[row[4]]

    if(action_type == 1):
        artists_play[artist_rank].append((action_time_hour, action_time_date))

for k, v in artists_play.items():
    v.sort(key = lambda item : item[1])
    artists_play[k] = v

for k, v in artists_play.items():
    vd = []
    c = 1
    dateTemp = -1
    itemTemp = (0, 0)
    for item in v:
        if(item[1] == dateTemp):
            c += 1
        else:
            vd.append((c, itemTemp[1]))
            dateTemp = item[1]
            itemTemp = item
            c = 1

    vd.append((c, itemTemp[1]))

    vd.pop(0)
    artists_play_inday[k] = vd

artists_play.clear()

ta1 = time.time()
print "It takes %f s to do action statistics" %(ta1-ta0)
print "===end actions statistics==================================="
######################### actions statistics##################################################
