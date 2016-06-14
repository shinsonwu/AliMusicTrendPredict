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

song_file_path = "./data/mars_tianchi_songs.csv"
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

ua_file_path1 = "./data/mars_tianchi_user_actions.csv"
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
# rank based index : the key is rank
#users_to_songs = defaultdict(lambda : defaultdict(list))
#users_to_songs_inday = defaultdict(lambda : defaultdict(list))
users_to_songs_play = defaultdict(lambda : defaultdict(list))
users_to_songs_play_inday = defaultdict(lambda : defaultdict(list))
users_to_songs_download = defaultdict(lambda : defaultdict(list))
users_to_songs_download_inday = defaultdict(lambda : defaultdict(list))
users_to_songs_collect = defaultdict(lambda : defaultdict(list))
users_to_songs_collect_inday = defaultdict(lambda : defaultdict(list))

# rank based index : the key is rank
#users_to_artists = defaultdict(lambda : defaultdict(list))
#users_to_artists_inday = defaultdict(lambda : defaultdict(list))
users_to_artists_play = defaultdict(lambda : defaultdict(list))
users_to_artists_play_inday = defaultdict(lambda : defaultdict(list))
users_to_artists_download = defaultdict(lambda : defaultdict(list))
users_to_artists_download_inday = defaultdict(lambda : defaultdict(list))
users_to_artists_collect = defaultdict(lambda : defaultdict(list))
users_to_artists_collect_inday = defaultdict(lambda : defaultdict(list))

# rank based index : the key is rank
#artists_actions = defaultdict(list)
#artists_actions_inday = defaultdict(list)
artists_play = defaultdict(list)
artists_play_inday = defaultdict(list)
artists_download = defaultdict(list)
artists_download_inday = defaultdict(list)
artists_collect = defaultdict(list)
artists_collect_inday = defaultdict(list)

artists_to_users = defaultdict(set)

# rank based index : the key is rank
#songs_actions = defaultdict(list)
#songs_actions_inday = defaultdict(list)
songs_play = defaultdict(list)
songs_play_inday = defaultdict(list)
songs_download = defaultdict(list)
songs_download_inday = defaultdict(list)
songs_collect = defaultdict(list)
songs_collect_inday = defaultdict(list)

songs_to_users = defaultdict(set)

print ""
print "===start action statistics=================================="
ta0 = time.time()

ua_file_path = "./data/mars_tianchi_user_actions.csv"
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

    artists_to_users[artist_rank].add(user_rank)
    songs_to_users[song_rank].add(user_rank)

    #users_to_songs[user_id][song_id].append((action_time_hour, action_time_date))
    #users_to_artists[user_id][songs_id_to_songinfo[song_id][0]].append((action_time_hour, action_time_date))
    #artists_actions[songs_id_to_songinfo[song_id][0]].append((action_time_hour, action_time_date))
    #songs_actions[song_id].append((action_time_hour, action_time_date))

    if(action_type == 1):
        users_to_songs_play[user_rank][song_rank].append((action_time_hour, action_time_date))
        users_to_artists_play[user_rank][artist_rank].append((action_time_hour, action_time_date))
        artists_play[artist_rank].append((action_time_hour, action_time_date))
        songs_play[song_rank].append((action_time_hour, action_time_date))
    elif(action_type == 2):
        users_to_songs_download[user_rank][song_rank].append((action_time_hour, action_time_date))
        users_to_artists_download[user_rank][artist_rank].append((action_time_hour, action_time_date))
        artists_download[artist_rank].append((action_time_hour, action_time_date))
        songs_download[song_rank].append((action_time_hour, action_time_date))
    elif(action_type == 3):
        users_to_songs_collect[user_rank][song_rank].append((action_time_hour, action_time_date))
        users_to_artists_collect[user_rank][artist_rank].append((action_time_hour, action_time_date))
        artists_collect[artist_rank].append((action_time_hour, action_time_date))
        songs_collect[song_rank].append((action_time_hour, action_time_date))


    #print idx, user_id, song_id, action_time_hour, action_type,\
    #       action_time_date
    #print user_actions[idx]

#for k, v in users_to_songs.items():
#    for ke, va in v.items():
#        va.sort(key = lambda item : item[0])
#        users_to_songs[k][ke] = va
for k, v in users_to_songs_play.items():
    for ke, va in v.items():
        va.sort(key = lambda item : item[1])
        users_to_songs_play[k][ke] = va
for k, v in users_to_songs_download.items():
    for ke, va in v.items():
        va.sort(key = lambda item : item[1])
        users_to_songs_download[k][ke] = va
for k, v in users_to_songs_collect.items():
    for ke, va in v.items():
        va.sort(key = lambda item : item[1])
        users_to_songs_collect[k][ke] = va

#for k, v in users_to_artists.items():
#    for ke, va in v.items():
#        va.sort(key = lambda item : item[1])
#        users_to_artists[k][ke] = va
for k, v in users_to_artists_play.items():
    for ke, va in v.items():
        va.sort(key = lambda item : item[1])
        users_to_artists_play[k][ke] = va
for k, v in users_to_artists_download.items():
    for ke, va in v.items():
        va.sort(key = lambda item : item[1])
        users_to_artists_download[k][ke] = va
for k, v in users_to_artists_collect.items():
    for ke, va in v.items():
        va.sort(key = lambda item : item[1])
        users_to_artists_collect[k][ke] = va

#for k, v in artists_actions.items():
#    v.sort(key = lambda item : item[1])
#    artists_actions[k] = v
for k, v in artists_play.items():
    v.sort(key = lambda item : item[1])
    artists_play[k] = v
for k, v in artists_download.items():
    v.sort(key = lambda item : item[1])
    artists_download[k] = v
for k, v in artists_collect.items():
    v.sort(key = lambda item : item[1])
    artists_collect[k] = v

#for k, v in songs_actions.items():
#    v.sort(key = lambda item : item[1])
#    songs_actions[k] = v
for k, v in songs_play.items():
    v.sort(key = lambda item : item[1])
    songs_play[k] = v
for k, v in songs_download.items():
    v.sort(key = lambda item : item[1])
    songs_download[k] = v
for k, v in songs_collect.items():
    v.sort(key = lambda item : item[1])
    songs_collect[k] = v


#for k, v in users_to_songs.items():
#    for ke, va in v.items():
#        vd = []
#        c = 1
#        dateTemp = -1
#        itemTemp = (0, 0)
#        for item in va:
#            if(item[1] == dateTemp):
#                c += 1
#            else:
#                vd.append((c, itemTemp[1]))
#                dateTemp = item[1]
#                itemTemp = item
#                c = 1
#
#        vd.append((c, itemTemp[1]))
#
#        vd.pop(0)
#        users_to_songs_inday[k][ke] = vd
#
#users_to_songs.clear()

for k, v in users_to_songs_play.items():
    for ke, va in v.items():
        vd = []
        c = 1
        dateTemp = -1
        itemTemp = (0, 0)
        for item in va:
            if(item[1] == dateTemp):
                c += 1
            else:
                vd.append((c, itemTemp[1]))
                dateTemp = item[1]
                itemTemp = item
                c = 1

        vd.append((c, itemTemp[1]))

        vd.pop(0)
        users_to_songs_play_inday[k][ke] = vd

users_to_songs_play.clear()


for k, v in users_to_songs_download.items():
    for ke, va in v.items():
        vd = []
        c = 1
        dateTemp = -1
        itemTemp = (0, 0)
        for item in va:
            if(item[1] == dateTemp):
                c += 1
            else:
                vd.append((c, itemTemp[1]))
                dateTemp = item[1]
                itemTemp = item
                c = 1

        vd.append((c, itemTemp[1]))

        vd.pop(0)
        users_to_songs_download_inday[k][ke] = vd

users_to_songs_download.clear()

for k, v in users_to_songs_collect.items():
    for ke, va in v.items():
        vd = []
        c = 1
        dateTemp = -1
        itemTemp = (0, 0)
        for item in va:
            if(item[1] == dateTemp):
                c += 1
            else:
                vd.append((c, itemTemp[1]))
                dateTemp = item[1]
                itemTemp = item
                c = 1

        vd.append((c, itemTemp[1]))

        vd.pop(0)
        users_to_songs_collect_inday[k][ke] = vd

users_to_songs_collect.clear()

#for k, v in users_to_artists.items():
#    for ke, va in v.items():
#        vd = []
#        c = 1
#        dateTemp = -1
#        itemTemp = (0, 0)
#        for item in va:
#            if(item[1] == dateTemp):
#                c += 1
#            else:
#                vd.append((c, itemTemp[1]))
#                dateTemp = item[1]
#                itemTemp = item
#                c = 1
#
#        vd.append((c, itemTemp[1]))
#
#        vd.pop(0)
#        users_to_artists_inday[k][ke] = vd
#
#users_to_artists.clear()

for k, v in users_to_artists_play.items():
    for ke, va in v.items():
        vd = []
        c = 1
        dateTemp = -1
        itemTemp = (0, 0)
        for item in va:
            if(item[1] == dateTemp):
                c += 1
            else:
                vd.append((c, itemTemp[1]))
                dateTemp = item[1]
                itemTemp = item
                c = 1

        vd.append((c, itemTemp[1]))

        vd.pop(0)
        users_to_artists_play_inday[k][ke] = vd

users_to_artists_play.clear()

for k, v in users_to_artists_download.items():
    for ke, va in v.items():
        vd = []
        c = 1
        dateTemp = -1
        itemTemp = (0, 0)
        for item in va:
            if(item[1] == dateTemp):
                c += 1
            else:
                vd.append((c, itemTemp[1]))
                dateTemp = item[1]
                itemTemp = item
                c = 1

        vd.append((c, itemTemp[1]))

        vd.pop(0)
        users_to_artists_download_inday[k][ke] = vd

users_to_artists_download.clear()

for k, v in users_to_artists_collect.items():
    for ke, va in v.items():
        vd = []
        c = 1
        dateTemp = -1
        itemTemp = (0, 0)
        for item in va:
            if(item[1] == dateTemp):
                c += 1
            else:
                vd.append((c, itemTemp[1]))
                dateTemp = item[1]
                itemTemp = item
                c = 1

        vd.append((c, itemTemp[1]))

        vd.pop(0)
        users_to_artists_collect_inday[k][ke] = vd

users_to_artists_collect.clear()

#for k, v in artists_actions.items():
#    vd = []
#    c = 1
#    dateTemp = -1
#    itemTemp = (0, 0)
#    for item in v:
#        if(item[1] == dateTemp):
#            c += 1
#        else:
#            vd.append((c, itemTemp[1]))
#            dateTemp = item[1]
#            itemTemp = item
#            c = 1
#
#    vd.append((c, itemTemp[1]))
#
#    vd.pop(0)
#    artists_actions_inday[k] = vd
#
#artists_actions.clear()

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

for k, v in artists_download.items():
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
    artists_download_inday[k] = vd

artists_download.clear()

for k, v in artists_collect.items():
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
    artists_collect_inday[k] = vd

artists_collect.clear()

#for k, v in songs_actions.items():
#    vd = []
#    c = 1
#    dateTemp = -1
#    itemTemp = (0, 0)
#    for item in v:
#        if(item[1] == dateTemp):
#            c += 1
#        else:
#            vd.append((c, itemTemp[1]))
#            dateTemp = item[1]
#            itemTemp = item
#            c = 1
#
#    vd.append((c, itemTemp[1]))
#
#    vd.pop(0)
#    songs_actions_inday[k] = vd
#
#songs_actions.clear()

for k, v in songs_play.items():
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
    songs_play_inday[k] = vd

songs_play.clear()

for k, v in songs_download.items():
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
    songs_download_inday[k] = vd

songs_download.clear()

for k, v in songs_collect.items():
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
    songs_collect_inday[k] = vd

songs_collect.clear()

ta1 = time.time()
print "It takes %f s to do action statistics" %(ta1-ta0)
print "===end actions statistics==================================="
######################### actions statistics##################################################

######################### plot ##################################################
# plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pcount = []
pdate = []
dcount = []
ddate = []
ccount = []
cdate = []
# plot jth song
def plotJS(j):
    plt.clf()
    p = songs_play_inday[j]
    del pcount[:]
    del pdate[:]
    for i in p:
        pcount.append(i[0])
        pdate.append(i[1])
    pser = pd.Series(pcount, index = pdate)
    pser.plot(use_index = 1, legend = True, label = str(j) + "th_song_play")

    d = songs_download_inday[j]
    del dcount[:]
    del ddate[:]
    for i in d:
        dcount.append(i[0])
        ddate.append(i[1])
    dser = pd.Series(dcount, index = ddate)
    dser.plot(use_index = 1, legend = True, label = str(j) + "th_song_download")

    c = songs_collect_inday[j]
    del ccount[:]
    del cdate[:]
    for i in c:
        ccount.append(i[0])
        cdate.append(i[1])
    cser = pd.Series(ccount, index = cdate)
    cser.plot(use_index = 1, legend = True, label = str(j) + "th_song_collect")

# plot jth artist
def plotJA(j):
    plt.clf()
    p = artists_play_inday[j]
    del pcount[:]
    del pdate[:]
    for i in p:
        pcount.append(i[0])
        pdate.append(i[1])
    pser = pd.Series(pcount, index = pdate)
    pser.plot(use_index = 1, legend = True, label = str(j) + "th_artist_play")

    d = artists_download_inday[j]
    del dcount[:]
    del ddate[:]
    for i in d:
        dcount.append(i[0])
        ddate.append(i[1])
    dser = pd.Series(dcount, index = ddate)
    dser.plot(use_index = 1, legend = True, label = str(j) + "th_artist_download")

    c = artists_collect_inday[j]
    del ccount[:]
    del cdate[:]
    for i in c:
        ccount.append(i[0])
        cdate.append(i[1])
    cser = pd.Series(ccount, index = cdate)
    cser.plot(use_index = 1, legend = True, label = str(j) + "th_artist_collect")

# jth user to kth song
def plotJUKS(j, k):
    plt.clf()
    p = users_to_songs_play_inday[j][k]
    del pcount[:]
    del pdate[:]
    for i in p:
        pcount.append(i[0])
        pdate.append(i[1])
    pser = pd.Series(pcount, index = pdate)
    pser.plot(use_index = 1, legend = True, label = str(j) + "th_user_play_" + str(k) + "th_song")

    d = users_to_songs_download_inday[j][k]
    del dcount[:]
    del ddate[:]
    for i in d:
        dcount.append(i[0])
        ddate.append(i[1])
    dser = pd.Series(dcount, index = ddate)
    dser.plot(use_index = 1, legend = True, label = str(j) + "th_user_download_" + str(k) + "th_song")

    c = users_to_songs_collect_inday[j][k]
    del ccount[:]
    del cdate[:]
    for i in c:
        ccount.append(i[0])
        cdate.append(i[1])
    cser = pd.Series(ccount, index = cdate)
    cser.plot(use_index = 1, legend = True, label = str(j) + "th_user_collect_" + str(k) + "th_song")

# jth user to kth artist
def plotJUKA(j, k):
    plt.clf()
    p = users_to_artists_play_inday[j][k]
    del pcount[:]
    del pdate[:]
    for i in p:
        pcount.append(i[0])
        pdate.append(i[1])
    pser = pd.Series(pcount, index = pdate)
    pser.plot(legend = True, label = str(j) + "th_user_play_" + str(k) + "th_artist")

    d = users_to_artists_download_inday[j][k]
    del dcount[:]
    del ddate[:]
    for i in d:
        dcount.append(i[0])
        ddate.append(i[1])
    dser = pd.Series(dcount, index = ddate)
    dser.plot(legend = True, label = str(j) + "th_user_download_" + str(k) + "th_artist")

    c = users_to_artists_collect_inday[j][k]
    del ccount[:]
    del cdate[:]
    for i in c:
        ccount.append(i[0])
        cdate.append(i[1])
    cser = pd.Series(ccount, index = cdate)
    cser.plot(legend = True, label = str(j) + "th_user_collect_" + str(k) + "th_artist")
########################## plot #################################################
