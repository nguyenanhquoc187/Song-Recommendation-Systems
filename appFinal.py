from flask import Flask, request, jsonify
import pandas as pd
import mysql.connector
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from sklearn.preprocessing import MinMaxScaler
from spotipy.oauth2 import SpotifyClientCredentials 
from difflib import SequenceMatcher
client_id = "0054a24f2fc643c69d56d020dd5f70be"
client_secret = "98b4a4b772ad4eca934a92ca60c246a0"
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) 

app = Flask(__name__)

# Kết nối tới MySQL
def getConnect():
    mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="quoc0562138187",
    database="musicmanage")
    return mydb

def similarity(a, b):
    a = a.lower()
    b = b.lower()
    return SequenceMatcher(None, a, b).ratio()

def features_extract(tilte, artist):
    search = tilte + " " + artist
    results = sp.search(search,limit=50, offset=0 ,type="track",market="VN")
    items = results['tracks']['items']
    maxSimilarity = 0
    track_id =  0
    for i in range(20):
        similarTitle = similarity(tilte, items[i]['name'])
        similarArtist = similarity(artist, items[i]['artists'][0]['name'])
        similar = 2*similarTitle + similarArtist
        if (similar > maxSimilarity):
            maxSimilarity = similar
            track_id = items[i]['id']
    features = sp.audio_features([track_id])[0]
    result = {}
    result['acousticness'] = features['acousticness']
    result['danceability'] = features['danceability'] 
    result['energy'] = features['energy']
    result['instrumentalness'] = features['instrumentalness']
    result['liveness'] = features['liveness']
    result['loudness'] = features['loudness']
    result['speechiness'] = features['speechiness']
    result['tempo'] = features['tempo']
    result['valence'] = features['valence']
    return result

# One hot encoding 
def oneHotEncoding(df, column):
    col_ohe = df[column].str.get_dummies(', ')
    return col_ohe

def create_feature_data(df):
    """ 
        Xử lý dữ liệu để tạo tập dữ liệu đặc trưng cuối cùng sử dụng cho khuyến nghị
    """
    # One hot encoding cột thể loại
    genre_OHE = oneHotEncoding(df,'genres')

    float_cols = df.dtypes[df.dtypes == 'float64'].index.values
    # Chuẩn hoá các cột đăng trưng audio
    scaler = MinMaxScaler()
    floats = df[float_cols].reset_index(drop = True)
    df_scaled = pd.DataFrame(scaler.fit_transform(floats), columns=floats.columns, index=floats.index)
    final = pd.concat([df_scaled,genre_OHE],axis=1)
    final['id']=df['id'].values
    return final
    

def history(userId):
    mydb = getConnect()
    sqlHistory = f""" 
        SELECT songs.id, songs.title, max(date) as date FROM history_listens 
        INNER JOIN songs ON songs.id = history_listens.song_id
        where user_id = {userId}
        group by song_id 
        order by date desc 
        limit 10
        """
    return pd.read_sql(sqlHistory, mydb)

def playlist(playlistId):
    mydb = getConnect()
    sqlPlaylist = f"SELECT song_id as id FROM playlist_includes WHERE playlist_id = {playlistId}"
    return pd.read_sql(sqlPlaylist,mydb)

def generate_history_feature(final_features_data, history_listen_df ):
    """
        Tóm tắt lịch sử nghe nhạc của người dùng thành một vector
    """
    # tìm đặc trưng của mỗi bài hát trong lịch sử
    final_features_history = final_features_data[final_features_data['id'].isin(history_listen_df['id'].values)]
    # tìm tất cả bài hát không có trong lịch sử
    final_features_data_non_history = final_features_data[~final_features_data['id'].isin(history_listen_df['id'].values)]
    result = final_features_history.drop(columns = "id")
    return result.sum(axis = 0), final_features_data_non_history


def generate_history_vector(userId):
    final_features_data = pd.read_csv('final_features_data.csv')
    history_listen_df = history(userId)
    print(history_listen_df)
    history_listen_vector, final_features_data_non_history = generate_history_feature(final_features_data, history_listen_df)
    return history_listen_vector, final_features_data_non_history

def generate_playlist_vector(playlistId):
    final_features_data = pd.read_csv('final_features_data.csv')
    playlist_df = playlist(playlistId)
    print(playlist_df)
    playlist_vector, final_features_data_non_playlist = generate_history_feature(final_features_data, playlist_df)
    return playlist_vector,final_features_data_non_playlist
def generate_recommended_song_list(features, non_history_features):
    """
    Đề xuất các bài hát
    """
    df = pd.read_csv('song_database.csv')
    non_playlist_df = df[df['id'].isin(non_history_features['id'].values)]
    # Find cosine similarity between the playlist and the complete song set
    non_playlist_df['similarity'] = cosine_similarity(non_history_features.drop('id', axis = 1).values, features.values.reshape(1, -1))[:,0]
    non_playlist_df_top_10 = non_playlist_df.sort_values('similarity',ascending = False).head(10)
    
    return non_playlist_df_top_10

# Lấy thông tin bài hát vừa thêm
def getSongById(songId):
    sql = f""" 
    select s.id, s.title, s.lyrics,genres , group_concat(ar.name separator ', ') as artist, album from 
    (
    SELECT s.id, s.title, s.lyrics, 
    group_concat(gs.name separator ', ') as genres, al.name as album
    FROM songs as s
    inner join genre_of as go on s.id = go.song_id
    inner join genres as gs on go.genre_id = gs.id
    left join albums as al on s.album_id = al.id
    group by title ) as s
    inner join song_by as sb on s.id = sb.song_id
    inner join artists as ar on sb.artist_id = ar.id
    where s.id = {songId}
    group by title
    """
    return pd.read_sql(sql,getConnect())

def getFeaturesOneSong(song):
    features_one_song = {}
    while len(features_one_song) == 0:
        try:
            features_one_song = features_extract(song['title'].values[0], song['artist'].values[0])
        except:
            print("Lỗi trích xuất đặc trưng")
    return features_one_song


def add_song(song_id):
    song = getSongById(song_id)
    features_one_song = getFeaturesOneSong(song)
    features_one_song_df = pd.DataFrame.from_dict(features_one_song,orient='index').T
    song_features = pd.concat([song,features_one_song_df], axis=1)
    print('OK')
    song_features.to_csv('song_database.csv',mode='a',header=False,index=False)

def reNormalize():
    df = pd.read_csv("song_database.csv")
    final_features_data = create_feature_data(df)
    final_features_data.to_csv("final_features_data.csv", index = False)

@app.route('/recommend', methods=['POST'])
def getRecommendHistory():
    data = request.get_json() # lấy dữ liệu được gửi lên bởi client
    print(data)
    userId = data['userId'] 
    history_listen_vector, final_features_data_non_history = generate_history_vector(userId)
    recommend = generate_recommended_song_list(history_listen_vector,final_features_data_non_history )
    recommend_json = recommend['id'].to_json(orient='values')
    json_dict = {"recommendId": recommend_json}
    print(json_dict)
    return jsonify(json_dict)

 
@app.route('/recommend-playlist', methods =['POST'])
def getRecommendPlaylist():
    data = request.get_json()
    playlist_id = data['playlistId']
    playlist_vector,final_features_data_non_playlist = generate_playlist_vector(playlist_id)
    recommend = generate_recommended_song_list(playlist_vector,final_features_data_non_playlist )
    recommend_json = recommend['id'].to_json(orient='values')
    json_dict = {"recommendId": recommend_json}
    return jsonify(json_dict)

@app.route('/song', methods = ['POST'])
def addSongtoCsv():
    data = request.get_json()
    song_id = data['songId']
    add_song(song_id)
    reNormalize()
    json_dict = {"song_id": song_id}
    return jsonify(json_dict)

@app.route('/del-song', methods = ['POST'])
def delSongtoCsv():
    data = request.get_json()
    song_id = data['songId']
    df1 = pd.read_csv("song_database.csv")
    df1 = df1.drop(df1[df1['id'] == song_id].index)
    df1.to_csv("song_database.csv", index=False)

    df2 = pd.read_csv("final_features_data.csv")
    df2 = df2.drop(df2[df2['id'] == song_id].index)
    df2.to_csv("final_features_data.csv", index=False)
    json_dict = {"": ""}
    return jsonify(json_dict)

if __name__ == '__main__':
    app.run(debug=True)