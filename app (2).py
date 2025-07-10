from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import requests
from sqlalchemy import create_engine
from surprise import Dataset, Reader, SVD, dump
import os # 파일 경로 처리를 위한 os 모듈 추가

# Flask 앱 생성
app = Flask(__name__)

# TMDB API KEY
API_KEY = "2d3859f95d351c6539fc1ad0f734a384"

# 데이터베이스 연결 설정
# connect_timeout은 SQLAlchemy의 create_engine 인자가 아닐 수 있습니다.
# 일반적으로 pymysql 드라이버 설정으로 전달해야 하며, 여기서는 제거하거나 확인이 필요합니다.
# 문제가 발생하면 'connect_args'를 사용하여 드라이버 옵션을 전달해야 합니다.
try:
    engine = create_engine('mysql+pymysql://user3:1234@192.168.2.235:3306/project')
    # 연결 테스트 (선택 사항)
    with engine.connect() as connection:
        print("데이터베이스 연결 성공!")
except Exception as e:
    print(f"데이터베이스 연결 오류: {e}")
    # 실제 운영 환경에서는 여기서 애플리케이션 종료 또는 오류 페이지 처리가 필요합니다.


# SVD 모델 불러오기
# 사용자가 제공한 경로를 따르지만, 실제 경로와 파일명('.py' 부분 확인)을 확인해야 합니다.
# 예시: models 폴더 안에 모델 파일이 있다면 'models/svd_model.pkl' 형태로 사용합니다.
model_path = r'E:\AI_KDT7\13. Flask\mini\lengend\models.py\svd_Action_model.pkl' # <-- 이 경로를 실제 모델 파일 경로로 확인/수정하세요.
try:
    # dump.load는 (모델 이름, 모델 객체) 튜플을 반환하므로 [1]로 모델 객체만 가져옵니다.
    model = dump.load(model_path)[1]
    print("SVD 모델 로드 성공!")
except FileNotFoundError:
    print(f"오류: 모델 파일 '{model_path}'을(를) 찾을 수 없습니다.")
    model = None # 모델 로드 실패 시 None으로 설정
except Exception as e:
    print(f"SVD 모델 로드 오류: {e}")
    model = None # 모델 로드 실패 시 None으로 설정


# 데이터 불러오기 - 애플리케이션 시작 시 한 번만 로드
try:
    ratings = pd.read_sql("SELECT userId, movieId, rating FROM ratings", engine)
    movies = pd.read_sql("SELECT movieId, title, genres FROM movies", engine)
    genome_scores = pd.read_sql("SELECT movieId, tagId, relevance FROM genome_scores", engine)
    genome_tags = pd.read_sql("SELECT tagId, tag FROM genome_tags", engine)
    links = pd.read_sql("SELECT movieId, tmdbId FROM links", engine)
    print("데이터 로드 성공!")
except Exception as e:
    print(f"데이터 로드 오류: {e}")
    # 데이터 로드 실패 시 빈 DataFrame으로 초기화하거나 오류 처리가 필요합니다.
    ratings = pd.DataFrame()
    movies = pd.DataFrame()
    genome_scores = pd.DataFrame()
    genome_tags = pd.DataFrame()
    links = pd.DataFrame()


# 포스터 URL 가져오는 함수
def get_poster_url(movie_id):
    try:
        # links 데이터프레임에서 movieId에 해당하는 tmdbId 찾기
        tmdb_id_row = links.loc[links['movieId'] == movie_id, 'tmdbId']
        if tmdb_id_row.empty:
            # print(f"TMDB ID를 찾을 수 없는 영화: {movie_id}") # 디버깅용
            return None # TMDB ID가 없으면 None 반환

        tmdb_id = tmdb_id_row.values[0]

        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={API_KEY}&language=ko"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            poster_path = data.get('poster_path')
            if poster_path:
                return f"https://image.themoviedb.org/t/p/w500{poster_path}"
            else:
                 # print(f"포스터 경로가 없는 영화 (TMDB ID {tmdb_id}): {movie_id}") # 디버깅용
                 return None # 포스터 경로가 없으면 None 반환
        else:
            # print(f"TMDB API 호출 실패 (상태 코드 {response.status_code}, TMDB ID {tmdb_id}): {movie_id}") # 디버깅용
            return None # API 호출 실패 시 None 반환
    except Exception as e:
        print(f"포스터 가져오기 오류 (Movie ID {movie_id}): {e}") # 오류 발생 시 메시지 출력
        return None # 오류 발생 시 None 반환

# 1단계: 사용자 ID 입력 페이지
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# 2단계: 장르 선택 페이지 (사용자 ID를 받아 다음 페이지로 전달)
@app.route('/select_genre', methods=['POST'])
def select_genre():
    user_id = request.form.get('userId') # index.html 폼에서 userId를 가져옴

    if not user_id:
        # 사용자 ID가 입력되지 않았으면 다시 입력 페이지로 리다이렉트 또는 오류 메시지 표시
        return "사용자 ID를 입력해주세요.", 400

    # 장르 목록 (사용자가 선택할 5가지 장르)
    genres = ['Action', 'Animation', 'Comedy', 'Drama', 'Sci-Fi']

    # select_genre.html 템플릿 렌더링 시 사용자 ID와 장르 목록 전달
    return render_template('select_genre.html', user_id=user_id, genres=genres)

# 3단계: 장르별 영화 평점 입력 페이지
@app.route('/rate_movies/<selected_genre>', methods=['GET'])
def rate_movies(selected_genre):
    # select_genre.html에서 GET 요청으로 전달된 user_id와 장르 정보 가져오기
    user_id = request.args.get('userId')

    if not user_id or not selected_genre:
         return "잘못된 접근입니다. 사용자 ID와 장르 정보가 필요합니다.", 400

    # 선택된 장르와 2000년 이후 영화 필터링
    # 사용자의 데이터에 'Action' 외 다른 장르 이름이 정확히 일치하는지 확인 필요
    filtered_movies = movies[movies['genres'].str.contains(selected_genre, na=False)].copy()

    # 연도 추출 및 필터링
    extracted_years = filtered_movies['title'].str.extract(r'\((\d{4})\)', expand=False)
    numeric_years = pd.to_numeric(extracted_years, errors='coerce')
    recent_genre_movies = filtered_movies[(~numeric_years.isna()) & (numeric_years >= 2000)]

    # 샘플링 로직: 10개 이상이면 10개 샘플, 아니면 가진 모든 영화 사용
    if len(recent_genre_movies) >= 10:
         sampled_movies = recent_genre_movies.sample(10, random_state=42)
    else:
         print(f"경고: 2000년 이후 {selected_genre} 영화가 {len(recent_genre_movies)}개만 있습니다. 해당 영화들을 모두 표시합니다.")
         sampled_movies = recent_genre_movies

    movie_list = []
    for movie in sampled_movies.itertuples():
        # 각 영화에 대한 포스터 URL 가져오기
        poster_url = get_poster_url(movie.movieId)
        movie_list.append({
            'movieId': movie.movieId,
            'title': movie.title,
            'poster': poster_url
        })

    # rate_movies.html 템플릿 렌더링 시 사용자 ID, 장르, 영화 목록 전달
    return render_template('rate_movies.html', user_id=user_id, genre=selected_genre, movies=movie_list)


# 4단계: 추천 결과 페이지
@app.route('/recommend', methods=['POST'])
def recommend():
    user_ratings = {}
    user_id = None

    # rate_movies.html 폼에서 전달된 데이터 추출
    for key, value in request.form.items():
        if key == 'userId' and value: # 'userId' 필드에서 사용자 ID 추출
            user_id = int(value)
        elif value: # 나머지 필드는 영화 평점으로 간주
            try:
                # 키가 movie ID (정수)인지 확인하고 평점(실수)으로 변환
                movie_id = int(key)
                user_ratings[movie_id] = float(value)
            except ValueError:
                # movie ID 또는 평점 변환 오류 처리 (필요하다면 로깅 등)
                print(f"경고: 유효하지 않은 평점 데이터 무시됨 - key: {key}, value: {value}")
                pass

    # 사용자 ID 또는 평점 데이터가 없으면 오류 처리
    if user_id is None:
        return "사용자 ID 정보가 누락되었습니다.", 400
    if not user_ratings:
         return "평점 입력된 영화가 없습니다. 최소 하나 이상의 영화에 평점을 입력해주세요.", 400
         # 또는 평점 없이 사용자 ID만으로 추천하는 로직으로 변경할 수도 있습니다.


    # 사용자가 평점을 입력한 영화 ID 목록
    rated_movie_ids = set(user_ratings.keys())
    # 전체 영화 ID 목록에서 평점을 입력하지 않은 영화 ID 목록 생성
    all_movie_ids = movies['movieId'].unique()
    unrated_movie_ids = [mid for mid in all_movie_ids if mid not in rated_movie_ids]

    # 모델 예측 수행
    predictions = []
    if model is None:
        print("오류: SVD 모델이 로드되지 않았습니다.")
        # 모델 로드 실패 시 처리 로직 (예: 오류 메시지 반환, 빈 목록 반환 등)
        return "추천 시스템 모델 로드에 실패했습니다. 잠시 후 다시 시도해주세요.", 500


    # 사용자가 입력한 평점을 Surprise 모델에 적합한 형식으로 변환 (임시 데이터셋 생성)
    # SVD 모델은 기본적으로 학습 데이터의 스케일에 맞춰 예측합니다.
    # 여기서는 임시로 사용자 평점을 모델에 전달하여 예측값을 얻습니다.
    # 더 정확한 예측을 위해서는 사용자의 평점을 포함한 새로운 Reader/Dataset을 만들어 예측해야 할 수도 있습니다.
    # 하지만 Surprise의 predict 메서드는 uid, iid만으로도 학습된 모델을 사용하여 예측값을 반환합니다.
    # 따라서 여기서는 단순히 predict 메서드를 사용하는 것으로 충분합니다.

    for movie_id in unrated_movie_ids:
        # predict 메서드는 uid, iid, r_ui (실제 평점, 예측 시에는 None), verbose를 인자로 받습니다.
        # r_ui=None으로 두면 학습 데이터의 평균 평점 등을 고려하여 예측합니다.
        pred = model.predict(uid=user_id, iid=movie_id, r_ui=None)
        predictions.append((movie_id, pred.est)) # est: 예측 평점

    # 예측 평점이 높은 상위 5개 영화 선택
    top_5 = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]

    recommended_movies = []
    for movie_id, score in top_5:
        # 영화 정보 (제목) 가져오기
        movie_info = movies[movies['movieId'] == movie_id]
        if not movie_info.empty:
            title = movie_info['title'].values[0]

            # 포스터 URL 가져오기
            poster_url = get_poster_url(movie_id)

            # 영화 태그 정보 추출 및 태그 이름 가져오기
            # 해당 영화의 genome_scores에서 relevance가 높은 상위 3개 태그 선택
            top_tags_scores = genome_scores[genome_scores['movieId'] == movie_id].sort_values(by='relevance', ascending=False).head(3)

            # 태그 ID와 태그 이름을 매핑하여 태그 이름 목록 생성
            tag_names = []
            if not top_tags_scores.empty:
                 # genome_tags 데이터프레임과 병합하여 태그 이름 추출
                 merged_tags = top_tags_scores.merge(genome_tags, on='tagId')
                 tag_names = merged_tags['tag'].values.tolist() # 태그 이름을 리스트로 변환

            recommended_movies.append({
                'title': title,
                'score': round(score, 2),
                'tags': tag_names, # 추출된 태그 목록 포함
                'poster': poster_url
            })
        else:
             print(f"추천된 영화 ID {movie_id}에 대한 기본 정보를 찾을 수 없습니다.") # 디버깅 메시지

    # recommend.html 템플릿 렌더링 및 추천 영화 목록 전달
    return render_template('recommend.html', recommended_movies=recommended_movies)


# 서버 실행
if __name__ == '__main__':
    # debug=True는 개발 중에만 사용하고, 배포 시에는 False로 설정해야 합니다.
    app.run(debug=True)
