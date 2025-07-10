from flask import Flask, render_template, request, redirect, url_for # session 추가 가능
import pandas as pd
import requests
from sqlalchemy import create_engine
from surprise import Dataset, Reader, SVD, dump
import os
import sys # 애플리케이션 종료를 위한 sys 모듈 추가

# Flask 앱 생성
app = Flask(__name__)

# TMDB API KEY
API_KEY = "2d3859f95d351c6539fc1ad0f734a384" # 실제 API 키로 변경하세요.

# --- 전역 변수 초기화 (데이터 및 모델) ---
# 데이터 로드 실패 시 None 상태를 유지하여 이후 라우트에서 체크 가능하도록 합니다.
engine = None
ratings = None
movies = None
genome_scores = None
genome_tags = None
links = None
model = None

# --- 데이터베이스 연결 및 데이터/모델 로드 함수 ---
def load_initial_data():
    """데이터베이스 연결 및 초기 데이터, 모델을 로드하는 함수."""
    global engine, ratings, movies, genome_scores, genome_tags, links, model

    print("--- 초기 설정 시작 ---")
    # 데이터베이스 연결 설정
    try:
        # 데이터베이스 연결 문자열을 실제 환경에 맞게 수정하세요.
        db_connection_str = 'mysql+pymysql://user3:1234@192.168.2.235:3306/project'
        print(f"데이터베이스 연결 시도: {db_connection_str}")
        engine = create_engine(db_connection_str)
        # 연결 테스트 (실패 시 예외 발생)
        with engine.connect() as connection:
            print("데이터베이스 연결 성공!")
    except Exception as e:
        print(f"오류: 데이터베이스 연결 실패: {e}", file=sys.stderr) # 오류를 stderr로 출력
        engine = None # 연결 실패 시 engine을 None으로 설정
        # 데이터베이스 연결 실패는 치명적이므로 여기서 로드 중단
        print("데이터베이스 연결 실패로 초기 로드 중단.")
        print("--- 초기 설정 종료 (오류 발생) ---")
        return False # 로드 실패 알림

    # 데이터 로드
    try:
        print("데이터 로드 시도...")
        # 각 테이블 이름 및 컬럼 이름이 데이터베이스와 정확히 일치하는지 확인하세요 (대소문자 구분).
        ratings = pd.read_sql("SELECT userId, movieId, rating FROM ratings", engine)
        movies = pd.read_sql("SELECT movieId, title, genres FROM movies", engine)
        genome_scores = pd.read_sql("SELECT movieId, tagId, relevance FROM genome_scores", engine)
        genome_tags = pd.read_sql("SELECT tagId, tag FROM genome_tags", engine)
        links = pd.read_sql("SELECT movieId, tmdbId FROM links", engine)
        print("데이터 로드 성공!")
        # 데이터가 제대로 로드되었는지 간단히 확인
        if ratings.empty or movies.empty or genome_scores.empty or genome_tags.empty or links.empty:
            print("경고: 일부 데이터프레임이 비어 있습니다. 데이터 소스를 확인하세요.")

    except Exception as e:
        print(f"오류: 데이터 로드 실패: {e}", file=sys.stderr) # 오류를 stderr로 출력
        # 데이터 로드 실패 시 해당 변수들을 None으로 설정
        ratings, movies, genome_scores, genome_tags, links = None, None, None, None, None
        print("데이터 로드 실패로 초기 로드 중단.")
        print("--- 초기 설정 종료 (오류 발생) ---")
        return False # 로드 실패 알림


    # SVD 모델 불러오기
    # 모델 파일 경로를 실제 모델 파일 경로로 수정하세요.
    model_path = r'E:\AI_KDT7\13. Flask\mini\lengend\models\svd_Action_model.pkl' # 예시 경로. 특정 장르가 아닌 범용 모델명으로 수정
    print(f"모델 로드 시도: {model_path}")
    try:
        # dump.load는 (모델 이름, 모델 객체) 튜플을 반환하므로 [1]로 모델 객체만 가져옵니다.
        model = dump.load(model_path)[1]
        print("SVD 모델 로드 성공!")
    except FileNotFoundError:
        print(f"오류: 모델 파일 '{model_path}'을(를) 찾을 수 없습니다.", file=sys.stderr)
        model = None # 모델 로드 실패 시 None으로 설정
    except Exception as e:
        print(f"오류: SVD 모델 로드 실패: {e}", file=sys.stderr)
        model = None # 모델 로드 실패 시 None으로 설정

    print("--- 초기 설정 완료 ---")
    return True # 로드 성공 알림

# --- 포스터 URL 가져오는 함수 ---
# 데이터 로드 실패 시 이 함수 내에서도 오류가 발생할 수 있으므로 링크 데이터 None 체크 추가
def get_poster_url(movie_id):
    global links # 전역 변수 사용 명시
    if links is None:
        print("경고: 링크 데이터가 로드되지 않아 포스터 URL을 가져올 수 없습니다.")
        return None

    try:
        # movieId에 해당하는 tmdbId 찾기
        link_info = links.loc[links['movieId'] == movie_id]
        if link_info.empty:
            print(f"경고: movieId {movie_id}에 해당하는 tmdbId를 링크 데이터에서 찾을 수 없습니다.")
            return None

        tmdb_id = link_info['tmdbId'].values[0]
        # tmdbId가 유효한 정수인지 확인
        if pd.isna(tmdb_id):
             print(f"경고: movieId {movie_id}의 tmdbId가 유효하지 않습니다: {tmdb_id}")
             return None
        tmdb_id = int(tmdb_id) # 정수로 변환

        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={API_KEY}&language=ko"
        # print(f"TMDB API 호출 시도: {url}") # 디버깅용
        response = requests.get(url, timeout=5) # 타임아웃 설정
        response.raise_for_status() # HTTP 오류 발생 시 예외 발생

        data = response.json()
        poster_path = data.get('poster_path')

        if poster_path:
            full_poster_url = f"https://image.themoviedb.org/t/p/w500{poster_path}"
            # print(f"포스터 URL 찾음: {full_poster_url}") # 디버깅용
            return full_poster_url
        else:
            print(f"경고: TMDB에서 movieId {movie_id} ({tmdb_id})의 poster_path를 찾을 수 없습니다.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"오류: TMDB API 호출 실패 (movieId: {movie_id}, tmdbId: {tmdb_id if 'tmdb_id' in locals() else 'N/A'}): {e}", file=sys.stderr)
        return None
    except ValueError as e:
        print(f"오류: tmdbId 변환 오류 (movieId: {movie_id}): {e}", file=sys.stderr)
        return None
    except Exception as e:
        # 다른 예상치 못한 예외 처리
        print(f"오류: get_poster_url 함수 실행 중 예외 발생 (movieId: {movie_id}): {e}", file=sys.stderr)
        return None


# --- 라우트 정의 ---

# 1단계: 사용자 ID 입력 페이지 (GET 요청만 처리)
@app.route('/', methods=['GET']) # POST 메소드 제거
def index():
    # 초기 데이터 로드 상태 확인 (선택 사항: 매 요청마다 할 필요는 없지만, 로드 실패 시 사용자에게 알릴 수는 있습니다.)
    # if not all([engine, ratings, movies, genome_scores, genome_tags, links, model]):
    #      return render_template('error.html', message="애플리케이션 초기 로드 중 오류가 발생했습니다. 서버 로그를 확인하세요.")

    print("index.html 페이지 렌더링 (GET 요청)")
    # GET 요청 시 index.html 렌더링 (사용자 ID 입력 폼 보여주기)
    # index.html 폼은 이제 직접 /select_genre로 POST 요청을 보냅니다.
    return render_template('index.html')

# 참고: index 라우트의 POST 처리는 이제 index.html 폼이 직접 /select_genre로 POST하면서 필요 없어졌습니다.
# 만약 index.html에서 / 로 POST 요청을 보내고 싶다면, 해당 POST 처리 로직을 복원해야 합니다.
# 현재는 장르 선택 페이지로 바로 연결하는 흐름을 위해 이렇게 수정합니다.


# 2단계: 장르 선택 페이지 (사용자 ID를 받아 장르 목록 표시)
# index.html 폼에서 POST 요청을 받습니다.
@app.route('/select_genre', methods=['POST'])
def select_genre_page():
    # 초기 데이터 로드 상태 확인
    global movies # movies 데이터를 사용하므로 체크
    if movies is None:
         print("오류: 영화 데이터(movies)가 로드되지 않아 장르 선택 페이지를 표시할 수 없습니다.", file=sys.stderr)
         # 데이터 로드 실패 시 사용자에게 보여줄 오류 메시지 템플릿 렌더링 또는 단순 메시지 반환
         return "애플리케이션 초기 로드 중 오류가 발생했습니다 (영화 데이터 누락). 서버 로그를 확인하세요.", 500

    # index.html 폼에서 전달된 userId 가져옴 (POST 요청의 폼 데이터)
    user_id_str = request.form.get('userId')

    print(f"/select_genre POST 요청 수신. 폼 데이터:", request.form)
    print(f"가져온 userId_str 값: '{user_id_str}'")

    if not user_id_str:
        # 사용자 ID가 없으면 오류 메시지 표시 또는 index 페이지로 리다이렉트 (잘못된 접근)
        print("경고: /select_genre 요청에 사용자 ID 누락.", file=sys.stderr)
        # index 페이지로 돌아가 사용자 ID 재입력 요청
        return redirect(url_for('index', error="사용자 ID가 누락되었습니다. 다시 입력해주세요.")) # 에러 메시지를 쿼리 파라미터로 전달 예시

    try:
        user_id = int(user_id_str)
        print(f"유효한 사용자 ID: {user_id}")
    except ValueError:
        # 사용자 ID 형식이 유효하지 않으면 오류 메시지 표시
        print(f"경고: /select_genre 요청에 유효하지 않은 사용자 ID 형식 입력됨: '{user_id_str}'", file=sys.stderr)
        return "유효하지 않은 사용자 ID 형식입니다. 숫자를 입력해주세요.", 400 # 400 Bad Request

    # 장르 목록 추출 또는 하드코딩
    genres = ['Action', 'Animation', 'Comedy', 'Drama', 'Sci-Fi'] # 하드코딩된 5가지 장르 사용


    print("select_genre.html 템플릿 렌더링.")
    # select_genre.html 템플릿에 사용자 ID와 장르 목록 전달
    # 사용자 ID는 다음 단계(/rate_movies)로 전달하기 위해 템플릿에 포함시켜야 합니다 (hidden input 사용).
    return render_template('select_genre.html', user_id=user_id, genres=genres)




# 3단계: 선택된 장르별 영화 평점 입력 페이지 (select_genre.html 폼에서 POST 요청을 받음)
@app.route('/rate_movies', methods=['POST']) # POST 요청으로 장르와 userId를 받도록 변경
def rate_movies_page(): # 라우트 함수 이름 변경 (rate_movies -> rate_movies_page)
    # 이 라우트는 select_genre.html 폼에서 POST 요청으로
    # 사용자 ID (hidden input)와 선택된 장르 (예: name="selectedGenre")를 받습니다.
    # ... (나머지 rate_movies_page 함수 코드는 이전에 합쳐드린 코드와 동일하게 유지) ...
    # 해당 코드가 user_id = request.form.get('userId') 와 selected_genre = request.form.get('selectedGenre') 를 사용해야 합니다.

    # --- 이전에 합쳐드린 rate_movies_page 함수 코드를 여기에 붙여넣으세요 ---
    # 예시 코드 시작:
    # 초기 데이터 로드 상태 확인
    global movies, ratings
    if movies is None or ratings is None:
         print("오류: 영화 또는 평점 데이터가 로드되지 않아 평점 입력 페이지를 표시할 수 없습니다.", file=sys.stderr)
         return "애플리케이션 초기 로드 중 오류가 발생했습니다 (데이터 누락). 서버 로그를 확인하세요.", 500

    user_id_str = request.form.get('userId')
    selected_genre = request.form.get('selectedGenre')

    print(f"/rate_movies POST 요청 수신. 폼 데이터:", request.form)
    print(f"가져온 userId_str: '{user_id_str}', 가져온 selected_genre: '{selected_genre}'")

    if not user_id_str or not selected_genre:
        print("경고: /rate_movies 요청에 사용자 ID 또는 장르 누락.", file=sys.stderr)
        return "잘못된 접근입니다. 사용자 ID와 장르 정보가 필요합니다.", 400

    try:
        user_id = int(user_id_str)
        print(f"유효한 사용자 ID: {user_id}, 선택 장르: {selected_genre}")
    except ValueError:
        print(f"경고: /rate_movies 요청에 유효하지 않은 사용자 ID 형식 입력됨: '{user_id_str}'", file=sys.stderr)
        return "유효하지 않은 사용자 ID 형식입니다. 숫자를 입력해주세요.", 400

    try:
        if 'genres' not in movies.columns or movies['genres'] is None:
             print("오류: movies 데이터프레임에 'genres' 컬럼이 없거나 유효하지 않습니다.", file=sys.stderr)
             return "영화 장르 정보를 찾을 수 없습니다. 서버 로그를 확인하세요.", 500

        filtered_movies = movies[movies['genres'].astype(str).str.contains(selected_genre, na=False)].copy()
        if filtered_movies.empty:
             print(f"경고: 선택된 장르 '{selected_genre}'에 해당하는 영화가 movies 데이터프레임에 없습니다.")
             return f"선택하신 장르 '{selected_genre}'에 해당하는 영화가 없습니다. 다른 장르를 선택해주세요.", 404

    except Exception as e:
        print(f"오류: 장르 필터링 중 오류 발생: {e}", file=sys.stderr)
        return "영화 목록 필터링 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.", 500

    try:
        if ratings is None or ratings.empty:
            print("경고: 평점 데이터(ratings)가 비어 있어 평점 개수 상위 영화를 선택할 수 없습니다.")
            top_rated_movies_ids = filtered_movies['movieId'].head(10).tolist()
        else:
            if 'movieId' not in ratings.columns:
                 print("오류: ratings 데이터프레임에 'movieId' 컬럼이 없습니다.", file=sys.stderr)
                 return "평점 데이터 구조에 문제가 있습니다. 서버 로그를 확인하세요.", 500

            popular_movies_in_genre = ratings[ratings['movieId'].isin(filtered_movies['movieId'])]
            if popular_movies_in_genre.empty:
                 print(f"경고: 선택된 장르 '{selected_genre}'의 영화 중 평점 데이터가 있는 영화가 없습니다.")
                 top_rated_movies_ids = filtered_movies['movieId'].head(10).tolist()
            else:
                top_rated_movies_ids = (
                    popular_movies_in_genre
                    .groupby('movieId')
                    .size()
                    .sort_values(ascending=False)
                    .head(10)
                    .index.tolist()
                )

    except Exception as e:
        print(f"오류: 평점 개수 기준으로 영화 선택 중 오류 발생: {e}", file=sys.stderr)
        return "영화 인기 순위 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.", 500

    try:
        if not top_rated_movies_ids:
             print(f"경고: 평점 개수 상위 영화 목록이 비어 있습니다. 필터링된 영화 중 처음 10개를 선택합니다.")
             selected_movies = filtered_movies.head(10)
        else:
             selected_movies = filtered_movies[filtered_movies['movieId'].isin(top_rated_movies_ids)].head(10)


        movie_list = []
        if selected_movies.empty:
             print(f"경고: 최종 선택된 영화 목록이 비어 있습니다. (장르: {selected_genre})")
             return f"선택하신 장르 '{selected_genre}'에 대해 보여줄 영화 목록을 찾을 수 없습니다.", 404

        for index, movie in selected_movies.iterrows():
            movie_id = movie['movieId']
            title = movie['title']
            poster_url = get_poster_url(movie_id)
            movie_list.append({
                'movieId': movie_id,
                'title': title,
                'poster': poster_url
            })

    except Exception as e:
        print(f"오류: 최종 영화 목록 생성 중 오류 발생: {e}", file=sys.stderr)
        return "영화 목록 준비 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.", 500

    print("rate_movies.html 템플릿 렌더링.")
    return render_template('rate_movies.html', user_id=user_id, genre=selected_genre, movies=movie_list)

    # 예시 코드 끝. 실제 rate_movies_page 함수의 전체 코드는 이 위에 붙여넣으세요.


# 4단계: 추천 결과 페이지 (영화 평점 입력 폼에서 POST 요청을 받음)
@app.route('/recommend', methods=['POST'])
def recommend():
    # 이전에 합쳐드린 recommend 함수의 전체 코드를 여기에 붙여넣으세요.
    # 해당 코드는 user_id = request.form.get('userId') 와 user_ratings 추출 로직을 사용해야 합니다.
    # ... (이전에 합쳐드린 recommend 함수 코드 유지) ...
    # 예시 코드 시작:
    global movies, genome_scores, genome_tags, model
    if movies is None or genome_scores is None or genome_tags is None or model is None:
         print("오류: 추천 시스템 실행에 필요한 데이터 또는 모델이 로드되지 않았습니다.", file=sys.stderr)
         return "애플리케이션 초기 로드 중 오류가 발생했습니다 (추천 시스템 데이터 누락). 서버 로그를 확인하세요.", 500

    user_ratings = {}
    user_id = None

    print(f"/recommend POST 요청 수신. 폼 데이터:", request.form)

    user_id_str = request.form.get('userId')

    if not user_id_str:
        print("경고: /recommend 요청에 사용자 ID 누락.", file=sys.stderr)
        return "사용자 ID 정보가 누락되었습니다. 다시 시도해주세요.", 400

    try:
        user_id = int(user_id_str)
        print(f"추출된 사용자 ID: {user_id}")
    except ValueError:
        print(f"경고: /recommend 요청에 유효하지 않은 사용자 ID 형식 입력됨: '{user_id_str}'", file=sys.stderr)
        return "사용자 ID 형식이 올바르지 않습니다. 숫자를 입력해주세요.", 400

    try:
        for key, value in request.form.items():
            if key == 'userId':
                continue

            if value:
                 try:
                     movie_id = int(key)
                     rating_value = float(value)
                     if 0.5 <= rating_value <= 5.0:
                         user_ratings[movie_id] = rating_value
                     else:
                         print(f"경고: 유효하지 않은 평점 값 무시됨 (범위 벗어남): key: {key}, value: {value}")
                 except ValueError:
                     print(f"경고: 유효하지 않은 평점 데이터 무시됨 (변환 오류): key: {key}, value: {value}", file=sys.stderr)
                     pass
    except Exception as e:
        print(f"오류: 평점 데이터 추출 중 예상치 못한 오류 발생: {e}", file=sys.stderr)
        return "평점 데이터 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.", 500

    if not user_ratings:
         print("경고: 평점 입력된 영화가 없습니다. 최소 하나 이상의 영화에 평점을 입력해야 추천 가능.", file=sys.stderr)
         return "평점 입력된 영화가 없습니다. 최소 하나 이상의 영화에 평점을 입력해주세요.", 400

    rated_movie_ids = set(user_ratings.keys())

    try:
        if movies is None or 'movieId' not in movies.columns:
             print("오류: movies 데이터프레임이 로드되지 않았거나 'movieId' 컬럼이 없습니다.", file=sys.stderr)
             return "영화 데이터 로드 상태에 문제가 있습니다. 서버 로그를 확인하세요.", 500

        all_movie_ids = movies['movieId'].unique()
        unrated_movie_ids = [mid for mid in all_movie_ids if mid not in rated_movie_ids]

        if not unrated_movie_ids:
             print("경고: 추천할 영화가 없습니다 (모든 영화 평점 입력 또는 데이터 부족).", file=sys.stderr)
             return "더 이상 추천할 새로운 영화가 없습니다. 더 많은 영화에 평점을 입력하거나 다른 장르를 시도해 보세요.", 200

    except Exception as e:
        print(f"오류: 평점 입력 영화 및 미평점 영화 목록 생성 중 오류 발생: {e}", file=sys.stderr)
        return "영화 목록 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.", 500

    predictions = []
    try:
        if model is None:
            print("오류: SVD 모델이 로드되지 않아 예측을 수행할 수 없습니다.", file=sys.stderr)
            return "추천 시스템 모델이 준비되지 않았습니다. 잠시 후 다시 시도해주세요.", 500

        for movie_id in unrated_movie_ids:
            try:
                pred = model.predict(uid=user_id, iid=movie_id, r_ui=None)
                predictions.append((movie_id, pred.est))
            except ValueError as ve:
                 print(f"경고: 모델 예측 중 ValueError 발생 (uid:{user_id}, iid:{movie_id}): {ve}. 해당 예측 건너뜁니다.", file=sys.stderr)
            except Exception as e:
                 print(f"경고: 모델 예측 중 예상치 못한 오류 발생 (uid:{user_id}, iid:{movie_id}): {e}. 해당 예측 건너뜁니다.", file=sys.stderr)

    except Exception as e:
        print(f"오류: 모델 예측 루프 실행 중 오류 발생: {e}", file=sys.stderr)
        return "영화 추천 예측 과정에서 오류가 발생했습니다. 잠시 후 다시 시도해주세요.", 500

    top_recommendations_count = 3
    recommended_movies = []
    try:
        if not predictions:
            print("경고: 모델 예측 결과가 없습니다.", file=sys.stderr)
            return "추천 결과를 생성할 수 없습니다. 더 많은 영화에 평점을 입력하거나 다른 장르를 시도해 보세요.", 200

        top_recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_recommendations_count]

        if genome_scores is None or genome_tags is None:
             print("오류: 영화 상세 정보 관련 데이터 로드에 실패했습니다.", file=sys.stderr)
             return "영화 상세 정보를 불러오는 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.", 500

        for movie_id, score in top_recommendations:
            try:
                movie_info = movies[movies['movieId'] == movie_id]
                if not movie_info.empty:
                    title = movie_info['title'].values[0]
                    poster_url = get_poster_url(movie_id)

                    if genome_scores is not None and genome_tags is not None:
                         top_tags_scores = genome_scores[genome_scores['movieId'] == movie_id].sort_values(by='relevance', ascending=False).head(3)
                         tag_names = []
                         if not top_tags_scores.empty:
                              merged_tags = top_tags_scores.merge(genome_tags, on='tagId')
                              tag_names = merged_tags['tag'].values.tolist()
                    else:
                         tag_names = ["태그 정보 없음"]

                    recommended_movies.append({
                        'title': title,
                        'score': round(score, 2),
                        'tags': tag_names,
                        'poster': poster_url
                    })
                else:
                     print(f"경고: 추천된 영화 ID {movie_id}에 대한 기본 정보를 찾을 수 없습니다. 추천 목록에서 제외합니다.", file=sys.stderr)
            except Exception as e:
                 print(f"오류: 추천 영화 ID {movie_id} 정보 처리 중 오류 발생: {e}. 해당 영화 건너뜁니다.", file=sys.stderr)

    except Exception as e:
        print(f"오류: 추천 결과 목록 생성 중 오류 발생: {e}", file=sys.stderr)
        return "추천 결과 정보를 준비하는 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.", 500

    if not recommended_movies:
        print("경고: 추천 결과 목록이 비어 있습니다.", file=sys.stderr)
        return "죄송합니다. 추천할 영화를 찾지 못했습니다.", 200


    print("recommend.html 템플릿 렌더링.")
    return render_template('recommend.html', recommended_movies=recommended_movies, user_id=user_id)
    # 예시 코드 끝. 실제 recommend 함수의 전체 코드는 이 위에 붙여넣으세요.


# 서버 실행 (이 부분은 그대로 유지합니다)
if __name__ == '__main__':
    load_success = load_initial_data()

    if not load_success:
        print("\n애플리케이션 초기 로드에 실패했습니다. Flask 앱이 실행되지 않거나 정상 작동하지 않을 수 있습니다.", file=sys.stderr)
        # sys.exit(1) # 필요하다면 주석 해제

    else:
         print("\n애플리케이션 실행 준비 완료.")

    print("Flask 앱 실행 중...")
    app.run(debug=True) # debug=True로 설정하여 오류 상세 정보 확인
