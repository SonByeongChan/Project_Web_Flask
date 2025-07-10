# 1. 라이브러리 임포트
import pandas as pd
from sqlalchemy import create_engine
from surprise import Reader, Dataset, SVD, accuracy, dump
from surprise.model_selection import train_test_split

# 2. 데이터베이스 연결 설정
engine = create_engine('mysql+pymysql://user3:1234@192.168.2.236:3306/project')

# 3. 데이터 불러오기
ratings = pd.read_sql("SELECT userId, movieId, rating FROM ratings", engine)
movies = pd.read_sql("SELECT movieId, title, genres FROM movies", engine)
genome_scores = pd.read_sql("SELECT movieId, tagId, relevance FROM genome_scores", engine)
genome_tags = pd.read_sql("SELECT tagId, tag FROM genome_tags", engine)

# 4.  장르 영화만 필터링
drama_movies = movies[movies['genres'].str.contains('Sci-Fi', na=False)]
drama_movie_ids = drama_movies['movieId'].unique()
ratings = ratings[ratings['movieId'].isin(drama_movie_ids)]

# 5. Surprise용 Reader 설정
reader = Reader(rating_scale=(0.5, 5.0))

# 6. Surprise용 Dataset 생성
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# 7. train/test 데이터 분리
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# 8. SVD 모델 선언
model = SVD(n_factors=100, random_state=42)

# 9. 모델 학습
model.fit(trainset)

# 10. 테스트셋 평가 (RMSE 출력)
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
print(f"테스트 세트 RMSE: {rmse:.4f}")

# 11. 학습된 모델 저장
dump.dump('svd_Sci-Fi_model.pkl', algo=model)

print("SVD 모델 학습 및 저장 완료!")

# ==============================================================
# 추가 : 추천 결과 상위 5개 + 주요 3개 태그까지 뽑기
# ==============================================================

# (임시) 가상 사용자 입력 평점 예시
user_id = 9999
user_ratings = {
    1: 5.0,
    110: 4.5,
    260: 4.0,
    356: 5.0,
    500: 4.0
}

# 기존 ratings에 추가
new_ratings = pd.DataFrame({
    'userId': [user_id] * len(user_ratings),
    'movieId': list(user_ratings.keys()),
    'rating': list(user_ratings.values())
})
ratings = pd.concat([ratings, new_ratings], ignore_index=True)

# 다시 Surprise Dataset 준비
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()

# 모델은 이미 학습된 걸 불러왔다고 가정
# (지금 model 변수에 이미 학습 완료된 상태)

# 추천할 영화 찾기
all_movie_ids = movies['movieId'].unique()
rated_movie_ids = set(user_ratings.keys())
unrated_movie_ids = [mid for mid in all_movie_ids if mid not in rated_movie_ids]

# 예측
recommendations = []
for movie_id in unrated_movie_ids:
    pred = model.predict(uid=user_id, iid=movie_id)
    recommendations.append((movie_id, pred.est))

# 상위 5개 추천
top_5 = sorted(recommendations, key=lambda x: x[1], reverse=True)[:5]

# 추천 결과 출력
for movie_id, score in top_5:
    # 영화 제목 가져오기
    title = movies[movies['movieId'] == movie_id]['title'].values[0]
    
    # genome 점수 중 relevance 높은 상위 3개 태그 가져오기
    top_tags = genome_scores[genome_scores['movieId'] == movie_id].sort_values(
        by='relevance', ascending=False).head(3)
    
    # tagId를 tag 이름으로 변환
    top_tag_names = top_tags.merge(genome_tags, on='tagId')['tag'].values
    
    # 결과 출력
    print(f"추천 영화: {title} (예상 평점: {score:.2f})")
    print("주요 특징:", ", ".join(top_tag_names))
    print("-" * 60)
