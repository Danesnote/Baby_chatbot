import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import preprocessing

#의도 분류 모델 모듈
class IntentModel:
    def __init__(self, model_name, proprocess):
        #의도 클래스별 레이블
        self.labels={0:'주문_취소_확인', 1 : '배송_비용_질문', 2 : '주문_변경_확인', 3 : '제품_방법_요청',
        4: '교환|반품|환불_시간_확인', 5 : '배송_방법_비교', 6 : '배송_오류_확인', 7 : '배송_지역_요청',
        8: '주문_변경_질문', 9: '제품_추천_요청', 10: '주문_취소_요청', 11: '교환|반품|환불_일반_질문',
        12: '교환|반품|환불_방법_요청', 13: '교환|반품|환불_방법_질문', 14: '제품_구매_질문', 15: '제품_불량_확인'}
        
        #의도 분류 모델 불러오기
        self.model=load_model(model_name)
        
        #챗봇 Preprocess 객체
        self.p=proprocess
        
    #의도 클래스 예측
    def predict_class(self, query):
        #형태소 분석
        pos=self.p.pos(query)
        
        #문장 내 키워드 추출(불용어 제거)
        keywords=self.p.get_keywords(pos, without_tag=True)
        sequences=[self.p.get_wordidx_sequence(keywords)]
        
        #단어 시퀀스 벡터 크기
        MAX_SEQ_LEN=15
        
        #패딩 처리
        padded_seqs=preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')
        
        predict=self.model.predict(padded_seqs)
        predict_class=tf.math.argmax(predict, axis=1)
        return predict_class.numpy()[0]