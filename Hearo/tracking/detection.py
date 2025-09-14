#!/usr/bin/env python3

import cv2
import numpy as np
import tensorflow as tf
import argparse
import sys
import os
import logging
import time


# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebcamDetector:
    """
    웹캠 영상을 받아 TFLite 모델로 객체를 탐지하는 클래스
    """
    def __init__(self, model_path, camera_index, confidence_threshold, nms_threshold):
        self.camera_index = camera_index
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # TFLite 모델 로드 및 인터프리터 생성
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # 모델의 입력 및 출력 정보 가져오기
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # 입력 크기 정보 (예: [1, 640, 640, 3])
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]

        # 사람 탐지용 색상 (녹색)
        self.colors = [(0, 255, 0)]  # person 클래스는 녹색

        # 웹캠 초기화
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logger.error(f"오류: 카메라를 열 수 없습니다 (인덱스: {self.camera_index})")
            sys.exit(1)

    def preprocess_image(self, image):
        """입력 이미지를 모델에 맞게 전처리"""
        resized_image = cv2.resize(image, (self.input_width, self.input_height))
        # BGR -> RGB 변환, 차원 확장(batch), float32 변환
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(rgb_image, axis=0).astype(np.uint8)
        return input_data, resized_image

    def postprocess_output(self, outputs, original_shape, resized_shape):
        """양자화된 YOLO-X 모델의 출력값을 후처리하여 바운딩 박스, 신뢰도, 클래스 ID 추출"""
        try:
            print(f"postprocess_output 호출: outputs={len(outputs)}, original_shape={original_shape}, resized_shape={resized_shape}")
            
            # 양자화된 YOLO-X 출력 파싱
            if len(outputs) >= 3:
                # 양자화된 출력 디코딩
                boxes_quantized = outputs[0][0]  # [8400, 4] UINT8
                scores_quantized = outputs[1][0]  # [8400] UINT8  
                classes_quantized = outputs[2][0]  # [8400] UINT8
                
                # 양자화 파라미터로 디코딩 (모델에서 직접 가져오기)
                boxes_quantization = self.output_details[0]['quantization_parameters']
                scores_quantization = self.output_details[1]['quantization_parameters']
                
                boxes_scale = boxes_quantization['scales'][0]
                boxes_zero_point = boxes_quantization['zero_points'][0]
                scores_scale = scores_quantization['scales'][0]
                scores_zero_point = scores_quantization['zero_points'][0]
                
                # 디코딩
                boxes = (boxes_quantized.astype(np.float32) - boxes_zero_point) * boxes_scale
                scores = (scores_quantized.astype(np.float32) - scores_zero_point) * scores_scale
                classes = classes_quantized.astype(np.int32)
                
                # 사람 클래스만 필터링 (COCO dataset에서 person = 0)
                person_indices = np.where(classes == 0)[0]
                
                # 디버깅 정보 출력
                if not hasattr(self, '_debug_logged'):
                    print(f"전체 탐지 후보 수: {len(scores)}")
                    print(f"신뢰도 범위: {scores.min():.4f} ~ {scores.max():.4f}")
                    print(f"탐지된 사람 후보 수: {len(person_indices)}")
                    if len(person_indices) > 0:
                        person_scores = scores[person_indices]
                        print(f"사람 클래스 신뢰도 범위: {person_scores.min():.4f} ~ {person_scores.max():.4f}")
                        print(f"임계값 {self.confidence_threshold} 이상인 사람 수: {np.sum(person_scores >= self.confidence_threshold)}")
                    self._debug_logged = True
                
                if len(person_indices) == 0:
                    return [], [], []
                
                # 사람 클래스 탐지 결과만 추출
                person_boxes = boxes[person_indices]
                person_scores = scores[person_indices]
                person_classes = classes[person_indices]
                
                # 신뢰도 임계값 필터링
                score_mask = person_scores >= self.confidence_threshold
                person_boxes = person_boxes[score_mask]
                person_scores = person_scores[score_mask]
                person_classes = person_classes[score_mask]
                
                # print(f"신뢰도 필터링 후: {len(person_boxes)}개 탐지")
                
                if len(person_boxes) == 0:
                    return [], [], []
                
                # NMS 적용하여 중복 제거
                boxes_for_nms = person_boxes.copy()
                # YOLO 출력은 [x1, y1, x2, y2] 형식이므로 그대로 사용
                selected_indices = tf.image.non_max_suppression(
                    boxes_for_nms, person_scores, max_output_size=1, iou_threshold=self.nms_threshold
                )
                
                # NMS 결과를 원본 이미지 크기로 변환
                h_orig, w_orig = original_shape
                h_res, w_res = resized_shape
                
                final_boxes, final_scores, final_class_ids = [], [], []
                
                for index in selected_indices:
                    x1, y1, x2, y2 = person_boxes[index]
                    
                    # 좌표가 이미 정규화되어 있는지 확인하고 적절히 변환
                    if x1 > 1.0 or y1 > 1.0 or x2 > 1.0 or y2 > 1.0:
                        # 이미 픽셀 좌표인 경우
                        x1_orig = max(0, int(x1))
                        y1_orig = max(0, int(y1))
                        x2_orig = min(w_orig, int(x2))
                        y2_orig = min(h_orig, int(y2))
                    else:
                        # 정규화된 좌표인 경우
                        x1_orig = max(0, int(x1 * w_orig))
                        y1_orig = max(0, int(y1 * h_orig))
                        x2_orig = min(w_orig, int(x2 * w_orig))
                        y2_orig = min(h_orig, int(y2 * h_orig))
                    
                    # 유효한 바운딩 박스인지 확인
                    if x2_orig > x1_orig and y2_orig > y1_orig and x2_orig <= w_orig and y2_orig <= h_orig:
                        final_boxes.append([x1_orig, y1_orig, x2_orig, y2_orig])
                        final_scores.append(person_scores[index])
                        final_class_ids.append(person_classes[index])
                        print(f"탐지된 사람: confidence={person_scores[index]:.3f}, bbox=[{x1_orig}, {y1_orig}, {x2_orig}, {y2_orig}]")
                
                # print(f"최종 반환: {len(final_boxes)}개 탐지")
                return final_boxes, final_scores, final_class_ids
            
        except Exception as e:
            print(f"후처리 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return [], [], []
        
        return [], [], []

    def draw_detections(self, frame, boxes, scores, class_ids):
        """탐지된 객체들을 프레임에 그림"""
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            
            label = 'person'  # 사람만 탐지하므로 고정
            color = self.colors[0]  # 첫 번째 색상 (녹색) 사용
            
            # 바운딩 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 라벨 그리기
            label_text = f'{label}: {score:.2f}'
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
            cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def run(self):
        """메인 실행 루프"""
        logger.info("실시간 '사람' 탐지를 시작합니다. 종료하려면 'q'를 누르세요.")
        
        prev_time = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("오류: 프레임을 읽어올 수 없습니다.")
                break

            input_data, resized_frame = self.preprocess_image(frame)
            
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            # 양자화된 YOLO-X 모델의 모든 출력 가져오기
            outputs = []
            for output_detail in self.output_details:
                output_data = self.interpreter.get_tensor(output_detail['index'])
                outputs.append(output_data)

            boxes, scores, class_ids = self.postprocess_output(
                outputs, frame.shape[:2], resized_frame.shape[:2]
            )
            
            # 탐지 결과 로그 출력
            if len(boxes) > 0:
                print(f"탐지된 사람 수: {len(boxes)}")
                for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                    print(f"  Person {i+1}: confidence={score:.3f}, bbox={box}")
            else:
                print("탐지된 사람 없음")

            self.draw_detections(frame, boxes, scores, class_ids)
            
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Person Detection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('c'):
                self.confidence_threshold = max(0.1, self.confidence_threshold - 0.1)
                logger.info(f"신뢰도 임계값 변경: {self.confidence_threshold:.1f}")
            elif key == ord('v'):
                self.confidence_threshold = min(1.0, self.confidence_threshold + 0.1)
                logger.info(f"신뢰도 임계값 변경: {self.confidence_threshold:.1f}")
        
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    """명령행 인수 파싱 및 메인 로직 실행"""
    parser = argparse.ArgumentParser(
        description="Yolo-X TFLite 모델을 사용한 실시간 '사람' 탐지",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
키보드 단축키:
  ESC 또는 'q': 프로그램 종료
  'c': 신뢰도 임계값 감소 (0.1씩)
  'v': 신뢰도 임계값 증가 (0.1씩)
        """
    )
    parser.add_argument('--model', type=str, default='Yolo-X_w8a8.tflite', help='TFLite 모델 파일 경로')
    parser.add_argument('--camera', type=int, default=0, help='웹캠 인덱스')
    parser.add_argument('--confidence', type=float, default=0.5, help='신뢰도 임계값')
    parser.add_argument('--nms', type=float, default=0.45, help='NMS 임계값')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        logger.error(f"모델 파일을 찾을 수 없습니다: {args.model}")
        sys.exit(1)

    try:
        detector = WebcamDetector(
            model_path=args.model,
            camera_index=args.camera,
            confidence_threshold=args.confidence,
            nms_threshold=args.nms
        )
        detector.run()
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
