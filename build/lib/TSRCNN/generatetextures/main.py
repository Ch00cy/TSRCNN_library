import os.path

from TSRCNN.generatetextures.utils.generate import *	# utils 파일에 generate.py 의 모든 내용 가져오기

"""
Texture Synthesis using patch for rotated synthesized texture.
"""

def GeneratePatchTextures(img_path, before_limit=False, before_scale=120, new_scale=2, rotate_num=8, img_save=False, block_size=20, 
						  overlap=1.0/6, result_name="resultPatch", tolerance=0.1):
	
	"""
	Generate Patch Texture Synthesis.

	:param img_path: 이미지 경로 - 필수 입력
	:param before_limit: 초기 이미지 크기 제한 여부 (False) - 가로세로 길이가 모두 제한크기 이상 넘어갈 경우 제한크기로 크기조절
	:param before_scale: 초기 이미지 제한 크기 (120픽셀) - 클수록 합성 시간 오래 걸림
	:param new_scale: 결과 이미지 사이즈 얼마나 배로 늘릴것인가 (2배)
	:param rotate_num: 360도(중심 기준)/n 만큼 회전시켜 총 n 개의 회전 예제 텍스처를 생성 (n=8개)
	:param img_save: 결과를 저장할 것인가? (False)
	:param block_size: 블록 사이즈 픽셀 (20픽셀)
	:param overlap: 오버랩 사이즈 픽셀 (1.0/6)
	:param result_name: 저장할 결과 이름 ("resultPatch"+".png")
	:param tolerance: 허용오차 (0.1)

	:return: list <- 회전 예제 합성텍스처 [ndarray] 를 rotate_num 개 만큼 저장
	"""
	
	# overlap 설정
	if overlap > 0:	# 디폴트 overlap = 1/6
		overlap = int(block_size*overlap)	# 블록 사이즈의 1/6
	else:
		overlap = int(block_size/6.0)

	# 이미지 준비
	image = cv2.imread(img_path)	# 이미지 읽어오기
	if image is None:	#방어코드
		print("Image is failed!")
		
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255.0	# openCv : 컬러값 BGR -> RGB 변경 => 0~255 -> 0~1 값 변경
	H, W = image.shape[:2]  # 이미지 Height, Width

	# before image -> resize
	if before_limit==True:
		# 이미지 사이즈 w,h중 작은 것에 맞춰서 정사각형으로 크기조절
		if (H < before_scale or W < before_scale):
			if H>W:
				image = cv2.resize(image, (W, W))
			else:
				image = cv2.resize(image, (H, H))
		else:
			image = cv2.resize(image, (before_scale, before_scale))
	else:
		if H>W:
				image = cv2.resize(image, (W, W))
		else:
			image = cv2.resize(image, (H, H))

	outH, outW = int(new_scale * H), int(new_scale * W)  # 아웃풋 결과 : 이미지의 (scale)배로 키워줌

	# 회전 텍스처 합성
	results = RotateExImgs(image, rotate_num, block_size, overlap, tolerance, outH, outW)
	for i, result in enumerate(results):
		results[i] = (255 * result).astype(np.uint8)  # 최종 결과 텍스쳐 맵 -> 0~1, RGB 형태 => 원래대로로 돌림 (0~155 , BGR형태 , unit8)

	# Save
	if img_save==True:
		def createDirectory(directory):
			try:
				if not os.path.exists(directory):
					os.makedirs(directory)
			except OSError:
				print("Error: Failed to create the directory.")

		path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "SavedTextures")
		print(path)
		createDirectory(path)

		for i, result in enumerate(results):
			save_img = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
			cv2.imwrite(os.path.join(path,result_name + str(i) + ".png"), save_img)

	return results
