import torch
import torch.optim as optim
from torchvision.transforms.functional import normalize

from TSRCNN.deeptextures.utilsFolder import vgg19, gram_loss
from TSRCNN.deeptextures.utilsFolder.utils import set_seed, set_device, prep_img, to_pil, MEAN, STD
from TSRCNN.generatetextures.main import GeneratePatchTextures

import os
import numpy as np

"""
Texture Synthesis with Rotation using CNN
"""

def  generateTSR(image, rotateNum=8, outputScale=2, generateRotation = True, resultNameRotate="resultTextureImg", resultSave=False,
            Patch_before_limit=False, Patch_before_scale=120):
    """
    :param image: 이미지 경로 - 필수 입력 (ex)"절대경로")
	:param rotateNum: 360도(중심 기준)/n 만큼 회전시켜 총 n 개의 회전 예제 텍스처를 생성 (n=8개)
	:param outputScale: 결과 이미지 사이즈 얼마나 배로 늘릴것인가 (2배)
	:param generateRotation: 회전 예제 이미지를 생성할것인가? (True) - False 일 경우 기존 회전 예제 이미지 사용할 것을 대비.
	:param resultNameRotate: 저장할 결과 이름 ("resultTextureImg"+".png")
	:param resultSave: 결과를 저장할 것인가? (False)
	:param Patch_before_limit: 초기 이미지 크기 제한 여부 (Flase) - 가로세로 길이가 모두 제한크기 이상 넘어갈 경우 제한크기로 크기조절
	:param Patch_before_scale: 초기 이미지 제한 크기 (120픽셀) - 클수록 합성 시간 오래 걸림

	:return: PIL image - Texture Synthesis result
    """

    # Set device
    device = set_device()

    textures = []
    # add rotation #####################
    # n개의 회전된 새 텍스처 생성 해줄것인가?(True) / 이미 생성된 n개 텍스처 사용할 것인가?(False)
    if generateRotation:
        textures = GeneratePatchTextures(image, before_limit=Patch_before_limit, before_scale=Patch_before_scale,
                                       new_scale=outputScale, rotate_num=rotateNum, img_save=False)

    target = []
    for texture in textures:
        target.append(prep_img(texture).to(device))

    # 기본적인 설정과 전처리 ###################
    # 자주 변경하는 값
    n_iters = 1000  # 학습 반복할 정도

    layers = ["conv1_1", "pool1", "pool2", "pool3", "pool4"]
    layers_weigths = [1e9] * len(layers)    # 0.000000001 * 레이어 개수(5)

    # Set seed
    seed = np.random.randint(0, 100)  # 고정 안해놔야 항상 결과 바뀜
    set_seed(seed)

    # Init model
    model = vgg19().to(device)  # 레이어만 생성

    # 갱신할 새 노이즈 이미지
    # Init input image
    synth = torch.randint_like(target[0], 0, 256)
    synth = synth.div(255)
    synth = normalize(synth, MEAN, STD)
    synth.requires_grad_(True).to(device)

    # train 학습 부분
    # Set optimizer
    optimizer = optim.LBFGS(
        [synth],
        tolerance_grad=0.0,
        tolerance_change=0.0,
        line_search_fn="strong_wolfe",
    )

    # Get feature maps for the target texture
    target_activations = []
    for rTarget in target:
        target_activations.append([*model(rTarget, layers)])

    iter_ = 0
    while iter_ <= n_iters:
        def closure():
            nonlocal iter_

            optimizer.zero_grad()

            synth_activations = [*model(synth, layers)]

            assert len(synth_activations) == len(target_activations[0]) == len(layers_weigths)

            total_loss = 0
            
            # Compute loss for each activation
            for target_activation in target_activations:
                losses = []
                for activations in zip(synth_activations, target_activation, layers_weigths):
                    losses.append(gram_loss(*activations).unsqueeze(0))
                    total_loss += torch.cat(losses).sum()
        
            total_loss.backward()

            iter_ += 1

            return total_loss

        optimizer.step(closure)

    # tensor -> pil img
    result = to_pil(synth.squeeze())

    # Save generated texture
    if resultSave == True:
        def createDirectory(directory):
            try:
                if not os.path.exists(directory):
                    os.makedirs(directory)
            except OSError:
                print("Error: Failed to create the directory.")

        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "SavedTextures")
        print(path)
        createDirectory(path)

        result.save(os.path.join(path, resultNameRotate + ".png"), 'png')

    return result