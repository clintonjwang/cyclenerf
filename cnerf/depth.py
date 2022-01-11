import torch
th = torch
nn = torch.nn
F = nn.functional

"""
MiDaS monocular depth estimator.
https://github.com/isl-org/MiDaS
"""

def load_depth_predictor(model_type="MiDaS_small"):
    # model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    midas = torch.hub.load("intel-isl/MiDaS", model_type).cuda().eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        midas.transform = midas_transforms.dpt_transform
    else:
        midas.transform = midas_transforms.small_transform

    return midas

def depth_predict(img, model):
    # url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    # urllib.request.urlretrieve(url, filename)
    # img = cv2.imread(filename)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = model.transform(img).cuda()

    with torch.no_grad():
        prediction = model(input_batch)

        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    return prediction
