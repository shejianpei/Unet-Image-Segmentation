import cv2
import torch
from torchvision import transforms
from PIL import Image
import time
from net.unet import unett

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
PATH = "./UNet_model.pth"


def imgtest(imgpath):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    model = unett()

    model.eval()
    with torch.no_grad():
        img_r = cv2.imread(imgpath)  # 读入图片
        img = Image.open(imgpath)
        [width1, height1, _] = img_r.shape
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)
        print(img_tensor.shape)

        start = time.time()
        model = model.to("cuda")
        img_tensor = img_tensor.to("cuda")
        y_pred = model(img_tensor)
        end = time.time()
        print(end-start)

        heat_map = torch.argmax(y_pred.squeeze().permute(1, 2, 0), axis=2).detach().cpu().numpy()
        [height, width] = heat_map.shape
        dst = cv2.resize(img_r, (width, height), interpolation=cv2.INTER_LINEAR)

        start = time.time()
        for i in range(width - 1):  # 遍历生成的numpy数组
            for j in range(height - 1):
                if heat_map[j, i] > 0:
                    pass
                else:
                    dst = cv2.circle(dst, (i, j), 1, (0, 0, 0), -1)  # 根据坐标在原图画点
        # cv2.namedWindow("result")
        # cv2.imshow("result", dst)
        cv2.imwrite("./res.jpg", cv2.resize(dst, (height1, width1)))
        end = time.time()
        print(end - start)
        # cv2.waitKey(0)
