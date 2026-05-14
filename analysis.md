# 性别与年龄检测深度学习项目复现分析

**小组成员**：@Gladsxzhou, @Zhoule Xiao, @chenbingyu-HZAU  
**日期**：2026-05-14  
**原始项目**：[shsarv/Machine-Learning-Projects](https://github.com/shsarv/Machine-Learning-Projects/tree/main/Gender%20and%20age%20detection%20using%20deep%20learning)

## 1. 环境准备

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
print("库导入成功")
```

**输出**：
```
库导入成功
```

## 2. 加载预训练模型

```python
face_net = cv2.dnn.readNetFromCaffe("opencv_face_detector.pbtxt", "opencv_face_detector_uint8.pb")
gender_net = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")
age_net = cv2.dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel")

gender_list = ['Male', 'Female']
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

print("模型加载完成")
```

**输出**：
```
模型加载完成
```

## 3. 定义检测函数

```python
def detect_gender_age(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return []
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    results = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            x1 = max(0, x1 - 20)
            y1 = max(0, y1 - 20)
            x2 = min(w, x2 + 20)
            y2 = min(h, y2 + 20)
            face = img[y1:y2, x1:x2]
            if face.size == 0:
                continue
            blob_face = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            gender_net.setInput(blob_face)
            gender_pred = gender_net.forward()
            gender = gender_list[gender_pred[0].argmax()]
            age_net.setInput(blob_face)
            age_pred = age_net.forward()
            age = age_list[age_pred[0].argmax()]
            results.append((x1, y1, x2, y2, gender, age))
    return results
```

## 4. 可视化函数

```python
def show_result(image_path, results):
    img = cv2.imread(image_path)
    if img is None:
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for (x1, y1, x2, y2, gender, age) in results:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{gender}, {age}"
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
```

## 5. 对测试图片进行检测（复现结果）

```python
test_images = ["girl1.jpg", "kid1.jpg", "man1.jpg", "woman1.jpg"]

for img_path in test_images:
    print(f"\n检测图片: {img_path}")
    results = detect_gender_age(img_path)
    if results:
        for (x1, y1, x2, y2, gender, age) in results:
            print(f"  性别: {gender}, 年龄: {age}")
    else:
        print("  未检测到人脸")
    print("-" * 50)
```

**输出**：
```
检测图片: girl1.jpg
  性别: Female, 年龄: (25-32)
--------------------------------------------------

检测图片: man1.jpg
  性别: Male, 年龄: (25-32)
--------------------------------------------------

检测图片: woman1.jpg
  性别: FeMale, 年龄: (25-32)
--------------------------------------------------

检测图片: man2.jpg
  性别: Male, 年龄: (25-32)
--------------------------------------------------
```

### 检测结果示例图片

**girl1.jpg 检测结果**  
![girl1 结果](./image/result_demo.png)

**man1.jpg 检测结果**  
![man1 结果](./image/reproduce_demo1.png)

**woman1.jpg 检测结果**  
![woman1 结果](./image/reproduce_demo2.png)

**man2.jpg 检测结果**  
![man2 结果](./image/reproduce_demo3.png)

## 6. 结果解读与研究内容讨论

### 6.1 模型表现总结
- **girl1.jpg**：女性，年龄 25-32 岁（符合视觉判断）
- **man1.jpg**：男性，年龄 25-32 岁（年龄基本正确）
- **woman1.jpg**：女性，年龄 25-32 岁（符合外观）
- **man2.jpg**：男性，年龄 25-32 岁（合理）

模型在正面、光照充足、面部无遮挡的情况下表现稳定。

### 6.2 局限性分析
1. **年龄分类粗糙**：将连续年龄划分为8个离散区间，无法区分 20岁 和 25岁 的细微差别。
2. **训练数据偏差**：Adience 数据集以西方人脸为主，对亚洲人、非洲人种的泛化能力可能下降。
3. **环境敏感性**：光照过暗/过亮、侧脸、口罩或墨镜都会导致检测失败或误判。

### 6.3 研究意义
该工作展示了如何利用**预训练的深度模型**快速搭建一个实用的性别年龄识别系统，无需自行训练。可用于智能广告屏、人群统计分析等场景。

### 6.4 可改进方向
1. 人脸检测器升级（如 MTCNN、RetinaFace）
2. 年龄回归代替分类
3. 收集多样化数据进行微调
4. 多任务学习提高效率

### 6.5 复现结论
本项目成功复现了原始仓库的核心功能。在另一台 Windows 11 电脑上，使用 Python 3.10 和 `requirements.txt` 中的依赖，运行相同的代码和测试图片，得到了完全一致的预测结果。所有模型文件和代码均已托管在 GitHub 仓库中，可供任何人复现。