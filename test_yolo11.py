import cv2
import os
import time
from ultralytics import YOLO
import torch
import ultralytics

def test_yolo11():
    """测试YOLO11模型（仅本地图片+视频）"""

    # ====================== 1. 检查运行环境 ======================
    # 打印导入的 ultralytics 模块所在路径11111
    print("ultralytics 模块路径：", ultralytics.__file__)
    print("=" * 50)
    print("正在检查运行环境...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    if device == 'cuda':
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"PyTorch版本: {torch.__version__}")
    else:
        print("警告: 未使用GPU加速，预测速度可能较慢")
    print("=" * 50)

    # ====================== 2. 加载YOLO11模型 ======================
    print("\n正在加载YOLO11模型...")
    model_size = "n"  # 可选：n/s/m/l/x（模型越小速度越快，精度略低）
    model = YOLO(f"yolo11{model_size}.pt").to(device)
    print("\n模型结构信息:")
    model.info()

    # ====================== 3. 准备本地测试图像 ======================
    print("\n" + "=" * 50)
    print("准备本地测试图像...")
    # 替换为你实际的本地图片路径（确保文件存在）
    test_image_paths = [
        os.path.join(os.path.dirname(__file__), "ultralytics", "assets", "bus.jpg"),
        os.path.join(os.path.dirname(__file__), "ultralytics", "assets", "zidane.jpg"),
        # 可继续添加更多本地图片路径 ↓
        # os.path.join(...)
    ]

    valid_image_paths = []
    for path in test_image_paths:
        if os.path.exists(path):
            valid_image_paths.append(path)
        else:
            print(f"警告：图像 {path} 不存在，已跳过")
    if not valid_image_paths:
        print("错误: 未找到任何有效本地测试图像")
        return
    print(f"找到 {len(valid_image_paths)} 张有效本地测试图像")

    # ====================== 4. 准备本地测试视频 ======================
    print("\n" + "=" * 50)
    print("准备本地测试视频...")
    # 替换为你实际的本地视频路径（确保文件存在，如MP4格式）
    test_video_path = os.path.join(
        os.path.dirname(__file__), "ultralytics", "assets", "8333.mp4"
    )
    valid_video_path = test_video_path if os.path.exists(test_video_path) else None
    if valid_video_path:
        print(f"找到测试视频: {test_video_path}")
    else:
        print(f"警告：视频 {test_video_path} 不存在，将跳过视频测试")

    # ====================== 5. 处理本地图像预测 ======================
    print("\n" + "=" * 50)
    print("开始图像预测...")
    output_dir = "ultralytics/assets/yolo11_test_results"
    os.makedirs(output_dir, exist_ok=True)

    for i, img_path in enumerate(valid_image_paths, 1):
        print(f"\n处理第 {i}/{len(valid_image_paths)} 张图像: {os.path.basename(img_path)}")
        start_time = time.time()
        results = model(img_path, conf=0.25)  # 置信度阈值0.25
        elapsed = time.time() - start_time
        print(f"预测耗时: {elapsed:.2f} 秒")

        for result in results:
            print(f"检测到 {len(result.boxes)} 个目标:")
            for box in result.boxes:
                class_name = result.names[int(box.cls)]
                confidence = float(box.conf)
                print(f"  - {class_name}: {confidence:.2f}")

            # 保存带检测框的图像
            save_path = os.path.join(
                output_dir, f"img_result_{i}_{os.path.basename(img_path).split('.')[0]}.jpg"
            )
            result.save(save_path)
            print(f"图像结果已保存至: {save_path}")

            # 显示结果（按任意键关闭）
            try:
                img = cv2.imread(save_path)
                img = cv2.resize(img, (800, int(img.shape[0] * 800 / img.shape[1])))
                cv2.imshow(f"YOLO11 图像检测 ({i})", img)
                print("按任意键继续...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"显示图像出错: {str(e)}")

    # ====================== 6. 处理本地视频预测（若存在） ======================
    if valid_video_path:
        print("\n" + "=" * 50)
        print("开始视频预测...")
        cap = cv2.VideoCapture(valid_video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_save = os.path.join(output_dir, "video_result.mp4")
        out = cv2.VideoWriter(video_save, fourcc, fps, (width, height))

        frame_count = 0
        start_time = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            results = model(frame, conf=0.25)
            annotated_frame = results[0].plot()

            out.write(annotated_frame)
            cv2.imshow("YOLO11 视频检测", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 按'q'提前退出
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        elapsed = time.time() - start_time
        print(f"视频预测完成，共处理 {frame_count} 帧，耗时: {elapsed:.2f} 秒")
        print(f"视频结果已保存至: {video_save}")

    # ====================== 7. 测试完成 ======================
    print("\n" + "=" * 50)
    print("YOLO11测试完成!")
    print(f"所有结果保存至: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    test_yolo11()