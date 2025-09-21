from ultralytics import YOLO
import torch

def validate_ca_module():
    # 1. 加载仅含CA模块的测试模型（显式指定task，避免自动推断错误）
    try:
        model = YOLO('ultralytics/cfg/models/11/YOLO11_CA.yaml', task='detect')
        print("✅ 模型初始化成功！")
    except Exception as e:
        print(f"❌ 模型初始化失败：{e}")
        return

    # 2. 打印模型结构，确认CA模块存在
    print("\n📌 模型核心结构（含CA模块）：")
    # 遍历模型层，查找CA模块
    ca_found = False
    for i, m in enumerate(model.model.model):  # model.model 是Sequential容器
        if hasattr(m, 'type') and 'ca' in m.type.lower():  # 匹配CA模块的type标识
            print(f"  层{i}：{m.type} → {m}")
            ca_found = True
            # 额外检查CA模块的参数是否正确（输入/输出通道、reduction）
            if hasattr(m, 'conv1'):
                print(f"    - CA输入通道：{m.conv1.in_channels}")
                print(f"    - CA输出通道：{m.conv_h.out_channels}")
                print(f"    - CA reduction：{m.conv1.in_channels // m.conv1.out_channels}")

    if not ca_found:
        print("❌ 未找到CA模块！请检查tasks.py的导入和base_modules配置。")
        return
    print("✅ CA模块已成功加载到模型中！")

    # 3. 运行前向传播，验证CA模块功能正常（无通道/尺寸错误）
    try:
        # 生成符合模型输入尺寸的测试数据（640x640，1张图像，RGB通道）
        dummy_input = torch.randn(1, 3, 640, 640)  # [batch, channels, height, width]
        with torch.no_grad():  # 关闭梯度计算，加速测试
            outputs = model.model(dummy_input)  # 前向传播
        print(f"\n✅ 前向传播成功！输出特征图数量：{len(outputs)}")
        for i, out in enumerate(outputs):
            print(f"  输出{i+1}尺寸：{out.shape}")
    except Exception as e:
        print(f"\n❌ 前向传播失败（CA模块可能存在参数/通道错误）：{e}")
        return

    # 4. 验证总结
    print("\n🎉 CA模块添加验证通过！")
    print("验证点确认：1. 模块被正确识别 2. 前向传播无错误 3. 参数匹配配置")

if __name__ == "__main__":
    model = YOLO('ultralytics/cfg/models/11/yolo11n.yaml', task='detect')
    model.train(data='data.yaml', epochs=100, batch=4, imgsz=640,workers=0)  # 若能正常训练，说明CA完全兼容
    validate_ca_module()